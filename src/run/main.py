import os
import sys
import json
import random
import argparse
import traceback
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data.load_questions import load_data, create_question
from src.utils.logging import Logger, _atomic_json_dump, make_log_func
from src.models.moa import SYNTHESIZE_PROMPT, run_moa
 
# ==============================
# Layer Configuration
# ==============================

AGENTS = [
    {"provider": "mistral", "model": "mistral-small-2506", "temperature": 0.7},
    {"provider": "mistral", "model": "ministral-14b-2512", "temperature": 0.7},
    {"provider": "mistral", "model": "ministral-8b-2512", "temperature": 0.7},
    {"provider": "mistral", "model": "ministral-3b-2512", "temperature": 0.7},
]

PROPOSER_LAYERS = [
    AGENTS,  # Layer 1
    AGENTS,  # Layer 2
]

# Layer 3
AGGREGATOR = {"provider": "mistral", "model": "mistral-large-2512", "temperature": 0.0, "max_tokens": 2048}

# ==============================
# Output Management
# ==============================

# Thread-safe lock for results and file operations
results_lock = threading.Lock()
file_lock = threading.Lock()


# ==============================
# Main Execution Loop
# ==============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='medqa', type=str)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--multithread', action='store_true', help='Enable multithreaded execution.')
    args = parser.parse_args()

    file_name = f"{args.dataset}_{args.num_samples}{'_' + str(args.seed) if args.seed is not None else ''}"
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    if args.multithread:
        # Create per-sample logs directory for multithreaded mode
        sample_logs_dir = os.path.join('logs', f"{file_name}_samples")
        os.makedirs(sample_logs_dir, exist_ok=True)
    else:
        # Single-threaded: redirect stdout to terminal + single file
        main_log_path = f"logs/{file_name}.log"
        sys.stdout = Logger(main_log_path)

    # Output + resume paths
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{file_name}.json")
    progress_path = os.path.join('logs', f"{file_name}.progress.json")

    # Load previous results if present (resume)
    results = []
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            if not isinstance(results, list):
                print(f"[WARN] Existing output is not a list. Starting fresh: {output_path}")
                results = []
        except Exception as e:
            print(f"[WARN] Failed to load existing output ({output_path}): {e}. Starting fresh.")
            results = []

    start_no = len(results)
    if start_no > 0:
        print(f"[INFO] Resuming from sample index {start_no} (already saved {start_no} results).")

    # Keep a tiny progress file too (helpful if output gets edited)
    _atomic_json_dump({"next_index": start_no}, progress_path)

    test_qa, _ = load_data(args.dataset)

    # Randomly select test samples for quicker testing (remove this part for full eval)
    if args.seed is not None:
        random.seed(args.seed)
        
    if args.num_samples is not None and args.num_samples < len(test_qa):
        test_qa = random.sample(test_qa, args.num_samples)

    # Prepare samples to process (skip already completed)
    samples_to_process = list(enumerate(test_qa[start_no:], start=start_no))
    if args.num_samples is not None:
        samples_to_process = [(no, s) for no, s in samples_to_process if no < args.num_samples]
        
    # Run samples
    if args.multithread:
        # ==================== MULTITHREADED MODE ====================
        def process_sample(no, sample):
            """Process a single sample - thread worker function"""
            sample_log_path = os.path.join(sample_logs_dir, f"sample_{no:04d}.log")
            log_lines = []
            log = make_log_func(multithread=True, log_lines=log_lines)
            
            try:
                log(f"[INFO] Processing sample {no}")
                question, img_path = create_question(sample, args.dataset)
                
                final_decision = run_moa(question, PROPOSER_LAYERS, AGGREGATOR, SYNTHESIZE_PROMPT, return_intermediate=False, log=log)
                
                if args.dataset == 'medqa':
                    result = {
                        'index': no,
                        'question': question,
                        'label': sample['answer_idx'],
                        'answer': sample['answer'],
                        'options': sample['options'],
                        'response': final_decision,
                    }
                else:
                    result = {
                        'index': no,
                        'question': question,
                        'response': final_decision,
                    }

                log(f"\n[INFO] Sample {no} completed successfully")
                
                # Write log to file
                with open(sample_log_path, 'w') as f:
                    f.write('\n'.join(log_lines))
                
                return no, result, None

            except Exception as e:
                error_info = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                log(f"[ERROR] Exception at sample {no}: {error_info}")
                
                # Write log to file even on error
                with open(sample_log_path, 'w') as f:
                    f.write('\n'.join(log_lines))
                
                return no, None, error_info

        # Thread-safe function to save results
        def save_results_thread_safe():
            with file_lock:
                _atomic_json_dump(results, output_path)
                _atomic_json_dump({"next_index": len(results)}, progress_path)

        # Main multithreaded loop
        errors = []
        shutdown_flag = threading.Event()

        NUM_WORKERS = 20  # Fixed number of worker threads
        print(f"[INFO] Starting processing with {NUM_WORKERS} worker threads...")
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(process_sample, no, sample): (no, sample) 
                for no, sample in samples_to_process
            }
            
            # Progress bar for completed tasks
            pbar = tqdm(total=len(test_qa), initial=start_no, desc="Processing samples")
            
            try:
                for future in as_completed(future_to_sample):
                    if shutdown_flag.is_set():
                        break
                        
                    no, result, error = future.result()
                    
                    if error:
                        errors.append((no, error))
                    else:
                        # Thread-safe result storage
                        with results_lock:
                            results.append(result)
                            # Sort by index to maintain order
                            results.sort(key=lambda x: x.get('index', 0))
                        
                        # Save progress periodically (thread-safe)
                        save_results_thread_safe()
                    
                    pbar.update(1)
                    
            except KeyboardInterrupt:
                print("\n[WARN] Interrupted by user (KeyboardInterrupt). Shutting down workers...")
                shutdown_flag.set()
                executor.shutdown(wait=False, cancel_futures=True)
                
            finally:
                pbar.close()

        # Remove the 'index' field from results before final save
        with results_lock:
            for r in results:
                r.pop('index', None)
        
        # Final save
        save_results_thread_safe()
        
        if errors:
            print(f"\n[WARN] {len(errors)} samples failed with errors")
            for no, err in errors:
                print(f"  - Sample {no}: {err.split(chr(10))[0]}")
        
        print(f"[INFO] Done. Saved {len(results)} samples to: {output_path}")
    
    else:
        # ==================== SINGLE-THREADED MODE ====================
        # log = print (stdout is already redirected to Logger)
        
        log = make_log_func(multithread=False)
        for no, sample in enumerate(
            tqdm(test_qa[start_no:], total=len(test_qa), initial=start_no),
            start=start_no 
        ):
            if args.num_samples is not None and no >= args.num_samples:
                break

            if no == 0:
                log(f"[INFO] no: {no}")
            else:    
                log(f"\n\n[INFO] no: {no}")

            try:
                question, img_path = create_question(sample, args.dataset)
                
                log(f"Question: {question}")
                
                final_decision = run_moa(question, PROPOSER_LAYERS, AGGREGATOR, SYNTHESIZE_PROMPT, return_intermediate=False, log=log)
                
                if args.dataset == 'medqa':
                    results.append({
                        'question': question,
                        'label': sample['answer_idx'],
                        'answer': sample['answer'],
                        'options': sample['options'],
                        'response': final_decision,
                    })
                else:
                    results.append({
                        'question': question,
                        'response': final_decision,
                    })

                # Save after each successful sample
                _atomic_json_dump(results, output_path)
                _atomic_json_dump({"next_index": len(results)}, progress_path)

            except KeyboardInterrupt:
                log("\n[WARN] Interrupted by user (KeyboardInterrupt). Saving progress and exiting...")
                _atomic_json_dump(results, output_path)
                _atomic_json_dump({"next_index": len(results)}, progress_path)
                break

            except Exception as e:
                log(f"\n[ERROR] Exception at sample index {no}: {type(e).__name__}: {e}")
                traceback.print_exc()
                log("[INFO] Saving progress up to last completed sample and exiting...")
                _atomic_json_dump(results, output_path)
                _atomic_json_dump({"next_index": len(results)}, progress_path)
                break  
            
        # Final save (in case loop ends normally)
        _atomic_json_dump(results, output_path)
        _atomic_json_dump({"next_index": len(results)}, progress_path)
        log(f"\n[INFO] Done. Saved {len(results)} samples to: {output_path}")


if __name__ == "__main__":
    main()