import os
import argparse
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import random
from prettytable import PrettyTable

try:
    from eval.call_llm import ask
except ImportError:
    from call_llm import ask


# Default valid choices for MedQA dataset
VALID_CHOICES = {"A", "B", "C", "D", "E"}
VALID_CHOICES_PLUS_X = VALID_CHOICES | {"X"}


def extract_response(sample: dict) -> str:
    """Extract response content for a given sample"""
    return sample["response"][TEMP] if isinstance(sample["response"], dict) else sample["response"]


def extract_answer(response: str) -> str:
    """Extract the answer from a LLM response by calling Ollama API"""
    system_prompt = (
        "You are a grading helper. Your only job is to extract the single final answer choice letter from a model response.\n"
        "Rules:\n"
        "- Output MUST be exactly one character: A, B, C, D, or E.\n"
        "- Output MUST contain nothing else (no words, no punctuation, no spaces, no newlines).\n"
        "- If the answer is not clear, output X (a single character).\n"
    )
    user_prompt = (
        "Extract the final answer letter (A/B/C/D/E) from the response below.\n\n"
        f"RESPONSE:\n<<<\n{response}\n>>>\n"
    )

    extracted_answer = ask(
        user_prompt=user_prompt,
        sys_prompt=system_prompt,
        model_name="gpt-oss:20b",
        thinking=False,
        max_tokens=3,
        temperature=0.0,
        infinite_retry=True,
    )

    # Normalize to a single char
    out = (extracted_answer or "").strip().upper()
    if not out:
        return "X"
    # Keep only first valid char among A-E/X
    for ch in out:
        if ch in {"A", "B", "C", "D", "E", "X"}:
            return ch
    return "X"


def _atomic_write_json(path: str, obj: Any) -> None:
    """Write JSON atomically to avoid corruption on interruption."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_existing_results(result_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(result_path):
        return []
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        # If file is corrupted/partial, back it up and start fresh
        bak = result_path + ".corrupted.bak"
        try:
            os.replace(result_path, bak)
            print(f"[WARN] Existing result file was corrupted. Backed up to: {bak}")
        except Exception:
            pass
        return []


def extract_and_save_predictions(
    samples: List[dict],
    result_path: str,
    flush_every: int = 1,
) -> List[Dict[str, Any]]:
    """
    Extract predictions and continuously save to result_path.
    - Resume: if result_path exists, skip already processed indices.
    - tqdm progress bar.
    - Writes atomically to avoid file corruption.
    """
    existing = _load_existing_results(result_path)

    # Resume strategy: store "idx" with each record; use it to skip
    done_idx = set()
    for rec in existing:
        if isinstance(rec, dict) and "idx" in rec:
            done_idx.add(rec["idx"])

    results: List[Dict[str, Any]] = existing[:]  # keep what we already have

    pbar = tqdm(total=len(samples), desc="Extracting labels", unit="sample")
    # If resuming, reflect progress
    if done_idx:
        pbar.update(len(done_idx))

    writes_since_flush = 0

    for i, sample in enumerate(samples):
        if i in done_idx:
            continue

        label = sample.get("label")
        response = extract_response(sample)
        pred = extract_answer(response)

        rec = {
            "idx": i,
            "label": label,
            "extracted_answer": pred,
        }
        results.append(rec)
        done_idx.add(i)

        writes_since_flush += 1
        if writes_since_flush >= flush_every:
            _atomic_write_json(result_path, results)
            writes_since_flush = 0

        pbar.update(1)

    # Final flush
    _atomic_write_json(result_path, results)
    pbar.close()

    return results

def _normalize_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    # If label might be like "(B)" or "B)" etc, normalize
    for ch in s:
        if ch in VALID_CHOICES:
            return ch
    return None

def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate extracted predictions:
    - overall accuracy (correct / total predictions passed in)
    - basic confusion-like stats for each class 
    """
    total = len(predictions)
    correct = 0
    
    # Store per-class stats: { 'A': {'total': 0, 'correct': 0}, ... }
    class_stats = {label: {'total': 0, 'correct': 0} for label in VALID_CHOICES}

    for rec in predictions:
        gold = _normalize_label(rec.get("label"))
        
        raw_pred = str(rec.get("extracted_answer", "")).strip().upper()
        # Treat 'X', None, or invalid as wrong
        pred = _normalize_label(raw_pred) if raw_pred != "X" else None
        
        # Check correctness
        if gold is not None and pred == gold:
            correct += 1

        # Update class stats if valid gold
        if gold in VALID_CHOICES:
            class_stats[gold]['total'] += 1
            if pred == gold:
                class_stats[gold]['correct'] += 1

    overall_accuracy = correct / total if total > 0 else 0.0

    summary = {
        "n_evaluated": total,
        "n_correct": correct,
        "accuracy": overall_accuracy,
        "per_class": {}
    }

    if verbose:
        print("\n" + "="*40)
        print("EVALUATION REPORT")
        print("="*40)
        print(f"Total Samples:    {total}")
        print(f"Overall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")
        print("-" * 40)
        
        pt = PrettyTable()
        pt.field_names = ["Class", "Total", "Correct", "Accuracy"]
        pt.align = "r"
        pt.align["Class"] = "c"
        
        for label in sorted(VALID_CHOICES):
            c_total = class_stats[label]['total']
            c_corr = class_stats[label]['correct']
            c_acc = (c_corr / c_total) if c_total > 0 else 0.0
            
            pt.add_row([ label, c_total, c_corr, f"{c_acc:.1%}" ])
            
            summary["per_class"][label] = {
                "total": c_total,
                "correct": c_corr,
                "accuracy": c_acc
            }
            
        print(pt)
        print("="*40 + "\n")

    return summary

if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--flush_every", type=int, default=1, help="Write to JSON every N new predictions")
    parser.add_argument("--mode", type=str, choices=["extract", "eval", "both"], default="both", help="Run extraction, evaluation, or both")
    parser.add_argument("--num_samples", type=int, default=None, help="Select random N samples in the prediction output JSON file for evaluation. If not set, evaluate all samples.")
    args = parser.parse_args()

    # Response file + result paths for extraction
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    files = args.file_path.split("/")
    file_name = files[-1].replace('.json', '')  # strip .json if present
    result_dir = os.path.join(project_root, "results", "preds")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{file_name}.predictions.json")
    
    # Evaluation summary path
    eval_dir = os.path.join(project_root, "results", "metrics")
    os.makedirs(eval_dir, exist_ok=True)
    eval_path = os.path.join(eval_dir, f"{file_name}.eval.json")
    
    # Load samples
    samples: List[dict] = []
    if os.path.exists(args.file_path):
        with open(args.file_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
            
    # Get temp value for extraction
    global TEMP
    if samples and "response" in samples[0]:
        first_response = samples[0]["response"]
        if isinstance(first_response, dict):
            TEMP = list(first_response.keys())[0]
        else:
            TEMP = ''

    # ----- PART 1: Extract and save predictions ----- #
    if args.mode in {"extract", "both"}:
        if not samples:
            raise FileNotFoundError(f"Cannot extract: dataset response file not found: {args.file_path}")
        predictions = extract_and_save_predictions(samples, result_path=result_path, flush_every=args.flush_every)
        print(f"[INFO] Saved predictions to: {result_path} (total records: {len(predictions)})")

    # ----- PART 2: Evaluate predictions ----- #
    if args.mode in {"eval", "both"}:
        preds = _load_existing_results(result_path)
        if not preds:
            raise FileNotFoundError(f"Cannot evaluate: predictions file not found/empty: {result_path}")
        if args.num_samples is not None:
            preds = random.sample(preds, args.num_samples)
        summary = evaluate_predictions(preds, verbose=True)

        _atomic_write_json(eval_path, summary)
        print(f"[INFO] Saved eval summary to: {eval_path}")