import sys
import json
import random
from src.models.moa import run_moa

def load_data(dataset):
    test_qa = []

    test_path = f'data/{dataset}/try.jsonl'
    with open(test_path, 'r') as file:
        for line in file:
            test_qa.append(json.loads(line))

    return test_qa

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question
    return sample['question']

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

main_log_path = f"logs/test.log"
logger = Logger(main_log_path)
sys.stdout = logger


def main():
    user_prompt = create_question(load_data('medqa')[0], 'medqa')
    
    proposer_layers = [
		[
			{"provider": "mistral", "model": "ministral-14b-2512", "temperature": 0.7},
			{"provider": "mistral", "model": "ministral-8b-2512", "temperature": 0.7},
			{"provider": "mistral", "model": "ministral-3b-2512", "temperature": 0.7},
			{"provider": "mistral", "model": "mistral-small-2512", "temperature": 0.7},
		]
	]
    
    aggregator = {
		"provider": "mistral",
		"model": "mistral-large-2512",
		"temperature": 0.0,
		"max_tokens": 512,
	}

    final_answer = run_moa(
        user_prompt=user_prompt,
        proposer_layers=proposer_layers,
        aggregator=aggregator,
    )
    
    print("\n[INFO] Final answer:\n", final_answer)
    logger.close()

if __name__ == "__main__":
	main()
