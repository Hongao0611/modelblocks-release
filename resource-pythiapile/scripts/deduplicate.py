"""
@author: Hongao ZHU

@description:
    extract a range of batches from the Pythia training data, iterate over the texts in the batches,
    save the detected texts to 'BATCH_RANGE.csv', and write logs into 'BATCH_RANGE.log'.

    the deduplication uses the most surprised tokens from the naturalstories corpus.

@example usage:
    TO FULLY DEDUPLICATE, RUN:
    python3 scripts/deduplicate.py -B 0_142998 -N 500 -C /fs/project/schuler.77/corpora/pythia_pile -D modelblocks-release/workspace/genmodel/naturalstories.unigram.itemmeasures

"""

import numpy as np, pandas as pd
import sys, json, os, re, logging
from transformers import AutoTokenizer
import argparse

# redirect output into logs
def setup_logging(output_file):
    sys.stdout = open(f'{output_file}.log', 'a')
    sys.stderr = open(f'{output_file}.log', 'a')

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
)

class BatchIterator:
    def __init__(self, start_batch=0, end_batch=142998, data_dir='/fs/project/schuler.77/corpora/pythia_pile'):
        self.data_dir = data_dir
        self.batch_size = 1024
        self.current_index = start_batch
        self.end_index = end_batch
        self.current_file = None
        self.current_data = None
        self.file_index = 0

    def get_batch(self, index):
        file_name = f"{self.data_dir}/batch_{index//1000*1000}_to_{index//1000*1000+1000}.npy"
        batch_start = (index % 1000) * self.batch_size
        batch_end = batch_start + self.batch_size
        # load data using np.memmap
        mmapped_data = np.memmap(file_name, dtype='uint16', mode='r', shape=(2098176000,))
        batch_data = mmapped_data[batch_start:batch_end]
        return batch_data

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.end_index:
            raise StopIteration
        self.current_data = self.get_batch(self.current_index)
        self.file_index += 1
        self.current_index += 1
        return self.current_data

def calculate_average_surprisal(file_path):
    word_surprisal_map = {}
    with open(file_path, 'r') as file:
        next(file)  # skip headers
        for line in file:
            word, surprisal = line.strip().split()
            word = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', word)
            surprisal = float(surprisal)
            if word in word_surprisal_map:
                word_surprisal_map[word]['count'] += 1
                word_surprisal_map[word]['total_surprisal'] += surprisal
            else:
                word_surprisal_map[word] = {'count': 1, 'total_surprisal': surprisal}

    return sorted({word: data['total_surprisal'] / data['count'] for word, data in word_surprisal_map.items()}.items(), key=lambda item: item[1])

def deduplicate(batch_range, unigram_itemmeasures_path, num=10, corpora_dir='/fs/project/schuler.77/corpora/pythia_pile'):
    # Get batch range and reference dictionary
    start_batch, end_batch = map(int, batch_range.split('_'))
    batches = BatchIterator(start_batch, end_batch, corpora_dir)
    ref_lst = calculate_average_surprisal(unigram_itemmeasures_path)
    if num<= len(ref_lst):
        ref_lst = ref_lst[:-num:-1]
    print(f"Searching for tokens:{ref_lst}:")

    # Process batches
    records = []
    for i, batch in enumerate(batches):
        print(f"processing batch {i+start_batch}...")
        for j, text in enumerate(batch):  # 1024 texts in one batch
            match_positions = []
            decoded_text = tokenizer.decode(text)
            words = re.findall(r'\b\w+\b', decoded_text)

            # mark the tokens in the ref_lst
            for i, word in enumerate(words):
                if word in ref_lst:
                    match_positions.append(i)
            
            # check if the marked tokens are connected (in a 5-gram)
            for i in range(len(match_positions) - 2):
                if match_positions[i+2] - match_positions[i] < 5:
                    # Record the duplicates
                    tokens = words[match_positions[i]], words[match_positions[i+1]], words[match_positions[i+2]]
                    print(f"Found duplicates in Batch: {i+start_batch} Text: {j}.")
                    records.append((i+start_batch,j,tokens,decoded_text))

        # Save the surprisal values everytime when one batch is completed
        df = pd.DataFrame(records)
        df.to_csv(f"{batch_range}.csv",index=False)
        del df
    print("Deduplication Complete!!!")


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Get surprisal estimation for the Pythia training batches.')
    parser.add_argument('--batch', '-B', type=str, default='0_1', help='Batch range. Default:0_1.')
    parser.add_argument('--num', '-N', type=int, default=10, help='Number of most surprised tokens.')
    parser.add_argument('--dict', '-D', type=str, default='naturalstories.unigram.itemmeasures', help='naturalstories.unigram.itemmeasures')
    parser.add_argument('--corpora', '-C', type=str, default='/fs/project/schuler.77/corpora/pythia_pile', help='/fs/project/schuler.77/corpora/pythia_pile')

    # Look for surprisal_dict.json
    args = parser.parse_args()

    setup_logging(args.batch)
    deduplicate(args.batch, args.dict, args.num, args.corpora)


if __name__ == '__main__':
    main()
