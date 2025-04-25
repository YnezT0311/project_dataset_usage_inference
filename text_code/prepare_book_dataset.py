import os
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict
from argparse import ArgumentParser
import json
import numpy as np
from huggingface_hub import HfApi

import utils
if __name__ == "__main__":
    # Initialize parser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BookMIA-in-sentences-25")
    args = parser.parse_args()
    
    dataset = load_dataset("swj0419/BookMIA")
    # Save the dataset
    dataset = dataset['train']
    k = 25
    train_set, book_snippets = utils.filter_snippets(dataset, k=k, target_label=0) # downsampling the dataset to only retain k snippets per book (for non-member books)
    test_set, _ = utils.filter_snippets(dataset, k=k, target_label=1) # downsampling the dataset to only retain k snippets per book (for member books), used merely for evaluating the task performance
    
    train_set = Dataset.from_list(train_set)
    test_set = Dataset.from_list(test_set)
    train_set, book_to_sentences = utils.sentence_chunking_dataset(train_set)
    test_set, _ = utils.sentence_chunking_dataset(test_set)
    dataset = DatasetDict({'train': train_set, 'test': test_set})
    
    os.makedirs(f'./{args.dataset}', exist_ok=True)
    book_snippets_path = f'./{args.dataset}/book_selected_snippets.json'
    book_sentences_path = f'./{args.dataset}/book_to_sentences.json'
    with open(book_snippets_path, 'w') as f:
        json.dump(book_snippets, f)
    with open(book_sentences_path, 'w') as f:
        json.dump(book_to_sentences, f)
    dataset.save_to_disk(f'./{args.dataset}')

    dataset.push_to_hub(f"YnezT/{args.dataset}")

    # Push additional JSON files manually
    api = HfApi()
    api.upload_file(
        path_or_fileobj=book_snippets_path,
        path_in_repo="book_selected_snippets.json",
        repo_id=f"YnezT/{args.dataset}",
        repo_type="dataset"
    )

    api.upload_file(
        path_or_fileobj=book_sentences_path,
        path_in_repo="book_to_sentences.json",
        repo_id=f"YnezT/{args.dataset}",
        repo_type="dataset"
    )


    
