import os
import wandb
import time

# os.environ["WANDB_PROJECT"] = "GPT2 BookMIA Generation using PEFT"
# os.environ["WANDB_RUN_NAME"] = "debug"
# # os.environ["WANDB_DISABLED"] = "true"  # Disable Weights & Biases logging

from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from argparse import ArgumentParser
from huggingface_hub import hf_hub_download

import random
import json
import numpy as np

import utils

def construct_dataset(dataset, book_to_sentences, n=5, random_sampling=False, test_book_ids=None):
    """
    Constructs a training and test dataset based on selected books and sentence proportions.

    Args:
        dataset (DatasetDict): The splitted dataset dictionary with "target", "population" and "test" splits.
        book_to_sentences (dict): A mapping of book_id to a list of its sentences for "target" and "population" splits.
        n (int): Number of books per proportion group.
        random_sampling (bool): If True, selects snippets randomly; otherwise, sequentially.

    Returns:
        DatasetDict: A dictionary containing 'train' and 'test' splits.
    """
    # randomly split the target books into 6 groups (each proportion group contains 5 books)
    target_set = dataset["target"]
    df = target_set.to_pandas()
    unique_books = df["book_id"].unique().tolist()
    random.shuffle(unique_books)
    book_groups = [unique_books[i:i+n] for i in range(0, len(unique_books), n)]

    # Save the book groups to dict with keys 0%, 20%, 40%, 60%, 80%, 100%
    book_groups_dict = {}
    for i, group in enumerate(book_groups):
        book_groups_dict[i*0.2] = group
    
    # Reverse the dictionary to look up range by book_id
    book_id_to_p = {}
    for key, book_ids in book_groups_dict.items():
        for book_id in book_ids:
            book_id_to_p[book_id] = key
    
    selected_sentences = {}
    for book_id in unique_books:
        proportion = book_id_to_p.get(book_id, -1)
        if proportion != -1:
            num_snippets = int(len(book_to_sentences[book_id])*proportion)
            if random_sampling:
                selected_sentences[book_id] = random.sample(book_to_sentences[book_id], num_snippets)
            else: #sequential sampling
                selected_sentences[book_id] = book_to_sentences[book_id][:num_snippets]
    
    # sample n books from test set
    test_set = dataset["test"]
    test_df = test_set.to_pandas()
    test_unique_books = test_df["book_id"].unique().tolist()

    if test_book_ids is not None:
        selected_remaining_test_books = test_book_ids
        remaining_test_books = set(test_unique_books) - set(selected_remaining_test_books)
        selected_test_books = random.sample(remaining_test_books, n) # sample 5 books from remaining test set
    else:
        selected_remaining_test_books = random.sample(test_unique_books, n)
        remaining_test_books = set(test_unique_books) - set(selected_remaining_test_books)
        selected_test_books = random.sample(remaining_test_books, n) # sample 5 books from remaining test set

    # ================= Construct New Train and Test Datasets =================
    train_entries = target_set.filter(lambda x: x["book_id"] in selected_sentences.keys() and x["sentence_id"] in selected_sentences[x["book_id"]])
    remaining_train_entries = test_set.filter(lambda x: x["book_id"] in selected_test_books)

    test_entries = test_set.filter(lambda x: x["book_id"] in selected_remaining_test_books)

    return DatasetDict({
        "train": concatenate_datasets([train_entries, remaining_train_entries]),
        "test": test_entries
    }), book_groups_dict, book_id_to_p, selected_sentences, selected_test_books, selected_remaining_test_books


# Define tokenization and label addition function
def tokenize_and_format(examples):
    # examples["sentence"] = [utils.cleaning(text) for text in examples["sentence"]]
    tokenized_inputs = tokenizer(
        examples["sentence"], padding="max_length", truncation=True, max_length=args.max_length, return_special_tokens_mask=True
    )
    tokenized_inputs["labels"] = tokenized_inputs[
        "input_ids"
    ].copy()  # Duplicate input_ids to labels
    return tokenized_inputs


if __name__ == "__main__":
    # Initialize parser
    parser = ArgumentParser()
    parser.add_argument("--id", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--load_from_checkpoint", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--sampling_type", type=str, default="sequential", choices=["random", "sequential"])
    parser.add_argument("--tar_dir", type=str, default="./")
    args = parser.parse_args()
    args.dataset = "BookMIA-in-sentences-25"
    args.num_per_group = 5

    # Initialize Wandb project
    wandb.login()
    wandb.init(project="GPT2 BookMIA Copyright", name=f"tar{args.id}-{args.sampling_type}")

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    tar_path = f'{args.tar_dir}/exp/{args.dataset}/tar_models/{args.sampling_type}/target_models_{args.id}'
    os.makedirs(tar_path, exist_ok=True)
    
    if os.path.exists(f'./{args.dataset}'):
        dataset = load_from_disk(f"./{args.dataset}")

        with open(f'./{args.dataset}/book_selected_snippets.json', 'r') as f:
            book_selected_snippets = json.load(f)
        book_selected_snippets = {int(k): v for k, v in book_selected_snippets.items()}

        with open(f'./{args.dataset}/book_to_sentences.json', 'r') as f:
            book_to_sentences = json.load(f)
        book_to_sentences = {int(k): v for k, v in book_to_sentences.items()}
    else:
        # download from huggingface
        repo_id = f"YnezT/{args.dataset}"
        dataset = load_dataset(repo_id)
        book_snippets_path = hf_hub_download(repo_id=repo_id, filename="book_selected_snippets.json", repo_type="dataset")
        book_sentences_path = hf_hub_download(repo_id=repo_id, filename="book_to_sentences.json", repo_type="dataset")
        with open(book_snippets_path, "r") as f:
            book_selected_snippets = json.load(f)
        with open(book_sentences_path, "r") as f:
            book_to_sentences = json.load(f)

    if os.path.exists(f'./{args.dataset}_splitted'):
        dataset = load_from_disk(f"./{args.dataset}_splitted")
    else:
        dataset = utils.tar_pop_split(dataset, num_target_books=args.num_per_group*6)
        dataset.save_to_disk(f'./{args.dataset}_splitted')

    """ Prepare train test datasets"""
    test_set = dataset["test"]
    test_df = test_set.to_pandas()
    test_unique_books = test_df["book_id"].unique().tolist()
    test_book_ids = sorted(test_unique_books)[:args.num_per_group]
    if args.sampling_type == "random":
        dataset, book_groups_dict, book_id_to_p, selected_sentences, selected_test_books, selected_remaining_test_books = construct_dataset(dataset, book_to_sentences, n=args.num_per_group, random_sampling=True, test_book_ids=test_book_ids)
    else:
        dataset, book_groups_dict, book_id_to_p, selected_sentences, selected_test_books, selected_remaining_test_books = construct_dataset(dataset, book_to_sentences, n=args.num_per_group, random_sampling=False, test_book_ids=test_book_ids)
    
    # Save the data selection to disk
    np.savez(os.path.join(tar_path, 'data_selection.npz'), book_groups_dict=book_groups_dict, book_id_to_p=book_id_to_p, selected_sentences=selected_sentences, selected_test_books=selected_test_books, selected_remaining_test_books=selected_remaining_test_books)

    """ Prepare training """
    # Tokenize and format the dataset
    dataset = dataset.map(tokenize_and_format, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = f'./results/{args.sampling_type}/target_models_{args.id}/checkpoint'
    logging_dir = f"./logs/{args.sampling_type}/target_models_{args.id}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=5e-5,
        report_to="wandb",  # Only if you want to log your runs online
        run_name=f"GPT2 BookMIA Copyright tar model {args.id} - {args.sampling_type}",
    )

    trainer = Trainer(
        model = model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    s = time.time()
    trainer.train(resume_from_checkpoint=args.load_from_checkpoint)
    e = time.time()
    print(f"Training time: {e-s}")

    # Save the model
    model.save_pretrained(tar_path)
