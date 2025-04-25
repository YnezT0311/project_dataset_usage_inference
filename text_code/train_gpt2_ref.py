import os
import wandb
import time

os.environ["WANDB_PROJECT"] = "GPT2 BookMIA Generation using PEFT"
os.environ["WANDB_RUN_NAME"] = "debug"
# os.environ["WANDB_DISABLED"] = "true"  # Disable Weights & Biases logging

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

def construct_ref_dataset(dataset, book_to_sentences, test_book_ids, n=5):
    """
    Constructs a training and test dataset based on selected books and sentence proportions.

    Args:
        dataset (DatasetDict): The splitted dataset dictionary with "target", "population" and "test" splits.
        book_to_sentences (dict): A mapping of book_id to a list of its sentences for "target" and "population" splits.
        n (int): Number of books per proportion group.

    Returns:
        DatasetDict: A dictionary containing 'train' and 'test' splits.
    """
    # randomly split the target books into 6 groups (each proportion group contains 5 books)
    target_set = dataset["target"]
    df = target_set.to_pandas()
    unique_books = df["book_id"].unique().tolist()
    
    proportion = 0.5
    # for each target book, select half of its sentences
    selected_sentences = {}
    for book_id in unique_books:
        num_snippets = int(len(book_to_sentences[book_id])*proportion)
        selected_sentences[book_id] = random.sample(book_to_sentences[book_id], num_snippets)
    
    # sample 2n books from population set
    pop_set = dataset["population"]
    pop_df = pop_set.to_pandas()
    pop_unique_books = pop_df["book_id"].unique().tolist()

    selected_pop_books = random.sample(pop_unique_books, 2*n)
    # for each selected population book, select half of its sentences
    selected_pop_sentences = {}
    for book_id in selected_pop_books:
        num_snippets = int(len(book_to_sentences[book_id])*proportion)
        selected_pop_sentences[book_id] = random.sample(book_to_sentences[book_id], num_snippets)

    # ================= Construct New Train and Test Datasets =================
    train_entries = target_set.filter(lambda x: x["book_id"] in selected_sentences.keys() and x["sentence_id"] in selected_sentences[x["book_id"]])
    pop_entries = pop_set.filter(lambda x: x["book_id"] in selected_pop_sentences.keys() and x["sentence_id"] in selected_pop_sentences[x["book_id"]])

    test_entries = test_set.filter(lambda x: x["book_id"] in test_book_ids)

    return DatasetDict({
        "train": concatenate_datasets([train_entries, pop_entries]),
        "test": test_entries
    }), selected_sentences, selected_pop_sentences


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
    parser.add_argument("--ref_dir", type=str, default="./")
    args = parser.parse_args()
    args.dataset = "BookMIA-in-sentences-25"
    args.num_per_group = 5

    # Initialize Wandb project
    wandb.login()
    wandb.init(project="GPT2 BookMIA Copyright", name=f"ref{args.id}")

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    ref_path = f'{args.ref_dir}/exp/{args.dataset}/ref_models/ref_models_{args.id}'
    os.makedirs(ref_path, exist_ok=True)
    
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
    dataset, selected_sentences, selected_pop_sentences = construct_ref_dataset(dataset, book_to_sentences, test_book_ids, n=args.num_per_group)
    
    # Save the data selection to disk
    np.savez(os.path.join(ref_path, 'data_selection.npz'), selected_sentences=selected_sentences, selected_pop_sentences=selected_pop_sentences, selected_remaining_test_books=test_book_ids)
             
    """ Prepare training """
    # Tokenize and format the dataset
    dataset = dataset.map(tokenize_and_format, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = f'./results/ref_models_{args.id}/checkpoint'
    logging_dir = f"./logs/ref_models_{args.id}"
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
        run_name="GPT2 BookMIA Copyright ref model {}".format(args.id),
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
    model.save_pretrained(ref_path)
