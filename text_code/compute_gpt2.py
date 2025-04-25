import argparse
import os
import json
import random
import time
from collections import defaultdict
from torch.nn.functional import cross_entropy

from transformers import AutoModelForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, load_from_disk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel

import utils

os.environ["WANDB_DISABLED"] = "true"  # Disable Weights & Biases logging
if not torch.cuda.is_available():
    raise ValueError("CUDA is not available. This script requires a GPU.")
device = torch.device("cuda")

def set_seed(seed=42):
    """Set the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Covers both single & multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_ppl_batch(args, examples, model, tokenizer, device):
    """Compute perplexity (PPL) for a batch of texts using a language model."""
    texts = examples["sentence"]
    
    encodings = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=args.max_length, truncation=True)
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
        # #subtracting the maximum value to prevent potential overflow
        logits = logits - torch.max(logits, dim=-1, keepdim=True).values

        shift_logits = logits[:, :-1, :].contiguous()  # Shift for next-token prediction
        shift_labels = input_ids[:, 1:].contiguous()  # Align labels with shifted logits
        # Set padding tokens in labels to -100 (so they are ignored in loss)
        shift_labels[attention_mask[:, 1:] == 0] = -100

        # Compute per-token loss using CrossEntropy
        loss_per_token = cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),  # Reshape logits to (N, vocab_size)
            shift_labels.view(-1),  # Flatten labels to match
            reduction='none'  # No reduction, keep individual losses
        )  # Shape: (64 * 255,)

        # Reshape back to (batch_size, seq_length-1)
        loss_per_token = loss_per_token.view(shift_logits.shape[0], shift_logits.shape[1])  # Shape: (64, 255)

        # Mask padding tokens so they don't contribute to the loss
        loss_per_token_masked = loss_per_token * attention_mask[:, 1:].float()  # Shape: (64, 255)

        # Compute average loss per sequence
        loss_per_record = loss_per_token_masked.sum(dim=1) / attention_mask[:, 1:].sum(dim=1)  # Shape: (64,)

    ppl = torch.exp(loss_per_record).cpu().numpy().tolist()  # Convert to list for dataset compatibility

    return ppl

def get_score_and_keeps(args, dataset, tokenizer, model_path, device, dataset_type='tar', model_type='tar'): # dataset_type: type of the queried dataset, tar or pop; model_type: type of the model, tar or ref
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.pad_token
    model.eval()
    model.to(device)

    data_selection = np.load(os.path.join(model_path, 'data_selection.npz'), allow_pickle=True)
    if dataset_type == 'tar':
        selected_sentences = data_selection['selected_sentences'].item()
    elif dataset_type == 'pop':
        if model_type == "ref":
            selected_sentences = data_selection['selected_pop_sentences'].item()
        else:
            selected_sentences = None

    # get ppl
    results = defaultdict(lambda: {"sentence_ids": [], "ppl": [], "memberships": []})
    def process_batch(batch):
        book_ids = batch["book_id"]
        sentence_ids = batch["sentence_id"]
        ppl_batch = compute_ppl_batch(args, batch, model, tokenizer, device)

        for i in range(len(book_ids)):
            results[book_ids[i]]["sentence_ids"].append(sentence_ids[i])
            results[book_ids[i]]["ppl"].append(ppl_batch[i])
            if selected_sentences is not None:
                if selected_sentences.get(book_ids[i]) is not None and sentence_ids[i] in selected_sentences[book_ids[i]]:
                    results[book_ids[i]]["memberships"].append(1)
                else:
                    results[book_ids[i]]["memberships"].append(0)
            else:
                results[book_ids[i]]["memberships"].append(0)

    dataset.map(process_batch, batched=True, batch_size=32)
    # Save the results
    with open(os.path.join(model_path, f'{dataset_type}_dataset_ppl.json'), 'w') as f:
        json.dump(results, f)
    
    # clear memory
    del model
    torch.cuda.empty_cache()
    
def main(args):
    """
    Perform inference of the saved model in order to generate the
    output logits, using a particular set of augmentations.
    """
    # set the seed
    set_seed(42)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    save_dir = os.path.join(args.save_dir, 'exp', args.dataset)
    tar_dir = os.path.join(save_dir, f'tar_models/{args.sampling_type}')
    ref_dir = os.path.join(save_dir, 'ref_models')

    """ Get all ppl for target and population datasets """
    ref_path_list = os.listdir(ref_dir)
    ref_path_list = [os.path.join(ref_dir, path) for path in ref_path_list]
    tar_path_list = os.listdir(tar_dir)
    tar_path_list = [os.path.join(tar_dir, path) for path in tar_path_list]

    if os.path.exists(f'./{args.dataset}_splitted'):
        dataset = load_from_disk(f"./{args.dataset}_splitted")
    else:
        dataset = utils.tar_pop_split(dataset, num_target_books=args.num_per_group*6)
        dataset.save_to_disk(f'./{args.dataset}_splitted')
    target_set = dataset['target']
    pop_set = dataset['population']

    # target dataset on target model
    s = time.time()
    for tar_path in tar_path_list:
        get_score_and_keeps(args, target_set, tokenizer, tar_path, device, dataset_type='tar', model_type='tar')
    end = time.time()
    print(f"Time taken to get target ppl on target models: {end-s}")

    # target dataset on ref models
    s = time.time()
    for ref_path in ref_path_list:
        get_score_and_keeps(args, target_set, tokenizer, ref_path, device, dataset_type='tar', model_type='ref')
    end = time.time()
    print(f"Time taken to get target ppl on ref models: {end-s}")

    # population dataset on target models
    s = time.time()
    for tar_path in tar_path_list:
        get_score_and_keeps(args, pop_set, tokenizer, tar_path, device, dataset_type='pop', model_type='tar')
    end = time.time()
    print(f"Time taken to get population ppl on target models: {end-s}")

    # population dataset on ref models
    s = time.time()
    for ref_path in ref_path_list:
        get_score_and_keeps(args, pop_set, tokenizer, ref_path, device, dataset_type='pop', model_type='ref')
    end = time.time()
    print(f"Time taken to get population ppl on ref models: {end-s}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./exp/', help='Path to the saved models.')
    parser.add_argument("--dataset", type=str, default="BookMIA-in-sentences-25")
    parser.add_argument("--sampling_type", type=str, default="sequential", choices=["random", "sequential"], help='Sampling type for target models.')
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()  
    main(args)