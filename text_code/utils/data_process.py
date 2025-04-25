from datasets import Dataset, DatasetDict
import random
import numpy as np
import json
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def cleaning(s):
    s = str(s)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace("[\w*"," ")
    s = s.replace("\\", " ")
    return s

def get_book_label_mapping(dataset):
    # Create a dictionary to hold book_id and their label
    book_labels = {}
    for entry in dataset:
        book_id = entry['book_id']
        label = entry['label']
        if book_id not in book_labels:
            book_labels[book_id] = label
    # sort
    book_labels = {k: v for k, v in sorted(book_labels.items(), key=lambda item: item[0])}
    
    # get the label to id mapping label:[]
    label_to_id = {}
    for book_id, label in book_labels.items():
        if label not in label_to_id:
            label_to_id[label] = []
        label_to_id[label].append(book_id)
    # sort
    for label, book_id in label_to_id.items():
        label_to_id[label] = sorted(book_id)

    return book_labels, label_to_id

def filter_snippets(data, k=20, target_label=0):
    # Create a dictionary to hold book_id and their snippet_ids
    book_snippets = {}
    
    # Iterate over each row in the dataset
    for entry in data:
        book_id = entry['book_id']
        snippet_id = entry['snippet_id']
        label = entry['label']
        
        # only use non-member books
        if label != target_label:
            continue

        # Check if book_id is already in the dictionary
        if book_id not in book_snippets:
            book_snippets[book_id] = []
        book_snippets[book_id].append(snippet_id)
    
    for book_id, snippet_ids in book_snippets.items():
        book_snippets[book_id] = sorted(snippet_ids)[:k]

    # Now filter the original dataset to only include valid book_id with snippet_id from 0 to k
    filtered_data = [entry for entry in data if (entry['label'] == target_label and entry['snippet_id'] in book_snippets[entry['book_id']])]
    print("Number of books:", len(book_snippets.keys()))
    return filtered_data, book_snippets

def sentence_chunking_dataset(dataset):
    # Download sentence tokenizer if not already downloaded
    # nltk.download('punkt')

    # Convert HF dataset to Pandas DataFrame
    df = pd.DataFrame(dataset)
    # data format: book_id, book, snippet_id, snippet

    # Step 1: Sort and concatenate snippets
    grouped_df = df.sort_values(by=['book_id', 'snippet_id']).groupby(['book_id', 'book'])['snippet'].apply(lambda x: ' '.join(x)).reset_index()

    book_to_sentences = {}

    # Step 2: Sentence-level segmentation
    chunked_data = []
    for _, row in grouped_df.iterrows():
        book_id = row['book_id']
        book = row['book']
        text = row['snippet']
        sentences = sent_tokenize(text)  # Split text into sentences

        book_to_sentences[book_id] = []
        
        for sentence_id, sentence in enumerate(sentences, start=1):
            # filter out sentences with length less than 3
            if len(sentence) < 3:
                continue
            chunked_data.append({
                'book_id': book_id, 
                'book': book, 
                'sentence_id': sentence_id, 
                'sentence': sentence
            })
            book_to_sentences[book_id].append(sentence_id)
            

    # Convert to Hugging Face Dataset
    chunked_dataset = Dataset.from_list(chunked_data)
    
    return chunked_dataset, book_to_sentences

def split_text(text, chunk_size=2000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Find the last space within the chunk size
        end = text.rfind(' ', start, end)
        if end == -1:
            end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

def split_into_chunks(text, tokenizer, max_length=512):
    chunk_size = int(1000*float(max_length/512))
    text_pieces = split_text(text, chunk_size)
    all_tokens = []
    for chunk in text_pieces:
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        all_tokens.extend(tokens)
    
    token_chunks = [all_tokens[i:i + max_length] for i in range(0, len(all_tokens), max_length)]
    chunks = [tokenizer.decode(chunk) for chunk in token_chunks]

    return chunks

def tar_pop_split(dataset, num_target_books):
    """
    Splits the training dataset into target and population datasets based on book_id.

    Args:
        dataset (DatasetDict): The Hugging Face dataset dictionary containing 'train' split.
        num_target_books (int): Number of books to select for the target dataset.
        num_pop_books (int): Number of books to select for the population dataset.

    Returns:
        DatasetDict: A dictionary containing 'target' and 'population' splits.
    """
    train_set = dataset["train"]
    df = train_set.to_pandas()
    unique_books = df["book_id"].unique().tolist()
    
    random.shuffle(unique_books)
    
    # Select target and population books
    target_books = set(unique_books[:num_target_books])
    pop_books = unique_books[num_target_books:]  # Remaining books as population

    # Split datasets
    target_df = df[df["book_id"].isin(target_books)]
    pop_df = df[df["book_id"].isin(pop_books)]
    
    target_dataset = Dataset.from_pandas(target_df)
    pop_dataset = Dataset.from_pandas(pop_df)
    
    return DatasetDict({
        "target": target_dataset,
        "population": pop_dataset,
        "test": dataset["test"]
    })