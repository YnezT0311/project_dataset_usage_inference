import argparse
import os
import json
import time
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict, concatenate_datasets


import utils
from modules.duci import duci

os.environ["WANDB_DISABLED"] = "true"  # Disable Weights & Biases logging
if not torch.cuda.is_available():
    raise ValueError("CUDA is not available. This script requires a GPU.")
device = torch.device("cuda")

def main(args):
    """
    Evaluate the DUCI performance
    """
    save_dir = os.path.join(args.save_dir, 'exp', args.dataset)
    tar_dir = os.path.join(save_dir, f'tar_models/{args.sampling_type}')
    ref_dir = os.path.join(save_dir, 'ref_models')

    ref_path_list = os.listdir(ref_dir)
    ref_path_list = [os.path.join(ref_dir, path) for path in ref_path_list]
    tar_path_list = os.listdir(tar_dir)
    tar_path_list = [os.path.join(tar_dir, path) for path in tar_path_list]

    if os.path.exists(f'./{args.dataset}_splitted'):
        dataset = load_from_disk(f"./{args.dataset}_splitted")
    target_set = dataset['target']
    pop_set = dataset['population']
    target_ids = np.unique(target_set['book_id'])
    pop_ids = np.unique(pop_set['book_id'])

    p_x_ref_collect = defaultdict(lambda: {"sentence_ids": [], "ppl": [], "memberships": []})
    p_z_ref_collect = defaultdict(lambda: {"sentence_ids": [], "ppl": [], "memberships": []})

    def load_and_sort_data(ref_path, filename, collect_dict):
        with open(os.path.join(ref_path, filename), "r") as f:
            for book_id, data in json.load(f).items():
                sentence_ids = np.array(data["sentence_ids"])
                ppl = np.array(data["ppl"])
                memberships = np.array(data["memberships"])
                
                sorted_indices = np.argsort(sentence_ids)
                collect_dict[book_id]["sentence_ids"].append(sentence_ids[sorted_indices])
                collect_dict[book_id]["ppl"].append(ppl[sorted_indices])
                collect_dict[book_id]["memberships"].append(memberships[sorted_indices])
    
    def sort_data(data_dict):
        for book_id, data in data_dict.items():
            sentence_ids = np.array(data["sentence_ids"])
            ppl = np.array(data["ppl"])
            memberships = np.array(data["memberships"])
            sorted_indices = np.argsort(sentence_ids)
            data["sentence_ids"] = sentence_ids[sorted_indices]
            data["ppl"] = ppl[sorted_indices]
            data["memberships"] = memberships[sorted_indices]
            data_dict[book_id] = data
        return data_dict

    for ref_path in ref_path_list:
        load_and_sort_data(ref_path, 'tar_dataset_ppl.json', p_x_ref_collect)
        load_and_sort_data(ref_path, 'pop_dataset_ppl.json', p_z_ref_collect)
    
    for key, values in p_x_ref_collect.items():
        p_x_ref_collect[key]["sentence_ids"] = np.array(values["sentence_ids"]).T
        p_x_ref_collect[key]["ppl"] = np.array(values["ppl"]).T
        p_x_ref_collect[key]["memberships"] = np.array(values["memberships"]).T
    
    for key, values in p_z_ref_collect.items():
        p_z_ref_collect[key]["sentence_ids"] = np.array(values["sentence_ids"]).T # (number_sentence, num_ref_models)
        p_z_ref_collect[key]["ppl"] = np.array(values["ppl"]).T
        p_z_ref_collect[key]["memberships"] = np.array(values["memberships"]).T

    p_z_ref = np.concatenate([p_z_ref_collect[str(z_book_id)]["ppl"] for z_book_id in pop_ids])
    memberships_z = np.concatenate([p_z_ref_collect[str(z_book_id)]["memberships"] for z_book_id in pop_ids])
    
    predictions = defaultdict(lambda: {"raw_agg_mia_p": [], "raw_agg_score_p": [], "debiased_p": [], "true_p": []})
    for tar_path in tar_path_list:
        with open(os.path.join(tar_path, f'tar_dataset_ppl.json'), "r") as f:
            p_x_tar_collect = json.load(f)
            p_x_tar_collect = sort_data(p_x_tar_collect)
        with open(os.path.join(tar_path, f'pop_dataset_ppl.json'), "r") as f:
            p_z_tar_collect = json.load(f)
            p_z_tar_collect = sort_data(p_z_tar_collect)
        
        data_selection = np.load(os.path.join(tar_path, 'data_selection.npz'), allow_pickle=True)
        book_id_to_p = data_selection['book_id_to_p'].item()

        for book_id in target_ids:
            p_x_tar = p_x_tar_collect[str(book_id)]["ppl"][:,np.newaxis]
            p_z_tar = np.concatenate([p_z_tar_collect[str(z_book_id)]["ppl"] for z_book_id in pop_ids])[:,np.newaxis]
            memberships_x = p_x_ref_collect[str(book_id)]["memberships"]
            p_x_ref = p_x_ref_collect[str(book_id)]["ppl"]

            # convert ppl to confidence such that higher value corresponds to member (for roc curve)
            p_x_tar = 1 / p_x_tar
            p_x_ref = 1 / p_x_ref
            p_z_tar = 1 / p_z_tar
            p_z_ref = 1 / p_z_ref

            raw_agg_mia_p, raw_agg_score_p, debiased_p = duci(p_x_tar, p_x_ref, p_z_tar, p_z_ref, memberships_x, memberships_z)
            predictions[book_id]["raw_agg_mia_p"].append(raw_agg_mia_p)
            predictions[book_id]["raw_agg_score_p"].append(raw_agg_score_p)
            predictions[book_id]["debiased_p"].append(debiased_p)
            predictions[book_id]["true_p"].append(book_id_to_p[book_id])
    

    #================= Evaluation =================
    # Extract predictions by true_p: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    n_proportions = 6
    raw_agg_mia_p_all, raw_agg_score_p_all, debiased_p_all, true_p_all = [], [], [], []
    for key, values in predictions.items():
        raw_agg_mia_p_all.extend(values["raw_agg_mia_p"])
        raw_agg_score_p_all.extend(values["raw_agg_score_p"])
        debiased_p_all.extend(values["debiased_p"])
        true_p_all.extend(values["true_p"])
    raw_agg_mia_p_all = np.array(raw_agg_mia_p_all)
    raw_agg_score_p_all = np.array(raw_agg_score_p_all)
    debiased_p_all = np.array(debiased_p_all)
    true_p_all = np.array(true_p_all)

    # reordered by true_p
    idx = np.argsort(true_p_all)
    true_p_all = true_p_all[idx]
    raw_agg_mia_p_all = raw_agg_mia_p_all[idx]
    raw_agg_score_p_all = raw_agg_score_p_all[idx]
    debiased_p_all = debiased_p_all[idx]
    # Reshape the arrays to (n_proportions, num_tests)
    true_p_matrix = true_p_all.reshape(n_proportions, -1)
    raw_agg_mia_p_matrix = raw_agg_mia_p_all.reshape(n_proportions, -1)
    raw_agg_score_p_matrix = raw_agg_score_p_all.reshape(n_proportions, -1)
    debiased_p_matrix = debiased_p_all.reshape(n_proportions, -1)

    # compute mae for each true_p
    raw_agg_mia_p_mae = np.mean(np.abs(raw_agg_mia_p_matrix - true_p_matrix), axis=1) # shape: (n_proportions,)
    raw_agg_score_p_mae = np.mean(np.abs(raw_agg_score_p_matrix - true_p_matrix), axis=1)
    debiased_p_mae = np.mean(np.abs(debiased_p_matrix - true_p_matrix), axis=1)

    print(f"Max MAE over proportions: DUCI = {np.max(debiased_p_mae)} | MIA Score = {np.max(raw_agg_score_p_mae)} | MIA Guess = {np.max(raw_agg_mia_p_mae)}")
    # print mae for each true_p
    for i, true_p in enumerate(np.arange(0.2, 1.1, 0.2)):
        print(f"MAE for true_p = {true_p:.1f}: DUCI = {debiased_p_mae[i]:.4f} | MIA Score = {raw_agg_score_p_mae[i]:.4f} | MIA Guess = {raw_agg_mia_p_mae[i]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./', help='Path to the saved models.')
    parser.add_argument("--dataset", type=str, default="BookMIA-in-sentences-25")
    parser.add_argument("--sampling_type", type=str, default="sequential", choices=["random", "sequential"])
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()  
    main(args)