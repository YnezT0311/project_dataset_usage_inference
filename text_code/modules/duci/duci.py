import numpy as np
from sklearn.metrics import roc_curve, auc

from modules.mia import rmia

def sweep(y_score, y_true):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC using sklearn.
    Only valid for single data case
    """
    fpr, tpr, threshs = roc_curve(y_true=y_true.astype(int), y_score=y_score, pos_label=1)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return (fpr, tpr, auc(fpr, tpr), acc, threshs)

def detect_invalid_sentences(p_z_tar, p_z_ref):
    """
    Detect invalid entries in the population book.
    """
    invalid_tar = np.isnan(p_z_tar).any(axis=1)
    invalid_ref = np.isnan(p_z_ref).any(axis=1)
    assert (invalid_ref == invalid_tar).sum() == invalid_tar.shape[0], "Invalid z entries are not consistent."
    invalid_z = invalid_tar | invalid_ref
    return invalid_z

def select_robust_threshold(tpr, fpr, thresholds, scores, memberships, top_percent=0.05):
    """
    Selects the threshold maximizing the TPR - FPR values (i.e., highest accuracy) by selecting the mean of the longest contiguous range in the `top_percent`
    of TPR - FPR values.

    Args:
        tpr (np.array): True Positive Rate values.
        fpr (np.array): False Positive Rate values.
        thresholds (np.array): Corresponding thresholds.
        scores (np.array): Scores to consider.
        memberhips (np.array): Memberships to consider.s
        top_percent (float): Percentage of top (TPR - FPR) values to consider.

    Returns:
        best_threshold (float): Selected threshold.
        best_tpr (float): TPR at the selected threshold.
        best_fpr (float): FPR at the selected threshold.
    """

    tpr_fpr_diff = tpr - fpr
    sorted_indices = np.argsort(tpr_fpr_diff)[::-1]

    # Select the top `top_percent` indices
    num_top = max(1, int(len(sorted_indices) * top_percent))  # Ensure at least one index

    # Find the longest contiguous range
    top_indices = np.sort(sorted_indices[:num_top])
    longest_range = []
    current_range = [top_indices[0]]

    for i in range(1, len(top_indices)):
        if top_indices[i] <= top_indices[i - 1] + 3:
            current_range.append(top_indices[i])
        else:
            if len(current_range) > len(longest_range):
                longest_range = current_range
            current_range = [top_indices[i]]

    # Final check in case the longest range is at the end
    if len(current_range) > len(longest_range):
        longest_range = current_range
    

    # Get the thresholds within this longest range
    continuous_indices = np.array(longest_range)
    selected_thresholds = thresholds[continuous_indices]
    best_threshold = np.mean(selected_thresholds)

    # Compute the corresponding TPR and FPR under the threshold
    best_tpr = np.sum((scores >= best_threshold) & (memberships == 1)) / np.sum(memberships == 1)
    best_fpr = np.sum((scores >= best_threshold) & (memberships == 0)) / np.sum(memberships == 0)

    return best_threshold, best_tpr, best_fpr

def duci(p_x_tar, p_x_ref, p_z_tar, p_z_ref, memberships_x, memberships_z):
    """
    Dataset usage cardinality inference (DUCI) for each target book.

    Args:
        p_x_tar (np.ndarray): The predicted probabilities over the entries in the target book on target models. Shape: (num_target_entries, num_target_models).
        p_x_ref (np.ndarray): The predicted probabilities over the entries in the target book on reference models. Shape: (num_target_entries, num_reference_models).
        p_z_tar (np.ndarray): The predicted probabilities over the entries in the population book on target models. Shape: (num_population_entries, num_target_models).
        p_z_ref (np.ndarray): The predicted probabilities over the entries in the population book on reference models. Shape: (num_population_entries, num_reference_models).
        memberships_x (np.ndarray): The memberships of the target book entries on reference models. Shape: (num_target_entries, num_reference_models).
        memberships_z (np.ndarray): The memberships of the population book entries on reference models. Shape: (num_population_entries, num_reference_models).
    """
    # filter out nan values z in p_z_tar and p_z_ref, as some sentences are "."
    invalid_z = detect_invalid_sentences(p_z_tar, p_z_ref)
    p_z_tar = p_z_tar[~invalid_z]
    p_z_ref = p_z_ref[~invalid_z]
    memberships_z = memberships_z[~invalid_z]

    # filter out nan values x in p_x_tar and p_x_ref, as some sentences are "."
    invalid_x = detect_invalid_sentences(p_x_tar, p_x_ref)
    p_x_tar = p_x_tar[~invalid_x]
    p_x_ref = p_x_ref[~invalid_x]
    memberships_x = memberships_x[~invalid_x]

    # get rmia score for each book in shape (num_target_entries, num_target_models)
    # Following common practice in RMIA for language models, we omit population data due to high inference costs (rmia_ratios = p_x_tar/p_x).
    # However, we allow the aggregated MIA score baseline access to population data (rmia_scores), as it requires probability computation.
    rmia_scores, rmia_ratios = rmia(p_x_tar, p_x_ref, p_z_tar, p_z_ref, memberships_x, memberships_z, offline_a = 0.3)

    #================= Debiasing =================
    num_reference_models = p_x_ref.shape[1]
    debias_scores = []
    debias_memberhips = []
    # when the number of the reference models is small (e.g., < 4), 
    # it is helpful to seperate the reference models used for debiasing and the reference models used for MIA 
    # if the MIA score is greatly influenced by the reference model outputs.
    # (e.g., In RMIA, when using a single reference model, the denominator is a deterministic linear transformation of 
    # f_ref(x), rather than an approximate distribution as in LiRA.)
    for i in range(num_reference_models):
        p_x_tar_i = p_x_ref[:, i][:, np.newaxis]
        p_z_tar_i = p_z_ref[:, i][:, np.newaxis]
        true_membership_x = memberships_x[:, i][:, np.newaxis]
        true_membership_z = memberships_z[:, i][:, np.newaxis]

        # drop selected reference model from the reference set
        p_x_ref_new = np.delete(p_x_ref, i, axis=1)
        p_z_ref_new = np.delete(p_z_ref, i, axis=1)
        memberships_x_new = np.delete(memberships_x, i, axis=1)
        memberships_z_new = np.delete(memberships_z, i, axis=1)

        _, rmia_ratio_ref_i = rmia(p_x_tar_i, p_x_ref_new, p_z_tar_i, p_z_ref_new, memberships_x_new, memberships_z_new, offline_a = 0.3)
        debias_scores.append(rmia_ratio_ref_i)
        debias_memberhips.append(true_membership_x)
    debias_scores = np.array(debias_scores).ravel()
    debias_memberhips = np.array(debias_memberhips).ravel()

    #================= Get the best threshold (maximizing the tpr-fpr) and TPR and FPR for debiasing =================
    fpr, tpr, auc, acc, threshs = sweep(debias_scores, debias_memberhips)
    best_thresh, tpr_at_best_thresh, fpr_at_best_thresh = select_robust_threshold(tpr, fpr, threshs, debias_scores, debias_memberhips)
    
    #================= Get the debiased prediction =================
    raw_preds = rmia_ratios >= best_thresh
    raw_agg_mia_p = np.mean(raw_preds)
    raw_agg_score_p = np.mean(rmia_scores)
    debiased_preds = (raw_preds - fpr_at_best_thresh)/(tpr_at_best_thresh - fpr_at_best_thresh)
    debiased_p = np.mean(debiased_preds)
    debiased_p = np.clip(debiased_p, 0, 1)

    return raw_agg_mia_p, raw_agg_score_p, debiased_p
    




