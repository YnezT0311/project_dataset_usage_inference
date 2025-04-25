import numpy as np

def get_p_denominator(p_ref, memberships, offline_a):
    # Compute p_x
    non_memberships = 1 - memberships
    p_out = p_ref * non_memberships
    # Sort the signals such that only the non-zero signals (out signals) for each sample are kept
    num_reference_models = p_ref.shape[1]
    
    if num_reference_models > 1:
        p_out = -np.sort(-p_out, axis=1)[:, :num_reference_models] #TODO: check whether devided by 2
    else:
        # Derive according to ((1+a)P_out + (1-a))/2 = P(x) = (P_out + P_in)/2
        if offline_a != 0:
            p_out += ((p_ref + offline_a - 1)/offline_a) * memberships
        else:
            p_out += ((p_ref - 0.7)/0.3) * memberships

    mean_out = np.mean(p_out, axis=1)
    p_denominator = (1 + offline_a) / 2 * mean_out + (1 - offline_a) / 2
    return p_denominator

def rmia(p_x_tar, p_x_ref, p_z_tar, p_z_ref, memberships_x, memberships_z, offline_a = 0.3, normalize = False):
    """
    Compute the RMIA score (offline) for each entry in the target book.

    Args:
        p_x_tar (np.ndarray): The predicted probabilities over the entries in the target book on target models. Shape: (num_target_entries, num_target_models).
        p_x_ref (np.ndarray): The predicted probabilities over the entries in the target book on reference models. Shape: (num_target_entries, num_reference_models).
        p_z_tar (np.ndarray): The predicted probabilities over the entries in the population book on target models. Shape: (num_population_entries, num_target_models).
        p_z_ref (np.ndarray): The predicted probabilities over the entries in the population book on reference models. Shape: (num_population_entries, num_reference_models).
        memberships_x (np.ndarray): The memberships of the target book entries on reference models. Shape: (num_target_entries, num_reference_models).
        memberships_z (np.ndarray): The memberships of the population book entries on reference models. Shape: (num_population_entries, num_reference_models).
        offline_a (float): The offline parameter for RMIA. Default: 0.3.
    
    Returns:
        np.ndarray: The RMIA score for each entry in the target book. Shape: (num_target_entries, num_target_models).
    """
    # If the number of reference is enough (e.g., > 2), we can set the normalization to be True. 
    # Then, prob_ratio_x (without population data) can be used for the most effective MIA
    # If the number of reference is not enough, we can set the normalization to be False.
    # Then, rmia_scores (with population data) can be used for effective and stable MIA

    if normalize:
        p_x = get_p_denominator(p_x_ref, memberships_x, offline_a)[:, np.newaxis]
        prob_ratio_x = p_x_tar / p_x

        p_z = get_p_denominator(p_z_ref, memberships_z, offline_a)[:, np.newaxis]
        prob_ratio_z = p_z_tar / p_z
    else:
        prob_ratio_x = p_x_tar
        prob_ratio_z = p_z_tar

    rmia_scores = []
    for i in range(p_x_tar.shape[1]):
        i_prob_ratio_x = prob_ratio_x[:, i]
        i_prob_ratio_z = prob_ratio_z[:, i]
        ratio = i_prob_ratio_x[:, np.newaxis] / i_prob_ratio_z # shape: (num_target_entries, num_population_entries)
        count = np.average(ratio > 1.0, axis=1)
        rmia_scores.append(count)
    return np.array(rmia_scores).T, np.array(prob_ratio_x)