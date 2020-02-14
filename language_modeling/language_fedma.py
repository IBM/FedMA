import copy
import logging
import numpy as np

from lapsolver import solve_dense

##########################
# For the use of SPAHM
##########################
#from gaus_marginal_matching import match_local_atoms


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def row_param_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j):

    match_norms = ((weights_j_l + global_weights) ** 2 / (sigma_inv_j + global_sigmas)).sum(axis=1) - (
                global_weights ** 2 / global_sigmas).sum(axis=1)

    return match_norms


def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J):

    Lj = weights_j.shape[0]
    counts = np.minimum(np.array(popularity_counts), 10)
    param_cost = np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)])
    param_cost += np.log(counts / (J - counts))

    ## Nonparametric cost
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added))
    cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * np.log(gamma / J)

    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost


def matching_upd_j(weights_j, global_weights, sigma_inv_j, global_sigmas, prior_mean_norm, prior_inv_sigma,
                   popularity_counts, gamma, J):

    L = global_weights.shape[0]

    full_cost = compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                             popularity_counts, gamma, J)

    #row_ind, col_ind = linear_sum_assignment(-full_cost)
    # please note that this can not run on non-Linux systems
    row_ind, col_ind = solve_dense(-full_cost)

    assignment_j = []

    new_L = L

    for l, i in zip(row_ind, col_ind):
        if i < L:
            popularity_counts[i] += 1
            assignment_j.append(i)
            global_weights[i] += weights_j[l]
            global_sigmas[i] += sigma_inv_j
        else:  # new neuron
            popularity_counts += [1]
            assignment_j.append(new_L)
            new_L += 1
            global_weights = np.vstack((global_weights, prior_mean_norm + weights_j[l]))
            global_sigmas = np.vstack((global_sigmas, prior_inv_sigma + sigma_inv_j))

    return global_weights, global_sigmas, popularity_counts, assignment_j


def objective(global_weights, global_sigmas):
    obj = ((global_weights) ** 2 / global_sigmas).sum()
    return obj


def patch_weights(w_j, L_next, assignment_j_c):
    if assignment_j_c is None:
        return w_j
    new_w_j = np.zeros((w_j.shape[0], L_next))
    new_w_j[:, assignment_j_c] = w_j
    return new_w_j


def split_weights(weight):
    '''
    we reconstruct the w_ii|w_io|w_ig|w_if matrices here
    '''
    split_range = np.split(np.arange(weight.shape[0]), 4)
    i_weights = [weight[indices, :] for indices in split_range]
    return np.hstack(i_weights)


def split_bias(bias):
    tempt_biases = np.split(bias, 4)
    return np.vstack(tempt_biases).T


def revert_split_weights(weight):
    '''
    we reconstruct the w_ii|w_io|w_ig|w_if matrices here
    '''
    split_range = np.split(np.arange(weight.shape[1]), 4)
    i_weights = [weight[:, indices] for indices in split_range]
    return np.vstack(i_weights)


def revert_split_bias(bias):
    split_range = np.split(np.arange(bias.shape[1]), 4)
    i_bias = [bias[:, indices] for indices in split_range]
    return np.vstack(i_bias)[:, 0]


def process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0):
    J = len(batch_weights)
    sigma_bias = sigma
    sigma0_bias = sigma0
    mu0_bias = 0.1
    softmax_bias = [batch_weights[j][-1] for j in range(J)]
    softmax_inv_sigma = [s / sigma_bias for s in last_layer_const]
    softmax_bias = sum([b * s for b, s in zip(softmax_bias, softmax_inv_sigma)]) + mu0_bias / sigma0_bias
    softmax_inv_sigma = 1 / sigma0_bias + sum(softmax_inv_sigma)
    return softmax_bias, softmax_inv_sigma


def match_layer(weights_bias, sigma_inv_layer, mean_prior, sigma_inv_prior, gamma, it):
    J = len(weights_bias)

    group_order = sorted(range(J), key=lambda x: -weights_bias[x].shape[0])

    batch_weights_norm = [w * s for w, s in zip(weights_bias, sigma_inv_layer)]
    prior_mean_norm = mean_prior * sigma_inv_prior

    global_weights = prior_mean_norm + batch_weights_norm[group_order[0]]
    global_sigmas = np.outer(np.ones(global_weights.shape[0]), sigma_inv_prior + sigma_inv_layer[group_order[0]])

    popularity_counts = [1] * global_weights.shape[0]

    assignment = [[] for _ in range(J)]

    assignment[group_order[0]] = list(range(global_weights.shape[0]))

    ## Initialize
    for j in group_order[1:]:
        global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                        global_weights,
                                                                                        sigma_inv_layer[j],
                                                                                        global_sigmas, prior_mean_norm,
                                                                                        sigma_inv_prior,
                                                                                        popularity_counts, gamma, J)
        assignment[j] = assignment_j

    ## Iterate over groups
    for iteration in range(it):
        random_order = np.random.permutation(J)
        for j in random_order:  # random_order:
            to_delete = []
            ## Remove j
            Lj = len(assignment[j])
            for l, i in sorted(zip(range(Lj), assignment[j]), key=lambda x: -x[1]):
                popularity_counts[i] -= 1
                if popularity_counts[i] == 0:
                    del popularity_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, l_ind in enumerate(assignment[j_clean]):
                            if i < l_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                            elif i == l_ind and j_clean != j:
                                logger.info('Warning - weird unmatching')
                else:
                    global_weights[i] = global_weights[i] - batch_weights_norm[j][l]
                    global_sigmas[i] -= sigma_inv_layer[j]

            global_weights = np.delete(global_weights, to_delete, axis=0)
            global_sigmas = np.delete(global_sigmas, to_delete, axis=0)

            ## Match j
            global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                            global_weights,
                                                                                            sigma_inv_layer[j],
                                                                                            global_sigmas,
                                                                                            prior_mean_norm,
                                                                                            sigma_inv_prior,
                                                                                            popularity_counts, gamma, J)
            assignment[j] = assignment_j

    logger.info('Number of global neurons is %d, gamma %f' % (global_weights.shape[0], gamma))
    logger.info("***************Shape of global weights after match: {} ******************".format(global_weights.shape))
    return assignment, global_weights, global_sigmas, popularity_counts


def layerwise_fedma(batch_weights, layer_index, sigma_layers, 
                                sigma0_layers, gamma_layers, it,
                                n_layers, matching_shapes):
    """
    We implement a layer-wise matching here:
    """
    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]

    # J: number of workers
    J = len(batch_weights)
    # init_num_kernel: the number of conv filters in the first conv layer 
    
    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    assignment_c = [None for j in range(J)]
    L_next = None

    sigma = sigma_layers[layer_index - 1]
    sigma_bias = sigma_bias_layers[layer_index - 1]
    gamma = gamma_layers[layer_index - 1]
    sigma0 = sigma0_layers[layer_index - 1]
    sigma0_bias = sigma0_bias_layers[layer_index - 1]

    if layer_index < 1:
        sentence_length = batch_weights[0][layer_index].T.shape[1]
        weights_bias = [batch_weights[j][layer_index].T for j in range(J)]

        sigma_inv_prior = np.array(sentence_length * [1 / sigma0])
        mean_prior = np.array(sentence_length * [mu0])
        sigma_inv_layer = [np.array(sentence_length * [1 / sigma]) for j in range(J)]

    elif layer_index == (n_layers - 1) and n_layers > 2:
        # our assumption is that this branch will consistently handle the last fc layers
        reconstructed_weights = [split_weights(batch_weights[j][layer_index]) for j in range(J)]
        reconstructed_bias = [split_bias(batch_weights[j][layer_index+2]) for j in range(J)]
        weights_bias = [np.hstack((reconstructed_weights[j], reconstructed_bias[j])) for j in range(J)]

        sigma_inv_prior = np.array((weights_bias[0].shape[1] - 4) * [1 / sigma0] + [1 / sigma0_bias] * 4)
        mean_prior = np.array((weights_bias[0].shape[1] - 4) * [mu0] + [mu0_bias] * 4)
        sigma_inv_layer = [np.array((weights_bias[0].shape[1] - 4) * [1 / sigma] + [1 / sigma_bias] * 4) for j in range(J)]

    elif (layer_index >= 1 and layer_index < (n_layers - 1)):
        # our assumption is that this branch will consistently handle the last fc layers
        if layer_index == 1:
            reconstructed_weights = [split_weights(batch_weights[j][layer_index]) for j in range(J)]
            reconstructed_bias = [split_bias(batch_weights[j][layer_index+2]) for j in range(J)]
            weights_bias = [np.hstack((reconstructed_weights[j], reconstructed_bias[j])) for j in range(J)]
        elif layer_index == 2:
            reconstructed_weights = [split_weights(batch_weights[j][layer_index+3]) for j in range(J)]
            reconstructed_bias = [split_bias(batch_weights[j][layer_index+3+2]) for j in range(J)]
            weights_bias = [np.hstack((reconstructed_weights[j], reconstructed_bias[j])) for j in range(J)]           


        sigma_inv_prior = np.array((weights_bias[0].shape[1] - 4) * [1 / sigma0] + [1 / sigma0_bias] * 4)
        mean_prior = np.array((weights_bias[0].shape[1] - 4) * [mu0] + [mu0_bias] * 4)
        sigma_inv_layer = [np.array((weights_bias[0].shape[1] - 4) * [1 / sigma] + [1 / sigma_bias] * 4) for j in range(J)]

    logger.info("weights bias: {}".format(weights_bias[0].shape))
    logger.info("sigma_inv_prior shape: {}".format(sigma_inv_prior.shape))
    logger.info("mean_prior shape: {}".format(mean_prior.shape))

    ########################################
    # For the use of PFNM (https://github.com/IBM/probabilistic-federated-neural-matching)
    ########################################
    assignment_c, global_weights_c, global_sigmas_c, popularity_counts = match_layer(weights_bias, sigma_inv_layer, mean_prior,
                                                                  sigma_inv_prior, gamma, it)

    ########################################
    # For the use of SPAHM (https://github.com/IBM/SPAHM)
    ########################################
    #assignment_c, global_weights_c, popularity_counts, hyper_params = match_local_atoms(local_atoms=weights_bias, 
    #                                                                sigma=sigma, sigma0=sigma0, gamma=gamma, 
    #                                                                it=it, optimize_hyper=True)

    logger.info("After matching layer: {}, the matched weight shape: {}, popularity_counts: {}, popularity_counts length: {}".format(
                        layer_index, global_weights_c.shape, popularity_counts, len(popularity_counts)))
    #logger.info("assignment 0: {}".format(assignment_c[0]))
    #logger.info("assignment 1: {}".format(assignment_c[1]))

    L_next = global_weights_c.shape[0]
    if layer_index < 1:
        global_weights_out = [global_weights_c]
        global_inv_sigmas_out = [global_sigmas_c]

        logger.info("Branch A, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))

    elif layer_index == (n_layers - 1) and n_layers > 2:
        pass

    elif (layer_index >= 1 and layer_index < (n_layers - 1)):
        #gwc_shape = global_weights_c.shape
        #global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1].T, global_weights_c[:, gwc_shape[1]-1]]
        #global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1].T, global_sigmas_c[:, gwc_shape[1]-1]]
        reconstructed_weights_shape = reconstructed_weights[0].shape
        global_weights_out = [revert_split_weights(global_weights_c[:, 0:reconstructed_weights_shape[1]]), revert_split_bias(global_weights_c[:, reconstructed_weights_shape[1]:])]
        global_inv_sigmas_out = [revert_split_weights(global_sigmas_c[:, 0:reconstructed_weights_shape[1]]), revert_split_bias(global_sigmas_c[:, reconstructed_weights_shape[1]:])]

        logger.info("Branch layer index, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))
    #logger.info("global inv sigma out shape: {}".format([giso.shape for giso in global_inv_sigmas_out]))
    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
    #return map_out, assignment_c, L_next
    return map_out, assignment_c, L_next, np.array(popularity_counts).astype(np.float32)
    #return global_weights_out, assignment_c, L_next