# part of the code blocks are modified from: 
# https://github.com/litian96/FedProx/blob/master/flearn/models/shakespeare/stacked_lstm.py
# by Hongyi Wang (hwang595 @ GitHub)
# credit goes to: Tian Li (litian96 @ GitHub)

import json
import logging
import numpy as np
import time
import math
import pickle
import copy

from itertools import product
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from language_utils import *
from language_frb import layerwise_fedma
from language_frb import patch_weights


import language_model

#from combine_nets import prepare_uniform_weights, prepare_sanity_weights, prepare_weight_matrix, normalize_weights, get_weighted_average_pred


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


BATCH_SIZE = 50
TRAIN_DATA_DIR = "./datum/leaf/data/shakespeare/data/train/"
TEST_DATA_DIR = "./datum/leaf/data/shakespeare/data/test/"

TRAIN_DATA_NAME = "all_data_niid_0_keep_10000_train_9.json"
TEST_DATA_NAME = "all_data_niid_0_keep_10000_test_9.json"

#TRIAL_USER_NAME = ["THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_HOT", "KING_RICHARD_THE_SECOND_KING_RICHARD"]

TRIAL_EPOCH=10

# since we used a relatively "fixed" model for shakespeare dataset
# we thus hardcode it here
NUM_LAYERS=3 # we start from 1-layer LSTM now (so the 3 layers now is encoder|hidden LSTM|decoder)

#NUM_LAYERS=4 # 2-layer LSTM (4 layers: encoder|hidden LSTM1|hidden LSTM2|decoder)


def prepare_weight_matrix(n_classes, weights: dict):
    weights_list = {}

    for net_i, cls_cnts in weights.items():
        cls = np.array(list(cls_cnts.keys()))
        cnts = np.array(list(cls_cnts.values()))
        weights_list[net_i] = np.array([0] * n_classes, dtype=np.float32)
        weights_list[net_i][cls] = cnts
        weights_list[net_i] = torch.from_numpy(weights_list[net_i]).view(1, -1)

    return weights_list


def prepare_uniform_weights(n_classes, net_cnt, fill_val=1):
    weights_list = {}

    for net_i in range(net_cnt):
        temp = np.array([fill_val] * n_classes, dtype=np.float32)
        weights_list[net_i] = torch.from_numpy(temp).view(1, -1)

    return weights_list


def prepare_sanity_weights(n_classes, net_cnt):
    return prepare_uniform_weights(n_classes, net_cnt, fill_val=0)


def normalize_weights(weights):
    Z = np.array([])
    eps = 1e-6
    weights_norm = {}

    for _, weight in weights.items():
        if len(Z) == 0:
            Z = weight.data.numpy()
        else:
            Z = Z + weight.data.numpy()

    for mi, weight in weights.items():
        weights_norm[mi] = weight / torch.from_numpy(Z + eps)

    return weights_norm


def get_weighted_average_pred(models: list, weights: dict, x, hidden_list, device="cpu"):
    out_weighted = None

    # Compute the predictions
    for model_i, model in enumerate(models):
        #logger.info("Model: {}".format(next(model.parameters()).device))
        #logger.info("data device: {}".format(x.device))
        hidden_test = hidden_list[model_i]
        hidden_test = repackage_hidden(hidden_test)
        #logger.info("x device: {}, hidden_test device: {}".format(x.device, hidden_test[0].device))
        output, hidden_test = model(x, hidden_test)
        hidden_list[model_i] = hidden_test

        out = F.softmax(output.t(), dim=-1)  # (N, C)

        weight = weights[model_i].to(device)

        if out_weighted is None:
            weight = weight.to(device)
            out_weighted = (out * weight)
        else:
            out_weighted += (out * weight)

    return out_weighted.t(), hidden_list


def pdm_prepare_weights(nets):
    weights = []

    for net_i, net in enumerate(nets):
        layer_i = 0
        statedict = net.state_dict()
        net_weights = []
        while True:

            if ('layers.%d.weight' % layer_i) not in statedict.keys():
                break

            layer_weight = statedict['layers.%d.weight' % layer_i].numpy().T
            layer_bias = statedict['layers.%d.bias' % layer_i].numpy()

            net_weights.extend([layer_weight, layer_bias])
            layer_i += 1

        weights.append(net_weights)

    return weights


def pdm_prepare_weights_cnn(nets):
    """
    Note that we only handle the FC parts and leave the conv layers as is
    """
    weights = []

    for net_i, net in enumerate(nets):
        layer_i = 0
        statedict = net.state_dict()

        net_weights = [np.zeros(statedict['fc1.weight'].numpy().T.shape[0]), # add a dummy layer
                        statedict['fc1.weight'].numpy().T,
                        statedict['fc1.bias'].numpy(),
                        statedict['fc2.weight'].numpy().T,
                        statedict['fc2.bias'].numpy(),
                        statedict['fc3.weight'].numpy().T,
                        statedict['fc3.bias'].numpy()]

        weights.append(net_weights)

    return weights


def pdm_prepare_freq(cls_freqs, n_classes):
    freqs = []

    for net_i in sorted(cls_freqs.keys()):
        net_freqs = [0] * n_classes

        for cls_i in cls_freqs[net_i]:
            net_freqs[cls_i] = cls_freqs[net_i][cls_i]

        freqs.append(np.array(net_freqs))

    return freqs


def compute_pdm_net_accuracy(weights, train_dl, test_dl, n_classes):

    dims = []
    dims.append(weights[0].shape[0])

    for i in range(0, len(weights), 2):
        dims.append(weights[i].shape[1])

    ip_dim = dims[0]
    op_dim = dims[-1]
    hidden_dims = dims[1:-1]

    logger.info("IP dim of matched NN: {}. OP dim of matched NN: {}, Hidden Dims of Matched NN: {}".format(ip_dim, op_dim, hidden_dims))
    logger.info("*"*30)

    pdm_net = FcNet(ip_dim, hidden_dims, op_dim)
    statedict = pdm_net.state_dict()

    i = 0
    layer_i = 0
    while i < len(weights):
        weight = weights[i]
        i += 1
        bias = weights[i]
        i += 1

        statedict['layers.%d.weight' % layer_i] = torch.from_numpy(weight.T)
        statedict['layers.%d.bias' % layer_i] = torch.from_numpy(bias)
        layer_i += 1

    pdm_net.load_state_dict(statedict)

    train_acc, conf_matrix_train = compute_ensemble_accuracy([pdm_net], train_dl, n_classes, uniform_weights=True)
    test_acc, conf_matrix_test = compute_ensemble_accuracy([pdm_net], test_dl, n_classes, uniform_weights=True)

    return train_acc, test_acc, conf_matrix_train, conf_matrix_test


def compute_pdm_cnn_accuracy(models, weights, train_dl, test_dl, n_classes):
    """Note that we only handle the FC weights for now"""
    # we need to figure out the FC dims first
    matched_weights = weights[1:] # get rid of the dummy layer, this should be deprecated later
    input_dim = matched_weights[0].shape[0] # hard coded for now, will make changes later
    hidden_dims = [matched_weights[0].shape[1], matched_weights[2].shape[1]]
    output_dim = matched_weights[-1].shape[0]

    logger.info("Input dim: {}, hidden_dims: {}, output_dim: {}".format(input_dim, hidden_dims, output_dim))
    #book_keeper = {4:0, 5:1, 6:2, 7:3, 8:4, 9:5}

    unmatched_cnn_blocks = []
    for model_i, model in enumerate(models):
        tempt_cnn = ConvBlock()
        
        new_state_dict = {}
        model_counter = 0
        # handle the conv layers part which is not changing
        for param_idx, (key_name, param) in enumerate(tempt_cnn.state_dict().items()):
            if "conv" in key_name:
                temp_dict = {key_name: models[model_i].state_dict()[key_name]}
            new_state_dict.update(temp_dict)
            model_counter += 1
        #for (k, v) in new_state_dict.items():
        #    print("New state dict key: {}, val: {}".format(k, v))
        tempt_cnn.load_state_dict(new_state_dict)
        unmatched_cnn_blocks.append(tempt_cnn)

    matched_state_dict = {}
    matched_fcs = FCBlock(input_dim, hidden_dims, output_dim)
    for param_idx, (key_name, param) in enumerate(matched_fcs.state_dict().items()):
        if "weight" in key_name:
            temp_dict = {key_name: torch.from_numpy(matched_weights[param_idx].T)}
        elif "bias" in key_name:
            temp_dict = {key_name: torch.from_numpy(matched_weights[param_idx])}
        new_state_dict.update(temp_dict)
    matched_fcs.load_state_dict(new_state_dict)

    logger.info("Cnn info:")
    for (k, v) in unmatched_cnn_blocks[0].items():
        logger.info("Cnn blocks keys: {}, values: {}".format(k, v))

    logger.info("fc info:")
    for (k, v) in matched_fcs.items():
        logger.info("FC blocks keys: {}, values: {}".format(k, v))



def compute_pdm_matching_multilayer(models, train_dl, test_dl, cls_freqs, n_classes, sigma0=None, it=0, sigma=None, gamma=None):
    #batch_weights = pdm_prepare_weights(models)
    batch_weights = pdm_prepare_weights_cnn(models)
    
    # gather the weights and biases of each layers of the FC_NN
    for i, weights in enumerate(batch_weights):
        for w in weights:
            logger.info(w.shape)
            logger.info("*"*20)
        logger.info("Batch index: {}".format(i)+"="*30)

    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    res = {}
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    #gammas = [1.0, 1e-3, 50.0] if gamma is None else [gamma]
    #sigmas = [1.0, 0.1, 0.5] if sigma is None else [sigma]
    #sigma0s = [1.0, 10.0] if sigma0 is None else [sigma0]
    gammas = [30.0]
    sigmas = [1.0]
    sigma0s = [1.0]

    for gamma, sigma, sigma0 in product(gammas, sigmas, sigma0s):
        
        logger.info("Gamma: ", gamma, "Sigma: ", sigma, "Sigma0: ", sigma0)

        hungarian_weights = pdm_multilayer_group_descent(
            batch_weights, sigma0_layers=sigma0, sigma_layers=sigma, batch_frequencies=batch_freqs, it=it, gamma_layers=gamma
        )

        for i, w in enumerate(hungarian_weights):
            logger.info("Hungarian weight index: {}, Hungarian weight shape: {}".format(i, w.shape))
        #exit()

        #train_acc, test_acc, _, _ = compute_pdm_net_accuracy(hungarian_weights, train_dl, test_dl, n_classes)
        train_acc, test_acc, _, _ = compute_pdm_cnn_accuracy(models, hungarian_weights, train_dl, test_dl, n_classes)
        exit()

        key = (sigma0, sigma, gamma)
        res[key] = {}
        res[key]['shapes'] = list(map(lambda x: x.shape, hungarian_weights))
        res[key]['train_accuracy'] = train_acc
        res[key]['test_accuracy'] = test_acc

        logger.info('Sigma0: %s. Sigma: %s. Shapes: %s, Accuracy: %f' % (
        str(sigma0), str(sigma), str(res[key]['shapes']), test_acc))

        if train_acc > best_train_acc:
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_weights = hungarian_weights
            best_sigma = sigma
            best_gamma = gamma
            best_sigma0 = sigma0

    logger.info('Best sigma0: %f, Best sigma: %f, Best Gamma: %f, Best accuracy (Test): %f. Training acc: %f' % (
    best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc))

    return (best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc, best_weights, res)


def compute_iterative_pdm_matching(models, train_dl, test_dl, cls_freqs, n_classes, sigma, sigma0, gamma, it, old_assignment=None):

    batch_weights = pdm_prepare_weights(models)
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)

    hungarian_weights, assignments = pdm_iterative_layer_group_descent(
        batch_weights, batch_freqs, sigma_layers=sigma, sigma0_layers=sigma0, gamma_layers=gamma, it=it, assignments_old=old_assignment
    )

    train_acc, test_acc, conf_matrix_train, conf_matrix_test = compute_pdm_net_accuracy(hungarian_weights, train_dl, test_dl, n_classes)

    batch_weights_new = [pdm_build_init(hungarian_weights, assignments, j) for j in range(len(models))]
    matched_net_shapes = list(map(lambda x: x.shape, hungarian_weights))

    return batch_weights_new, train_acc, test_acc, matched_net_shapes, assignments, hungarian_weights, conf_matrix_train, conf_matrix_test


def flatten_weights(weights_j):
    flat_weights = np.hstack((weights_j[0].T, weights_j[1].reshape(-1,1), weights_j[2]))
    return flat_weights


def build_network(clusters, batch_weights, D):
    cluster_network = [clusters[:,:D].T, clusters[:,D].T, clusters[:,(D+1):]]
    bias = np.mean(batch_weights, axis=0)[-1]
    cluster_network += [bias]
    return cluster_network


def compute_ensemble_accuracy(models: list, global_test_data, global_test_label, n_classes, 
    global_num_samples_test, device="cpu"):

    correct, total = 0, 0
    true_labels_list, pred_labels_list = np.array([]), np.array([])
    global_eval_batch_size = 10

    was_training = [False]*len(models)
    for i, model in enumerate(models):
        if model.training:
            was_training[i] = True
            model.eval()

    hidden_list = []
    for m in models:
        hidden_test = m.init_hidden(global_eval_batch_size)
        hidden_list.append(hidden_test)

    weights_list = prepare_uniform_weights(n_classes, len(models))
    weights_norm = normalize_weights(weights_list)

    with torch.no_grad():
        for i in range(int(global_num_samples_test / global_eval_batch_size)):
            input_data, target_data = process_x(global_test_data[global_eval_batch_size*i:global_eval_batch_size*(i+1)]), process_y(global_test_label[global_eval_batch_size*i:global_eval_batch_size*(i+1)])
            data, targets = torch.from_numpy(input_data).to(device), torch.from_numpy(target_data).to(device)
            out, hidden_list = get_weighted_average_pred(models, weights_norm, data, hidden_list=hidden_list, device=device)
            #_, pred_label = torch.max(out, 1)
            _, pred_label = torch.max(out.t(), 1)
            correct += (pred_label == torch.max(targets, 1)[1]).sum().item()
            logger.info("Correct: {}, data? :{}, batch index: {}/{}".format(correct, data.data.size()[0], i, int(global_num_samples_test / global_eval_batch_size)))

    logger.info('*' * 89)
    logger.info("Ensemble result: Correct: {}, Total: {}, Accs: {}".format(correct, global_num_samples_test, correct / float(global_num_samples_test)))
    logger.info('*' * 89)
    return correct / float(global_num_samples_test)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def collect_weights(models):
    logger.info("Collecting weights ...")
    collected_weights = []
    for model_index, model in enumerate(models):
        param_vals = []
        for param_index, (name, param) in enumerate(model.named_parameters()):
            logger.info("Layer index: {}, Layer name: {}, Layer shape: {}".format(param_index, name, param.size()))
            param_vals.append(param.cpu().detach().numpy())
        collected_weights.append(param_vals)
    return collected_weights



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Experiment running on device: {}".format(device))

    with open(TRAIN_DATA_DIR+TRAIN_DATA_NAME) as json_file:
        train_data = json.load(json_file)

    with open(TEST_DATA_DIR+TEST_DATA_NAME) as json_file:
        test_data = json.load(json_file)

    lr = 0.8
    clip = 0.25
    n_clients = 66
    retrain_flag = False


    TRIAL_USER_NAME = train_data["users"][0:n_clients] #this can be of length in range from 1 to 132
    start_time = time.time()
    total_loss = 0.0

    nets_list = []
    criterion = nn.CrossEntropyLoss()

    for client_index in range(n_clients):
        if retrain_flag:
            logger.info("Start local training process for client {} ....".format(client_index))
            client_user_name = TRIAL_USER_NAME[client_index]
            num_samples_train = len(train_data["user_data"][client_user_name]['x'])
            num_samples_test = len(test_data["user_data"][client_user_name]['x'])

            user_train_data = train_data["user_data"][client_user_name]
            user_test_data = test_data["user_data"][client_user_name]

            model = language_model.RNNModel('LSTM', 80, 8, 256, 1, 0.2, tie_weights=False).to(device)
            #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001, amsgrad=True)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

            for epoch in range(TRIAL_EPOCH):
                model.train()
                epoch_start_time = time.time()

                hidden_train = model.init_hidden(BATCH_SIZE)
                for i in range(int(num_samples_train / BATCH_SIZE)):
                    input_data, target_data = process_x(user_train_data['x'][BATCH_SIZE*i:BATCH_SIZE*(i+1)]), process_y(user_train_data['y'][BATCH_SIZE*i:BATCH_SIZE*(i+1)])
      
                    data, targets = torch.from_numpy(input_data).to(device), torch.from_numpy(target_data).to(device)
                    optimizer.zero_grad()

                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = model(data, hidden_train)

                    loss = criterion(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    cur_loss = total_loss
                    elapsed = time.time() - start_time
                    #logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    #        'loss {:5.2f} | ppl {:8.2f}'.format(
                    #    epoch, i, num_samples_train // BATCH_SIZE, lr,
                    #    elapsed * 1000, cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()

                eval_batch_size = 10
                model.eval()
                total_val_loss = 0.
                ntokens = 80
                hidden_test = model.init_hidden(eval_batch_size)
                correct_prediction = 0
                with torch.no_grad():
                    for i in range(int(num_samples_test / eval_batch_size)):
                        input_data, target_data = process_x(user_test_data['x'][eval_batch_size*i:eval_batch_size*(i+1)]), process_y(user_test_data['y'][eval_batch_size*i:eval_batch_size*(i+1)])
                        data, targets = torch.from_numpy(input_data).to(device), torch.from_numpy(target_data).to(device)

                        hidden_test = repackage_hidden(hidden_test)
                        output, hidden_test = model(data, hidden_test)
                        loss = criterion(output.t(), torch.max(targets, 1)[1])
                        _, pred_label = torch.max(output.t(), 1)
                        correct_prediction += (pred_label == torch.max(targets, 1)[1]).sum().item()

                        total_val_loss += loss.item()
                logger.info('-' * 89)
                logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | pred: {}/{} | acc: {:.4f}%'.format(epoch, (time.time() - epoch_start_time),
                                                       total_val_loss, correct_prediction, num_samples_test, correct_prediction/num_samples_test*100.0))
                logger.info('-' * 89)
            
            nets_list.append(model)

            # we save the trained model here:
            with open("trained_model_client_{}".format(client_index), 'wb') as trained_model_file:
                pickle.dump(model, trained_model_file)
        else:
            # load the trained model from local disk, and forward
            with open("trained_model_client_{}".format(client_index), 'rb') as trained_model_file:
                model = pickle.load(trained_model_file)
                nets_list.append(model)


    # we will need to construct a new global test set based on all test data on each of the clients
    global_test_data = []
    global_test_label = []
    global_num_samples_test = 0
    global_num_samples_train = 0
    for client_index in range(n_clients):
        client_user_name = TRIAL_USER_NAME[client_index]
        global_num_samples_test += len(test_data["user_data"][client_user_name]['x'])
        global_num_samples_train += len(train_data["user_data"][client_user_name]['x'])
        global_test_data += test_data["user_data"][client_user_name]['x']
        global_test_label += test_data["user_data"][client_user_name]['y']

    logger.info("Total number of training data points: {}".format(global_num_samples_train))
    logger.info("Total number of test data points: {}".format(global_num_samples_test))
    global_eval_batch_size = 10

    compute_ensemble_accuracy(models=nets_list, global_test_data=global_test_data, global_test_label=global_test_label, n_classes=80, global_num_samples_test=global_num_samples_test,
            device=device)
