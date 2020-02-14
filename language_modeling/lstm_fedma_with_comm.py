# part of the code blocks are modified from: https://github.com/litian96/FedProx/blob/master/flearn/models/shakespeare/stacked_lstm.py
# credit goes to: Tian Li (litian96 @ GitHub)

import json
import logging
import numpy as np
import time
import math
import pickle
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from language_utils import *
from language_fedma import layerwise_fedma
from language_fedma import patch_weights

import language_model


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


BATCH_SIZE = 50
TRAIN_DATA_DIR = "./datum/leaf/data/shakespeare/data/train/"
TEST_DATA_DIR = "./datum/leaf/data/shakespeare/data/test/"

# all_data_niid_0_keep_10000_train_9.json
TRAIN_DATA_NAME = "all_data_niid_0_keep_10000_train_9.json"
TEST_DATA_NAME = "all_data_niid_0_keep_10000_test_9.json"

TRIAL_EPOCH=10

# since we used a relatively "fixed" model for shakespeare dataset
# we thus hardcode it here
#NUM_LAYERS=3 # we start from 1-layer LSTM now (so the 3 layers now is encoder|hidden LSTM|decoder)

NUM_LAYERS=3 # 2-layer LSTM (4 layers: encoder|hidden LSTM1|hidden LSTM2|decoder)


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
            #logger.info("Layer index: {}, Layer name: {}, Layer shape: {}".format(param_index, name, param.size()))
            param_vals.append(param.cpu().detach().numpy())
        collected_weights.append(param_vals)
    return collected_weights


def reconstruct_local_weights(global_weights, ori_assignment, client_index):
    def __reconstruct_x_weights(wegihts, assignments_row, assignments_col):
        slice_ranges = np.split(np.arange(wegihts.shape[0]), 4)
        res_weights = []
        for i in range(4):
            i_wegihts = wegihts[slice_ranges[i], :]
            reconstructed_i_weights_temp1 = i_wegihts[assignments_row, :]
            reconstructed_i_weights = reconstructed_i_weights_temp1[:, assignments_col]
            res_weights.append(reconstructed_i_weights)
        return np.vstack(res_weights)
    def __reconstruct_bias(bias, assignments_row):
        slice_ranges = np.split(np.arange(bias.shape[0]), 4)
        res_bias = [bias[slice_ranges[i]][assignments_row] for i in range(4)]
        return np.hstack(res_bias)
    def __reconstruct_h_weights(wegihts, assignments_row):
        slice_ranges = np.split(np.arange(wegihts.shape[0]), 4)
        res_weights = []
        for i in range(4):
            i_wegihts = wegihts[slice_ranges[i], :]
            reconstructed_i_weights_temp1 = i_wegihts[assignments_row, :]
            reconstructed_i_weights = reconstructed_i_weights_temp1[:, assignments_row]
            res_weights.append(reconstructed_i_weights)
        return np.vstack(res_weights)


    res_local_weights = []

    for layer_index in range(NUM_LAYERS-1):
        if layer_index == 0:
            # this one is simple, we only need to slice the 
            _assignment = ori_assignment[layer_index][client_index]
            _reconstructed_weight = global_weights[layer_index][:, _assignment]
            #logger.info("global_weights shape: {}, reconstructed shape: {}".format(global_weights[layer_index].shape, _reconstructed_weight.shape))
            res_local_weights.append(_reconstructed_weight)
        elif layer_index == 1:
            #lstm_net  = language_model.RNNModel('LSTM', 80, 8, 256, 2, 0.2, tie_weights=False)
            #for layer_index, (layer_name, layer_param) in enumerate(lstm_net.named_parameters()):
            #    logger.info("####### layer index: {}, layer name: {}, layer shape: {}".format(layer_index, layer_name, layer_param.size()))
            
            # the index we should work on is (1, 3) for reconstructing x weights and bias
            res_i_weights = __reconstruct_x_weights(global_weights[layer_index], 
                                            ori_assignment[layer_index][client_index], 
                                            ori_assignment[layer_index-1][client_index])

            res_i_bias = __reconstruct_bias(global_weights[layer_index+2], ori_assignment[layer_index][client_index])

            res_h_weights = __reconstruct_h_weights(global_weights[layer_index+1], 
                                            ori_assignment[layer_index][client_index])

            res_h_bias = __reconstruct_bias(global_weights[layer_index+3], ori_assignment[layer_index][client_index])

            res_local_weights.append(res_i_weights)
            res_local_weights.append(res_h_weights)
            res_local_weights.append(res_i_bias)
            res_local_weights.append(res_h_bias)

    # now we also need to handle the decoder layer
    global_decoder_weight = global_weights[-2]
    res_decoder_weight = global_decoder_weight[:, ori_assignment[-1][client_index]]

    res_local_weights.append(res_decoder_weight)
    res_local_weights.append(global_weights[-1])
    return res_local_weights


def local_retraining_process(n_clients, train_data, test_data, res_local_weight_list, lr, user_names):
    nets_list = []
    # construct a global test set
    global_test_data = []
    global_test_label = []
    global_num_samples_test = 0

    for client_index in range(n_clients):
        client_user_name = user_names[client_index]
        global_num_samples_test += len(test_data["user_data"][client_user_name]['x'])
        global_test_data += test_data["user_data"][client_user_name]['x']
        global_test_label += test_data["user_data"][client_user_name]['y']
    global_eval_batch_size = 10

    for client_index in range(n_clients):
        logger.info("Start local training process for client {} ....".format(client_index))
        client_user_name = user_names[client_index]
        num_samples_train = len(train_data["user_data"][client_user_name]['x'])
        num_samples_test = len(test_data["user_data"][client_user_name]['x'])

        user_train_data = train_data["user_data"][client_user_name]
        user_test_data = test_data["user_data"][client_user_name]
        model = language_model.RNNModel('LSTM', 80, 8, 256, 1, 0.2, tie_weights=False)
        criterion = nn.CrossEntropyLoss()
        ##### load the weights to the model
        ####################################
        new_state_dict = {}
        for param_idx, (key_name, param) in enumerate(model.state_dict().items()):
            #logger.info("param idx: {}, para shape: {}, client_index: {}, len res: {}".format(param_idx, param.shape, client_index, len(res_local_weight_list)))
            temp_dict = {key_name: torch.from_numpy(res_local_weight_list[client_index][param_idx])}
            new_state_dict.update(temp_dict)
        model.load_state_dict(new_state_dict)
        model.to(device)

        #######################################################
        # evaluate the model performance on global eval dataset
        #######################################################
        model.eval()

        eval_batch_size = 10
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
        logger.info('| Local Global Testset | client index: {} | valid loss {:5.2f} | pred: {}/{} | acc: {:.4f}%'.format(client_index, 
                                               total_val_loss, correct_prediction, num_samples_test, correct_prediction/num_samples_test*100.0))
        logger.info('-' * 89)

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        total_loss = 0.0
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
            logger.info('| end of epoch {:3d} | valid loss {:5.2f} | pred: {}/{} | acc: {:.4f}%'.format(epoch, 
                                                   total_val_loss, correct_prediction, num_samples_test, correct_prediction/num_samples_test*100.0))
            logger.info('-' * 89)
        nets_list.append(model)
    return nets_list


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Experiment running on device: {}".format(device))

    with open(TRAIN_DATA_DIR+TRAIN_DATA_NAME) as json_file:
        train_data = json.load(json_file)

    with open(TEST_DATA_DIR+TEST_DATA_NAME) as json_file:
        test_data = json.load(json_file)

    lr = 0.25
    clip = 0.25
    n_clients = 66
    retrain_flag = False
    communication_rounds = 10

    TRIAL_USER_NAME = train_data["users"][0:n_clients] #this can be of length in range from 1 to 132
    start_time = time.time()
    total_loss = 0.0

    logger.info("Learning rate: {}".format(lr))
    logger.info("number of clients: {}, len trial user name: {}".format(n_clients, len(TRIAL_USER_NAME)))

    for cr in range(communication_rounds):
        logger.info("Start to work on communication round-{}".format(cr))

        with open("lstm_matching_assignments", "rb") as assignment_file:
            assignments_list = pickle.load(assignment_file)
        with open("lstm_matching_shapes", "rb") as ms_file:
            matching_shapes = pickle.load(ms_file)
        with open("matched_global_weights", "rb") as matched_weight_file:
            global_matched_model = pickle.load(matched_weight_file)


        # we will need to construct a new global test set based on all test data on each of the clients
        global_test_data = []
        global_test_label = []
        global_num_samples_test = 0
        for client_index in range(n_clients):
            client_user_name = TRIAL_USER_NAME[client_index]
            global_num_samples_test += len(test_data["user_data"][client_user_name]['x'])
            global_test_data += test_data["user_data"][client_user_name]['x']
            global_test_label += test_data["user_data"][client_user_name]['y']
        global_eval_batch_size = 10

        global_correct_prediction = 0
        total_val_loss = 0.0
        global_matched_model.to(device)
        criterion = nn.CrossEntropyLoss()
        hidden_test = global_matched_model.init_hidden(global_eval_batch_size)
        global_matched_model.eval()
        with torch.no_grad():
            for i in range(int(global_num_samples_test / global_eval_batch_size)):
                input_data, target_data = process_x(global_test_data[global_eval_batch_size*i:global_eval_batch_size*(i+1)]), process_y(global_test_label[global_eval_batch_size*i:global_eval_batch_size*(i+1)])
                data, targets = torch.from_numpy(input_data).to(device), torch.from_numpy(target_data).to(device)
                hidden_test = repackage_hidden(hidden_test)
                output, hidden_test = global_matched_model(data, hidden_test)
                loss = criterion(output.t(), torch.max(targets, 1)[1])
                _, pred_label = torch.max(output.t(), 1)
                global_correct_prediction += (pred_label == torch.max(targets, 1)[1]).sum().item()
                total_val_loss += loss.item()
        logger.info('*' * 89)
        logger.info('| Matched model on Global Testset | valid loss {:5.2f} | pred: {}/{} | acc: {:.4f}%'.format(total_val_loss, global_correct_prediction, global_num_samples_test, global_correct_prediction/global_num_samples_test*100.0))
        logger.info('*' * 89)
        #exit()


        frb_global_model_weights = []
        for gmm_index, gmm in enumerate(global_matched_model.parameters()):
            logger.info("Matched model index: {}, matched model shape: {}".format(gmm_index, gmm.size()))
            frb_global_model_weights.append(gmm.detach().cpu().numpy())

        ##########################################
        # stage of reconstruct each local model
        #########################################
        res_local_weight_list = []
        for client_index in range(n_clients):
            res_local_weights = reconstruct_local_weights(frb_global_model_weights, assignments_list, client_index)
            res_local_weight_list.append(res_local_weights)

        ###############################################
        # local retraining process
        ###############################################
        nets_list = local_retraining_process(n_clients=n_clients, train_data=train_data, test_data=test_data, 
                                res_local_weight_list=res_local_weight_list, lr=lr, user_names=TRIAL_USER_NAME)
        collected_weights = collect_weights(nets_list)
        #exit()

        #######################################################
        # start to conduct FedMA process
        #######################################################
        gamma = 1e-3
        sigma = 1.0
        sigma0 = 1.0
        it=5

        matching_shapes = []
        assignments_list = []

        for i in range(NUM_LAYERS-1):
            matched_weights, assignments, next_layer_shape, pop_counts = layerwise_fedma(batch_weights=collected_weights, 
                                                                            layer_index=i, 
                                                                            sigma_layers=sigma, 
                                                                            sigma0_layers=sigma0, 
                                                                            gamma_layers=gamma, 
                                                                            it=it,
                                                                            n_layers=NUM_LAYERS,
                                                                            matching_shapes=matching_shapes)
            matching_shapes.append(next_layer_shape)
            assignments_list.append(assignments)

            if i == 0:
                avg_encoding_weights = np.zeros(matched_weights[0].T.shape) # (80, 8)
                for client_index in range(n_clients):
                    #avg_encoding_weights += 1/2 * patch_weights(collected_weights[client_index][0], next_layer_shape, assignments[client_index])
                    avg_encoding_weights += patch_weights(collected_weights[client_index][0], next_layer_shape, assignments[client_index])
                avg_encoding_weights *= (1/pop_counts)
            elif i == 1:
                avg_h_weights = np.zeros((next_layer_shape*4, next_layer_shape), dtype=np.float32)
                avg_h_bias = np.zeros(next_layer_shape*4, dtype=np.float32)

                avg_i_weights = np.zeros((next_layer_shape*4, matching_shapes[i-1]), dtype=np.float32)
                avg_i_bias = np.zeros(next_layer_shape*4, dtype=np.float32)
                
                for client_index in range(n_clients):
                    h_weights = collected_weights[client_index][i+1]
                    h_bias = collected_weights[client_index][i+3]
                    permutated_h_weights = patch_h_weights(h_weights, next_layer_shape, assignments[client_index])
                    permutated_h_bias = patch_biases(h_bias, next_layer_shape, assignments[client_index])
                    avg_h_weights += (1/n_clients * permutated_h_weights)
                    avg_h_bias += (1/n_clients * permutated_h_bias)

                    i_weights = collected_weights[client_index][i]
                    i_bias = collected_weights[client_index][i+2]
                    permutated_i_weights = perm_i_weights(i_weights, next_layer_shape, assignments[client_index])
                    permutated_i_bias = patch_biases(i_bias, next_layer_shape, assignments[client_index])      
                    #avg_i_weights += (1/n_clients * permutated_i_weights)
                    #avg_i_bias += (1/n_clients * permutated_i_bias)
                    avg_i_weights += permutated_i_weights
                    avg_i_bias += permutated_i_bias
                coef_pop = 1/np.hstack((pop_counts, pop_counts, pop_counts, pop_counts)).astype(np.float32)
                avg_i_weights = (avg_i_weights.T*coef_pop).T
                avg_i_bias *= coef_pop
                #avg_i_weights *= (1/pop_counts)
                #avg_i_bias *= (1/pop_counts)
            # we need to permutate the next layer first:
            retrain_nets_list = []
            for client_index in range(n_clients):
                if i == 0:
                    temp_retrained_weights = []
                    next_layer_weights = collected_weights[client_index][i+1]
                    patched_next_layer_weights = patch_weights(next_layer_weights, next_layer_shape, assignments[client_index])

                    for layer_index, layer_weight in enumerate(collected_weights[client_index]):
                        if layer_index == i:
                            #temp_retrained_weights.append(matched_weights[0].T)
                            temp_retrained_weights.append(avg_encoding_weights)
                        elif layer_index == (i+1):
                            temp_retrained_weights.append(patched_next_layer_weights)
                        else:
                            temp_retrained_weights.append(layer_weight)

                    retrain_model = language_model.RNNModel('LSTM', 80, next_layer_shape, 256, 1, 0.2, tie_weights=False) # we now start from 1 layer
                elif i == 1:
                    # we now get the match results based on the w_{i.} matrices we will need to use this to permutate three dimensions
                    # i) col of w_{h.} matrices ii) row of w_{h.} matrices iii) input shape of the next layer
                    # we process the next layer first:
                    next_layer_weights = collected_weights[client_index][i+4]
                    patched_next_layer_weights = patch_weights(next_layer_weights, next_layer_shape, assignments[client_index])

                    temp_retrained_weights = [collected_weights[client_index][0], avg_i_weights, avg_h_weights, avg_i_bias, avg_h_bias, patched_next_layer_weights, collected_weights[client_index][-1]]
                    retrain_model = language_model.RNNModel('LSTM', 80, matching_shapes[0], next_layer_shape, 1, 0.2, tie_weights=False) # we now start from 1 layer
                
                # we write a retrain function here:
                # retrain step i) load weights
                new_state_dict = {}
                for param_idx, (key_name, param) in enumerate(retrain_model.state_dict().items()):
                    temp_dict = {key_name: torch.from_numpy(temp_retrained_weights[param_idx])}
                    new_state_dict.update(temp_dict)
                retrain_model.load_state_dict(new_state_dict)
                retrain_model.to(device)

                # retrain step ii) retraining
                logger.info("Start the retraining process over client: {}".format(client_index))
                client_user_name = TRIAL_USER_NAME[client_index]
                num_samples_train = len(train_data["user_data"][client_user_name]['x'])
                user_train_data = train_data["user_data"][client_user_name]

                optimizer = optim.SGD(retrain_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
                
                # we only fine tune the last layer in this version : i.e. last weight and last bias
                if i == 0:
                    for param_idx, param in enumerate(retrain_model.parameters()):
                        if param_idx <= i:
                            param.requires_grad = False
                            logger.info("param index: {}, requires grad: {}".format(param_idx, param.requires_grad))
                elif i == 1:
                    freeze_indices = [idx for idx in range(0, 5)]
                    for param_idx, param in enumerate(retrain_model.parameters()):
                        if param_idx in freeze_indices:
                            param.requires_grad = False

                for epoch in range(5):
                    retrain_model.train()
                    epoch_start_time = time.time()

                    hidden_train = retrain_model.init_hidden(BATCH_SIZE)
                    for batch_index in range(int(num_samples_train / BATCH_SIZE)):
                        input_data, target_data = process_x(user_train_data['x'][BATCH_SIZE*batch_index:BATCH_SIZE*(batch_index+1)]), process_y(user_train_data['y'][BATCH_SIZE*batch_index:BATCH_SIZE*(batch_index+1)])
          
                        data, targets = torch.from_numpy(input_data).to(device), torch.from_numpy(target_data).to(device)
                        optimizer.zero_grad()

                        hidden_train = repackage_hidden(hidden_train)
                        output, hidden_train = retrain_model(data, hidden_train)

                        loss = criterion(output.t(), torch.max(targets, 1)[1])
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                        cur_loss = total_loss
                        total_loss = 0
                        start_time = time.time()

                retrain_nets_list.append(retrain_model)
            collected_weights = collect_weights(retrain_nets_list)
        
        avg_last_weight = np.zeros(collected_weights[0][-2].shape, dtype=np.float32)
        avg_last_bias = np.zeros(collected_weights[0][-1].shape, dtype=np.float32)

        for client_index in range(n_clients):
            avg_last_weight += (1/n_clients * collected_weights[client_index][-2])
            avg_last_bias += (1/n_clients * collected_weights[client_index][-1])
        global_matched_weights = [collected_weights[0][l_idx] for l_idx in range(len(collected_weights[0])-2)] + [avg_last_weight] + [avg_last_bias]

        global_matched_model = language_model.RNNModel('LSTM', 80, matching_shapes[0], matching_shapes[1], 1, 0.2, tie_weights=False)
        new_state_dict = {}
        for param_idx, (key_name, param) in enumerate(global_matched_model.state_dict().items()):
            temp_dict = {key_name: torch.from_numpy(global_matched_weights[param_idx])}
            new_state_dict.update(temp_dict)

        global_matched_model.load_state_dict(new_state_dict)
        global_matched_model.to(device)
        global_matched_model.eval()
        hidden_test = global_matched_model.init_hidden(global_eval_batch_size)
        global_correct_prediction = 0.0
        with torch.no_grad():
            for i in range(int(global_num_samples_test / global_eval_batch_size)):
                input_data, target_data = process_x(global_test_data[global_eval_batch_size*i:global_eval_batch_size*(i+1)]), process_y(global_test_label[global_eval_batch_size*i:global_eval_batch_size*(i+1)])
                data, targets = torch.from_numpy(input_data).to(device), torch.from_numpy(target_data).to(device)
                hidden_test = repackage_hidden(hidden_test)
                output, hidden_test = global_matched_model(data, hidden_test)
                loss = criterion(output.t(), torch.max(targets, 1)[1])
                _, pred_label = torch.max(output.t(), 1)
                global_correct_prediction += (pred_label == torch.max(targets, 1)[1]).sum().item()
                total_val_loss += loss.item()
        logger.info('*' * 89)
        logger.info('| Matched model on Global Testset | valid loss {:5.2f} | pred: {}/{} | acc: {:.4f}%'.format(total_val_loss, global_correct_prediction, global_num_samples_test, global_correct_prediction/global_num_samples_test*100.0))
        logger.info('*' * 89)

        with open("lstm_matching_assignments", "wb") as assignment_file:
            pickle.dump(assignments_list, assignment_file)
        with open("lstm_matching_shapes", "wb") as ms_file:
            pickle.dump(matching_shapes, ms_file)
        with open("matched_global_weights", "wb") as matched_weight_file:
            pickle.dump(global_matched_model, matched_weight_file)