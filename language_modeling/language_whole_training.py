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

import language_model


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


BATCH_SIZE = 50
TRAIN_DATA_DIR = "./datum/leaf/data/shakespeare/data/train/"
TEST_DATA_DIR = "./datum/leaf/data/shakespeare/data/test/"

TRAIN_DATA_NAME = "all_data_niid_0_keep_10000_train_9.json"
TEST_DATA_NAME = "all_data_niid_0_keep_10000_test_9.json"


TRIAL_EPOCH=30

# since we used a relatively "fixed" model for shakespeare dataset
# we thus hardcode it here
#NUM_LAYERS=3 # we start from 1-layer LSTM now (so the 3 layers now is encoder|hidden LSTM|decoder)

NUM_LAYERS=4 # 2-layer LSTM (4 layers: encoder|hidden LSTM1|hidden LSTM2|decoder)


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
    retrain_flag = True


    TRIAL_USER_NAME = train_data["users"][0:n_clients] #this can be of length in range from 1 to 132
    start_time = time.time()
    total_loss = 0.0

    num_samples_list = [len(train_data["user_data"][TRIAL_USER_NAME[client_index]]['x']) for client_index in range(n_clients)]
    nets_list = []
    criterion = nn.CrossEntropyLoss()

    # we will need to construct a new global test set based on all test data on each of the clients
    global_test_data = []
    global_test_label = []
    global_num_samples_test = 0

    global_train_data = []
    global_train_label = []
    global_num_samples_train = 0
    for client_index in range(n_clients):
        client_user_name = TRIAL_USER_NAME[client_index]

        global_num_samples_test += len(test_data["user_data"][client_user_name]['x'])
        global_test_data += test_data["user_data"][client_user_name]['x']
        global_test_label += test_data["user_data"][client_user_name]['y']

        global_num_samples_train += len(train_data["user_data"][client_user_name]['x'])
        global_train_data += train_data["user_data"][client_user_name]['x']
        global_train_label += train_data["user_data"][client_user_name]['y']

    global_eval_batch_size = 10

    logger.info("Start training over the entire dataset ....")

    model = language_model.RNNModel('LSTM', 80, 8, 256, 1, 0.2, tie_weights=False).to(device)
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001, amsgrad=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    for epoch in range(TRIAL_EPOCH):
        model.train()
        epoch_start_time = time.time()

        hidden_train = model.init_hidden(BATCH_SIZE)
        for i in range(int(global_num_samples_train / BATCH_SIZE)):
            input_data, target_data = process_x(global_train_data[BATCH_SIZE*i:BATCH_SIZE*(i+1)]), process_y(global_train_label[BATCH_SIZE*i:BATCH_SIZE*(i+1)])

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

            total_loss = 0
            start_time = time.time()

        global_eval_batch_size = 10
        model.eval()
        total_val_loss = 0.
        ntokens = 80
        hidden_test = model.init_hidden(global_eval_batch_size)
        correct_prediction = 0
        with torch.no_grad():
            for i in range(int(global_num_samples_test / global_eval_batch_size)):
                input_data, target_data = process_x(global_test_data[global_eval_batch_size*i:global_eval_batch_size*(i+1)]), process_y(global_test_label[global_eval_batch_size*i:global_eval_batch_size*(i+1)])
                data, targets = torch.from_numpy(input_data).to(device), torch.from_numpy(target_data).to(device)

                hidden_test = repackage_hidden(hidden_test)
                output, hidden_test = model(data, hidden_test)
                loss = criterion(output.t(), torch.max(targets, 1)[1])
                _, pred_label = torch.max(output.t(), 1)
                correct_prediction += (pred_label == torch.max(targets, 1)[1]).sum().item()

                total_val_loss += loss.item()
        logger.info('-' * 89)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | pred: {}/{} | acc: {:.4f}%'.format(epoch, (time.time() - epoch_start_time),
                                               total_val_loss, correct_prediction, global_num_samples_test, correct_prediction/global_num_samples_test*100.0))
        logger.info('-' * 89)