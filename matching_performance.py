import logging
from model import *
from utils import *

from vgg import *
from vgg import matched_vgg11

def compute_model_averaging_accuracy(models, weights, train_dl, test_dl, n_classes, args):
    """An variant of fedaveraging"""
    if args.model == "lenet":
        avg_cnn = LeNet()
    elif args.model == "vgg":
        avg_cnn = vgg11()
    elif args.model == "simple-cnn":
        if args.dataset in ("cifar10", "cinic10"):
            avg_cnn = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
        elif args.dataset == "mnist":
            avg_cnn = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
    elif args.model == "moderate-cnn":
        if args.dataset in ("cifar10", "cinic10"):
            avg_cnn = ModerateCNN()
        elif args.dataset == "mnist":
            avg_cnn = ModerateCNNMNIST()
    
    new_state_dict = {}
    model_counter = 0

    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(avg_cnn.state_dict().items()):
        if "conv" in key_name or "features" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}

        new_state_dict.update(temp_dict)

    avg_cnn.load_state_dict(new_state_dict)

    # switch to eval mode:
    avg_cnn.eval()
    ##    
    
    correct, total = 0, 0
    for batch_idx, (x, target) in enumerate(test_dl):
        out_k = avg_cnn(x)
        _, pred_label = torch.max(out_k, 1)
        total += x.data.size()[0]
        correct += (pred_label == target.data).sum().item()
        
    logger.info("Accuracy for Fed Averaging correct: {}, total: {}".format(correct, total))


def compute_pdm_cnn_accuracy(models, weights, train_dl, test_dl, n_classes, assignments):
    """Note that we only handle the FC weights for now"""
    # we need to figure out the FC dims first
    matched_weights = weights[1:] # get rid of the dummy layer, this should be deprecated later
    input_dim = matched_weights[0].shape[0] # hard coded for now, will make changes later
    hidden_dims = [matched_weights[0].shape[1], matched_weights[2].shape[1]]
    output_dim = matched_weights[-1].shape[0]

    logger.info("Input dim: {}, hidden_dims: {}, output_dim: {}".format(input_dim, hidden_dims, output_dim))
    args_n_nets = len(models)
    #book_keeper = {4:0, 5:1, 6:2, 7:3, 8:4, 9:5}

    unmatched_cnn_blocks = []
    for model_i, model in enumerate(models):
        tempt_cnn = ConvBlock()
        
        #logger.info("Keys of layers of convblock ...")
        #prilogger.infont(tempt_cnn.state_dict().keys())
        new_state_dict = {}
        model_counter = 0
        # handle the conv layers part which is not changing
        for param_idx, (key_name, param) in enumerate(tempt_cnn.state_dict().items()):
            if "conv" in key_name:
                temp_dict = {key_name: models[model_i].state_dict()[key_name]}
            new_state_dict.update(temp_dict)
            model_counter += 1

        tempt_cnn.load_state_dict(new_state_dict)
        unmatched_cnn_blocks.append(tempt_cnn)

    matched_state_dict = {}
    matched_fcs = FCBlock(input_dim, hidden_dims, output_dim)
    for param_idx, (key_name, param) in enumerate(matched_fcs.state_dict().items()):
        if "weight" in key_name:
            temp_dict = {key_name: torch.from_numpy(matched_weights[param_idx].T)}
        elif "bias" in key_name:
            temp_dict = {key_name: torch.from_numpy(matched_weights[param_idx])}
        matched_state_dict.update(temp_dict)
    matched_fcs.load_state_dict(matched_state_dict)

    # switch to eval mode:
    for model in unmatched_cnn_blocks:
        model.eval()
    matched_fcs.eval()
    ##    
    
    correct, total = 0, 0
    for batch_idx, (x, target) in enumerate(test_dl):
        #combined_outputs = []
        outputs_aggregator = np.zeros((x.size()[0], weights[0].shape[0]), dtype=np.float32)
        for model_idx in range(args_n_nets):
            # at here, we need to do
            # i) aligning the outputs according to the assignments of the input layer
            # ii) avaraging the aligned outputs
            out = unmatched_cnn_blocks[model_idx](x)
            out_numpy = out.detach().numpy()
            padded_out = np.zeros((out.size()[0], weights[0].shape[0]), dtype=np.float32)
            padded_out[:, assignments[2][model_idx]] = out_numpy
            outputs_aggregator += padded_out
            #combined_outputs.append(padded_out)

        outputs_aggregator /= args_n_nets # averaging step
        
        combined_conv_block_out = torch.from_numpy(outputs_aggregator)
        out_k = matched_fcs(combined_conv_block_out)
        
        _, pred_label = torch.max(out_k, 1)
        total += x.data.size()[0]
        correct += (pred_label == target.data).sum().item()
        
    logger.info("Accuracy for Neural Matching correct: {}, total: {}".format(correct, total))

def compute_pdm_vgg_accuracy(models, weights, train_dl, test_dl, n_classes, assignments):
    """Note that we only handle the FC weights for now"""
    # we need to figure out the FC dims first
    matched_weights = weights[1:] # get rid of the dummy layer, this should be deprecated later
    input_dim = matched_weights[0].shape[0] # hard coded for now, will make changes later
    hidden_dims = [matched_weights[0].shape[1], matched_weights[2].shape[1]]
    output_dim = matched_weights[-1].shape[0]

    logger.info("Input dim: {}, hidden_dims: {}, output_dim: {}".format(input_dim, hidden_dims, output_dim))
    args_n_nets = len(models)

    unmatched_cnn_blocks = []
    for model_i, model in enumerate(models):
        tempt_cnn = VGGConvBlocks(make_layers(cfg['A'], batch_norm=True), num_classes=10)
        
        new_state_dict = {}
        model_counter = 0
        # handle the conv layers part which is not changing
        for param_idx, (key_name, param) in enumerate(tempt_cnn.state_dict().items()):
            if "classifier" not in key_name:
                temp_dict = {key_name: models[model_i].state_dict()[key_name]}
            new_state_dict.update(temp_dict)
            model_counter += 1

        tempt_cnn.load_state_dict(new_state_dict)
        unmatched_cnn_blocks.append(tempt_cnn)

    matched_state_dict = {}
    matched_fcs = FCBlockVGG(input_dim, hidden_dims, output_dim)
    for param_idx, (key_name, param) in enumerate(matched_fcs.state_dict().items()):
        if "weight" in key_name:
            temp_dict = {key_name: torch.from_numpy(matched_weights[param_idx].T)}
        elif "bias" in key_name:
            temp_dict = {key_name: torch.from_numpy(matched_weights[param_idx])}
        matched_state_dict.update(temp_dict)
    matched_fcs.load_state_dict(matched_state_dict)

    # switch to eval mode:
    for model in unmatched_cnn_blocks:
        model.eval()
    matched_fcs.eval()
    ##    
    
    correct, total = 0, 0
    for batch_idx, (x, target) in enumerate(test_dl):
        #combined_outputs = []
        outputs_aggregator = np.zeros((x.size()[0], weights[0].shape[0]), dtype=np.float32)
        for model_idx in range(args_n_nets):
            # at here, we need to do
            # i) aligning the outputs according to the assignments of the input layer
            # ii) avaraging the aligned outputs
            out = unmatched_cnn_blocks[model_idx](x)
            out_numpy = out.detach().numpy()
            padded_out = np.zeros((out.size()[0], weights[0].shape[0]), dtype=np.float32)
            padded_out[:, assignments[2][model_idx]] = out_numpy
            outputs_aggregator += padded_out
            #combined_outputs.append(padded_out)
        #print(combined_outputs)
        outputs_aggregator /= args_n_nets # averaging step
        
        combined_conv_block_out = torch.from_numpy(outputs_aggregator)
        out_k = matched_fcs(combined_conv_block_out)
        
        _, pred_label = torch.max(out_k, 1)
        total += x.data.size()[0]
        correct += (pred_label == target.data).sum().item()
        
    logger.info("Accuracy for Neural Matching correct: {}, total: {}".format(correct, total))


def compute_full_cnn_accuracy(models, weights, train_dl, test_dl, n_classes, device, args):
    """Note that we only handle the FC weights for now"""
    # we need to figure out the FC dims first

    #LeNetContainer
    # def __init__(self, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10)

    # this should be safe to be hard-coded since most of the modern image classification dataset are in RGB format
    #args_n_nets = len(models)

    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]

        output_dim = weights[-1].shape[0]

        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset == "mnist":
            input_channel = 1
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=10)
    elif args.model == "moderate-cnn":
        #[(35, 27), (35,), (68, 315), (68,), (132, 612), (132,), (132, 1188), (132,), 
        #(260, 1188), (260,), (260, 2340), (260,), 
        #(4160, 1025), (1025,), (1025, 515), (515,), (515, 10), (10,)]
        num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
        input_dim = weights[12].shape[0]
        hidden_dims = [weights[12].shape[1], weights[14].shape[1]]
        if args.dataset in ("cifar10", "cinic10"):
            matched_cnn = ModerateCNNContainer(3,
                                                num_filters, 
                                                kernel_size=3, 
                                                input_dim=input_dim, 
                                                hidden_dims=hidden_dims, 
                                                output_dim=10)
        elif args.dataset == "mnist":
            matched_cnn = ModerateCNNContainer(1,
                                                num_filters, 
                                                kernel_size=3, 
                                                input_dim=input_dim, 
                                                hidden_dims=hidden_dims, 
                                                output_dim=10)

    #logger.info("Keys of layers of convblock ...")
    new_state_dict = {}
    model_counter = 0
    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        #print("&"*30)
        #print("Key: {}, Weight Shape: {}, Matched weight shape: {}".format(key_name, param.size(), weights[param_idx].shape))
        #print("&"*30)
        if "conv" in key_name or "features" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}

        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    matched_cnn.to(device)
    matched_cnn.eval()

    ##    
    
    correct, total = 0, 0
    for batch_idx, (x, target) in enumerate(test_dl):
        x, target = x.to(device), target.to(device)
        out_k = matched_cnn(x)
        _, pred_label = torch.max(out_k, 1)
        total += x.data.size()[0]
        correct += (pred_label == target.data).sum().item()
        
    logger.info("Accuracy for Neural Matching correct: {}, total: {}".format(correct, total))