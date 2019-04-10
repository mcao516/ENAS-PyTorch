# -*- coding: utf-8 -*-

"""Configuration class.

   Author: Meng Cao
"""

import os

from datetime import datetime
from .utils import get_logger


class Config(): 

    def __init__(self, operation=""):
        """Initialize hyperparameters and load vocabs.
        """
        self.dir_output = "results/{}/{:%Y%m%d_%H%M%S}/".format(operation, 
            datetime.now())
        self.dir_model  = self.dir_output + "model/"
        self.path_log   = self.dir_output + "log.txt"
        self.path_params  = self.dir_output + "params.txt"
        self.path_summary = self.dir_output + "summary"
        
        # directory for training outputs
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)

        # create instance of logger
        self.logger = get_logger(self.path_log)
    

    multi_gpu = True
    device_ids = [0, 1]
    data_dir = 'data/cifar10/'  # training data directorys
    
    # micro child hyper parameters
    lr = 5e-3               # for learning rate cosine annealing 
    lr_min = 1e-4           # for learning rate cosine annealing 
    T_max = 10              # for learning rate cosine annealing 
    fixed = False           # if use fixed arc
    l2_reg = 2e-4           # l2 regularization
    node_num = 5            # node in each ENAS layer
    class_num = 10          # classification number
    image_size = 32         # input image size
    batch_size = 144 * 2    # batch size
    num_epochs = 300        # training epoch number
    out_channels = 36       # out channels
    use_aux_heads = True    # if auxiliary head is used
    num_layers = 15         # total number of layers
    keep_prob = 0.6             # dropout at the last projection layer
    drop_path_keep_prob = 0.6   # droput out at each internal ENAS layer
    optimizer = 'adam'      # optimizer type
    
    # controller
    bl_dec = 0.99
    contro_lr = 3.5e-3
    entropy_weight = 1e-4
    contro_train_epochs = 30
    contro_num_epochs = 5
    lstm_size = 32
    num_branches = 5
    temperature = None
    tanh_constant = 1.10
    op_tanh_constant = tanh_constant / 2.5
    num_lstm_layer = 2
    use_bias = False
    train_contro_every = 2

    # def write_params_to_log(self, file_path=None):
    #     """Save model parameters to a file.
    #     """
    #     if not file_path:
    #         file_path = self.path_params

    #     with open(file_path, 'w', encoding='utf-8') as file_object:
    #         file_object.write("num_classes: {}\n".format(self.num_classes))
    #         file_object.write("image_size: {}\n".format(self.image_size))
    #         file_object.write("batch_size: {}\n".format(self.batch_size))
    #         file_object.write("feature_lr: {}\n".format(self.feature_lr))
    #         file_object.write("fc_lr: {}\n".format(self.fc_lr))
    #         file_object.write("num_epochs: {}\n".format(self.num_epochs))

    #         file_object.write("model_name: {}\n".format(self.model_name))
    #         file_object.write("optimizer: {}\n".format(self.optimizer))
    #         file_object.write("momentum: {}\n".format(self.momentum))
    #         file_object.write("feature_extract: {}\n".format(self.feature_extract))
    #         file_object.write("use_pretrained: {}".format(self.use_pretrained))
            
