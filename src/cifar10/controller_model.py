# -*- coding: utf-8 -*-


"""This python file contains class for ENAS controller training.

   Author: Meng Cao
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


from src.utils import Progbar
from tensorboardX import SummaryWriter
from src.cifar10.micro_controller import MicroController


class ReinforceLoss(nn.Module):
    """Compute reinforce loss."""
    
    def __init__(self, entropy_weight=None, bl_dec=0.999):
        super(ReinforceLoss, self).__init__()
        self.entropy_weight = entropy_weight
        self.bl_dec = bl_dec
        
        # baseline = torch.tensor(0)
        # self.register_buffer('baseline', baseline)
        self.baseline = 0.
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, reward, logits, target):
        """
        Args:
            reward: float, valid accuracy of child model.
            logits: list whose length equals to node number [node_num, 4]. Each element 
                in the list is a tensor with size: [1, num_of_prev_nodes or num_of_branches].
            target: tensor, list of integers, [index_1, index_2, op_1, op_2] * (node number)
        """
        log_prob, sample_entropy = self._get_log_prob_and_entropy(logits, target)
        if self.entropy_weight is not None:
            reward += self.entropy_weight * sample_entropy
        
        self.baseline -= (1.0 - self.bl_dec) * (self.baseline - reward)
        loss = log_prob * (reward - self.baseline)        

        return loss
        
        
    def _get_log_prob_and_entropy(self, logits, target):
        """Iterate over all nodes and operations.
        """
        node_num, sample_num = len(logits), len(logits[0])
        assert sample_num == 4, "Each node should have 4 samples."
        
        log_prob, entropy = 0., 0.
        for i in range(node_num):
            for j in range(sample_num):
                sampled_id, logit = target[i * 4 + j], logits[i][j]
                sampled_id = torch.unsqueeze(sampled_id, dim=0)
                
                log_prob += self._get_cross_entropy_loss(logit, sampled_id)
                entropy += self._get_entropy(logit)
        
        log_prob = log_prob / (node_num * sample_num)
        entropy = entropy / (node_num * sample_num)
        
        return log_prob, entropy
                
        
    def _get_cross_entropy_loss(self, logits, index):
        """
        Args:
            logits: [1, num_of_prev_nodes or num_of_branches]
            index: [1]
        """
        return self.criterion(logits, index)
    
    
    def _get_entropy(self, logits):
        """Compute entropy using sampled label as ground truth.
        
        Args:
            logits: [1, num_of_prev_nodes or num_of_branches]
        """
        logits = torch.squeeze(logits, dim=0)
        entropy = self._softmax_cross_entropy_with_logits(logits, 
                      F.softmax(logits, dim=-1))
        return entropy.item()
    
    
    def _softmax_cross_entropy_with_logits(self, logits, labels):
        """Implement tensorflow "tf.nn.softmax_cross_entropy_with_logits"

        Args:
            logits: [batch_size, num_classes]
            labels: [batch_size, num_classes]
        """
        assert logits.shape == labels.shape, "Logits and labels should have same shape!"
        loss = torch.sum(-labels * F.log_softmax(logits, dim=-1), -1)
        return loss.mean()



class ControllerModel:
    """Implement controller model for controller training, validating and testing."""
    
    def __init__(self, config, device, write_summary=True):
        """Initialize the model.
        """
        self.config = config
        self.device = device
        self.logger = config.logger

        # build and initialize model
        self.logger.info("- Building and initializing controller model...")
        self.model = self._build_model(config, self.device)
        self._initialize_model()
        
        # multi-gpu training
        # if torch.cuda.device_count() > 1 and config.multi_gpu:
        #     self.logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
        #     self.model = nn.DataParallel(self.model, device_ids=config.device_ids)
        self.model.to(device)

        # create optimizer and criterion
        self.logger.info("- Creating optimizer and criterion for controller model...")
        self.optimizer = self._get_optimizer(config, self.model)
        self.criterion = self._get_criterion(config)

        # create summary for tensorboard visualization
        if write_summary:
            self.writer = SummaryWriter(self.config.path_summary)
        else:
            self.writer = None
        
        
    def _build_model(self, config, device):
        """Build controller model.
        """
        return MicroController(config, device)
        
    def _initialize_model(self):
        """Model initialization.
        """
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, a=-0.1, b=0.1)
        return self.model
    
    def _get_optimizer(self, config, model):
        """Create Optimizer for training.
        """
        return torch.optim.Adam(model.parameters(), lr=config.contro_lr, 
                                betas=(0.9, 0.98), eps=1e-9)

    def _get_criterion(self, config):
        """Loss function.
        """
        return ReinforceLoss(entropy_weight=config.entropy_weight, 
                             bl_dec=config.bl_dec)

    def load_weights(self, path):
        """Load pre-trained weights.
        """
        self.model.load_state_dict(torch.load(path))
        
    def save_model(self):
        """Save model's weights.
        """
        torch.save(self.model.state_dict(), self.config.dir_model + "contro_model.pickle")
        self.logger.info("- controller is saved at: {}".format(self.config.dir_model))

    def sample(self, training=True):
        """Sample the arcitecture of normal cell and reduction cell.
        """
        if training: self.model.train()
        else: self.model.eval()

        (prev_c, prev_h), arc_1, logits_1 = self.model(use_bias=True)
        _, arc_2, logits_2 = self.model(prev_c, prev_h, use_bias=False)

        return (arc_1, arc_2), (logits_1, logits_2)

    
    def train_epoch(self, acc, arc_and_logits, retrain=False):
        """Train the model for one single epoch.

           acc: float, CNN model classification accuracy.
           arc_and_logits: [(arc_1, logits_1), (arc_2, logits_2)]
        """
        self.model.train()
        loss = 0.

        for i, (target, logits) in enumerate(arc_and_logits):
            # compute loss and update model parameters on a batch of data
            batch_loss = self.criterion(acc, logits, target)
            loss += batch_loss

        if self.optimizer is not None:
            with torch.set_grad_enabled(True):
                loss.backward(retain_graph=retrain) # compute gradients
                self.optimizer.step()               # update weights
                self.optimizer.zero_grad()

        # compute the average loss
        return loss.item() / len(arc_and_logits)
    

    def fit(self, acc, dataset):
        """Model training.

           acc: float, CNN model classification accuracy.
           dataset: [(arc_1, logits_1), (arc_2, logits_2)]
        """
        num_epochs = self.config.contro_num_epochs
        # prog = Progbar(target=num_epochs) # progress bar for visualization

        # self.logger.info("Controller info:")
        # self.logger.info("- acc: {}".format(acc))
        # self.logger.info("- baseline: {}".format(self.criterion.baseline))
        
        avg_loss = 0.
        for epoch in range(num_epochs):
            # train
            # self.scheduler.step()
            # print("Learning rate: [{:.5}]".format(self.scheduler.get_lr()[0]))

            loss = self.train_epoch(acc, dataset, (epoch + 1) != num_epochs)
            # prog.update(epoch + 1, [("batch loss", loss)])
            
            # monitor loss and accuracy
            # if self.writer is not None:
            #     self.writer.add_scalar('controller_loss', loss)
                # self.writer.add_scalar('lr', self.scheduler.get_lr()[0], epoch)

            avg_loss += loss
        
        return avg_loss / num_epochs

    def log_info(self):
        """Log controller information.
        """ 
        self.logger.info("- baseline: {}".format(self.criterion.baseline))