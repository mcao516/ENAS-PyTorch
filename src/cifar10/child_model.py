# -*- coding: utf-8 -*-

"""This python file contains ChildModel class which is for microChild training.
   
   Author: Meng Cao
"""

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils import Progbar
from src.config import Config
from src.cifar10.micro_child import MicroChild
from src.data_utils import read_cifar10, DatasetBuilder, get_mean_and_std


class CELossWithAuxHead(nn.Module):
    """Module for computing cross entropy loss with possible auxiliary loss."""
    
    def __init__(self, aux_weight=0.4):
        super(CELossWithAuxHead, self).__init__()
        self.aux_weight = aux_weight
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, logits, target, aux_logits=None):
        """
        Args:
            logits: [batch_size, class_um]
            target: [batch_size]
            aux_logits: [batch_size, class_um]
        """
        loss = self.criterion(logits, target)
        
        if aux_logits is not None:
            aux_loss = self.criterion(aux_logits, target)
            loss = loss + aux_loss * self.aux_weight
        
        return loss


class ChildModel:
    """The class for child model training, validating and testing."""
    
    def __init__(self, config, device, write_summary=True):
        """Initialize model.
        """
        self.config = config
        self.logger = self.config.logger
        self.global_step = 0
        
        # find device
        self.device = device

        # build and initialize model
        self.logger.info("- Building and initializing cihld model...")
        self.model = self._build_model(config)
        self._initialize_model(self.model)
        
        # multi-gpu training
        if torch.cuda.device_count() > 1 and config.multi_gpu:
            self.logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=config.device_ids)
        self.model.to(device)
        
        # create optimizer and criterion
        self.logger.info("- Creating optimizer and criterion for child model...")
        self.optimizer = self._get_optimizer(config, self.model)
        self.scheduler = self._get_scheduler(self.optimizer, config.T_max, config.lr_min)
        self.criterion = self._get_criterion(config).to(device)

        # create summary for tensorboard visualization
        if write_summary:
            self.writer = SummaryWriter(self.config.path_summary)
        else:
            self.writer = None        
        
        self.arcs = None # architecture for normal and reduction cell
        
        
    def _build_model(self, config):
        """Build a child model.
        """
        return MicroChild(config)
    
        
    def _initialize_model(self, model):
        """Model initialization.
        """
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    
    def _get_optimizer(self, config, model):
        """Get optimizer.
        """
        if config.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), 
                                         lr=config.lr, betas=(0.9, 0.98), 
                                         eps=1e-9, weight_decay=config.l2_reg)
        elif config.optimizer == 'momentum':
            optimizer = torch.optim.SGD(model.parameters(), 
                                        lr=config.lr, momentum=0.9, 
                                        weight_decay=config.l2_reg, 
                                        nesterov=True)
        else:
            raise ValueError("Unknown optimizer type: {}".format(config.optimizer))

        return optimizer

    def _get_scheduler(self, optimizer, T_max, lr_min=0, last_epoch=-1):
        """Learning rate scheduler for Cosine Annealing.
        """
        return CosineAnnealingLR(optimizer, T_max, 
                                 eta_min=lr_min, 
                                 last_epoch=last_epoch)
    
    def _get_criterion(self, config):
        """No need explaintion. 
        """
        return CELossWithAuxHead()
    
    def _get_step_ratio(self, num_train_batches):
        train_steps = self.config.num_epochs * num_train_batches
        step_ratio = min(1.0, (self.global_step + 1) / train_steps)
        assert self.global_step / train_steps <= 1.0 + 1e-6, "Global step: {}; Train step: {}".format(self.global_step, train_steps)
        return step_ratio
    
    def load_weights(self, path):
        """Load pre-trained weights.
        """
        self.model.load_state_dict(torch.load(path))

    def save_model(self):
        """Save model's weights.
        """
        torch.save(self.model.state_dict(), self.config.dir_model + "child_model.pickle")
        self.logger.info("- child model is saved at: {}".format(self.config.dir_model))
        
    def set_arcs(self, arcs):
        """Set architectures. Must do this before calling the forward method.
        
        Args:
            arcs: tuple of two lists that contain integers represents the 
                architecture of a normal cell and a reduce cell.
        """
        if not isinstance(arcs, tuple):
            raise ValueError("Arcs must be a tuple of normal and reduciton cell architecture.")
        self.arcs = arcs
        
    def loss_batch(self, loss_func, logits, target, aux_logits=None, norm=None, optimizer=None):
        """Compute loss and update model weights on a batch of data.

        Args:
            logits: [batch_size, class_num]
            target: [batch_size]
            aux_logits: [batch_size, class_num]

        Return:
            loss: float, average loss value
        """
        loss = loss_func(logits, target, aux_logits)
        if norm is not None:
            loss /= norm
        
        if optimizer is not None:
            with torch.set_grad_enabled(True):
                loss.backward()     # compute gradients
                optimizer.step()    # update weights
                optimizer.zero_grad()
                
        return loss.item()
    
    def train_epoch(self, dataset, epoch):
        """Train the model for one single epoch.
        """
        if self.arcs is None:
            raise Exception("Did you forget to set model arcs?")

        self.model.train() # set the model to train mode
        prog = Progbar(target=len(dataset)) # progress bar for visualization
        
        train_loss = 0.
        for i, (images, labels) in enumerate(dataset):
            # logits, aux_logits = model(images, self.arcs, self._get_step_ratio(len(dataset)))
            logits, aux_logits = self.model(images, self.arcs)
            
            if aux_logits is not None:
                assert logits.shape == aux_logits.shape, "Logits and auxiliary logits should have same shape."            

            # compute loss and update model parameters on a batch of data
            batch_loss = self.loss_batch(self.criterion, logits, labels, 
                                         aux_logits=aux_logits, 
                                         optimizer=self.optimizer)
            prog.update(i + 1, [("batch loss", batch_loss)])
            train_loss += batch_loss

            self.global_step += 1

            if self.writer is not None: # write summary to tensorboard
                self.writer.add_scalar('batch_loss', batch_loss, epoch*len(dataset) + i + 1)
            
                # draw the diagram for the first batch of data
                # if i == 0 and epoch == 0:
                #     self.writer.add_graph(self.model, (en_input, en_mask, de_input, de_mask), verbose=False)

        self.scheduler.step() # update learning rate
        epoch_loss = train_loss / len(dataset) # average loss

        return epoch_loss
    
    def evaluate(self, dataset):
        """Evaluate the model, return average loss and accuracy.
        """
        if self.arcs is None:
            raise Exception("Did you forget to set model arcs?")

        self.model.eval()
        with torch.no_grad():
            sample_num, eval_loss, eval_corrects = 0, 0., 0.
            for i, sample_batched in enumerate(dataset):
                images, labels = sample_batched
                logits, _ = self.model(images, self.arcs) # logits: [N, class_num]
                
                # compute loss and update model parameters on a batch of data
                batch_loss = self.loss_batch(self.criterion, logits, labels, optimizer=None)
                eval_loss += batch_loss
                
                pred_labels = torch.argmax(logits, dim=-1)
                assert pred_labels.shape == labels.shape, "Predition output shape {} and actual labels shape {} does Not match.".format(pred_labels.shape, labels.shape)
                eval_corrects += torch.sum(pred_labels == labels).item()
                sample_num += logits.shape[0]
                
            avg_loss = eval_loss / len(dataset) # divide by batch number
            avg_acc  = eval_corrects / sample_num

        return avg_loss, avg_acc
    
    def fit(self, train_set, development_set, saving=False):
        """Model training.
        """
        num_epochs = self.config.num_epochs
        best_acc = 0.

        for epoch in range(num_epochs):
            self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            # print('-' * 10)
            # train
            self.logger.info("Learning rate: [{:.5}]".format(self.scheduler.get_lr()[0]))
            train_loss = self.train_epoch(train_set, epoch)
            self.logger.info("Traing Loss: {}".format(train_loss))

            # eval
            eval_loss, eval_acc = self.evaluate(development_set)
            self.logger.info("Evaluation:")
            self.logger.info("- loss: {}".format(eval_loss))
            self.logger.info("- acc: {}".format(eval_acc))
            
            # monitor loss and accuracy
            if self.writer is not None:
                self.writer.add_scalar('epoch_loss', train_loss, epoch)
                self.writer.add_scalar('eval_loss', eval_loss, epoch)
                self.writer.add_scalar('eval_acc', eval_acc, epoch)
                self.writer.add_scalar('lr', self.scheduler.get_lr()[0], epoch)

            # save the model
            if eval_acc >= best_acc:
                self.logger.info("New best score!")
                best_acc = eval_acc
                if saving:
                    self.save_model()

        return eval_loss, eval_acc
    
    def predict(self, inputs):
        """
        Args:
            inputs: input images with shape [N, C, H, W]
        
        Return:
            outputs: logits [N, class_num]
            pred_labels: [N]
        """
        self.model.eval()
        with torch.no_grad():
            if self.arcs is None:
                raise Exception("Did you forget to set model arcs?")
            outputs = self.model(inputs, self.arcs)    # outputs: [N, class_num]
            pred_labels = torch.argmax(outputs, dim=-1) # [N]
            
        return outputs, pred_labels
    
    def test(self, dataset):
        """Test the model's accuracy and print out a report.
        """
        self.model.eval() 
        with torch.no_grad():
            total_samples, corrects = 0, 0
            pred_class, label_class = [], []
            for images, labels in dataset:
                _, pred_labels = self.predict(images)

                corrects += torch.sum(labels == pred_labels).double()
                total_samples += labels.shape[0]
                
                for p, l in zip(pred_labels, labels):
                    pred_class.append(p)
                    label_class.append(l)
 
            accuracy = corrects / total_samples
            self.logger.info('\n')
            self.logger.info('Accuracy: {:.3}\n\n'.format(accuracy))
            self.logger.info(classification_report(label_class, pred_class))
        
        return label_class, pred_class

