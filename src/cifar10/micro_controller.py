# -*- coding: utf-8 -*-

"""This python file contains PyTorch implementation of the ENAS Micro Controller
   described in paper: Efficient Neural Architecture Search via Parameter Sharing.
   The code is heavily based on the paper's source code, which uses tensorflow 
   library.

   Paper address: https://arxiv.org/abs/1802.03268

   Tensorflow implementation address: 
   https://github.com/melodyguan/enas/blob/master/src/cifar10/micro_controller.py
   

   Author: Meng Cao
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroController(nn.Module):
    """Implement the LSTM micro controller for convolutional cell archecture generation.
    """
    def __init__(self, config, device):
        """Constructs MicroController.
        """
        super(MicroController, self).__init__()
        self.config = config
        self.device = device

        # LSTM, index embedding and operation embeddings
        self.lstm = nn.LSTM(config.lstm_size, config.lstm_size, config.num_lstm_layer)
        self.g_emb = nn.Parameter(torch.randn(1, 1, config.lstm_size, requires_grad=True))
        self.w_emb = nn.Embedding(config.num_branches, config.lstm_size)
        
        self.w_attn_1 = nn.Linear(config.lstm_size, config.lstm_size)
        
        self.opt_sampler = OptSampler(config.lstm_size, config.num_branches)
        self.index_sampler = IndexSampler(config.lstm_size)
        
        
    def forward(self, prev_h=None, prev_c=None, use_bias=False):
        """Search for a cell stracture.
           
        Args:
            prev_h: None (normal cell) or the last output c of LSTM (reduction).
            prev_c: None (normal cell) or the last output c of LSTM (reduction).
            use_bias: if the no-learn bias is used.
            
        Returns:
            prev_h, prev_c: the last output of LSTM
            arc_seq: list of integers, [index_1, index_2, op_1, op_2] * node_num
            logits: list with length equals to node number. Each element in the list is
                also a list that contains 4 tensors with size [1, num_of_prev_nodes or
                num_of_branches].
        """
        arc_seq, logits = [], []
        anchors, anchors_w = [], [] # anchors to save all generated nodes
        
        if prev_c is None and prev_h is None:
            prev_h = torch.zeros([self.config.num_lstm_layer, 1, self.config.lstm_size], 
                                 device=self.device)
            prev_c = torch.zeros([self.config.num_lstm_layer, 1, self.config.lstm_size], 
                                 device=self.device)
        else:
            assert prev_c is not None and prev_h is not None, "Prev_c and prev_h mush both be None!"
        
        # sample 2 inputs
        h, c = self._sample_input_nodes(self.g_emb, prev_h, prev_c, anchors, anchors_w)
        
        # sample the rest B - 2 nodes
        for i in range(self.config.node_num):
            (h, c), node_logits = self._sample_node(self.g_emb, h, c, arc_seq, 
                                                    anchors, anchors_w)
            assert len(anchors) == len(anchors_w) == i + 3
            logits.append(node_logits)
            
        assert len(arc_seq) // 4 == len(logits)
        return (h, c), arc_seq, logits
    
    
    def _sample_input_nodes(self, inputs, h, c, anchors, anchors_w):
        """Generate the first two input nodes, which are the outpus of previous cells,
           and save them to anchors.
        
        Args:
            inputs: tensor, [1, batch_size, lstm_size]
            h: tensor, [num_layers, batch_size, hidden_size]
            c: tensor, [num_layers, batch_size, hidden_size]
            anchors: list of tensor: [1, 1, lstm_size]
            anchors_w: list of tensor: : [1, lstm_size]
            
        Returns:
            h, c: the output of LSTM at 5th step
        """            
        for i in range(2):
            output, (h, c) = self.lstm(inputs, (h, c))
            anch = torch.zeros_like(h[-1], device=self.device)
            anchors.append(torch.unsqueeze(anch, dim=0))
            anchors_w.append(self.w_attn_1(h[-1]))
            
        return h, c
        
        
    def _sample_node(self, inputs, prev_h, prev_c, arc_seq, anchors, anchors_w):
        """Sample one node which has 2 indexs and 2 operations. 
           
           Note that there are in total 5 steps for LSTM. The first two steps to sample 
           two indexs, the second 2 steps to sample two operations, and the least step 
           to add the node to anchors.
        
        Returns:
            h, c: the output of LSTM at 5th step.
            node_logits: list of 4 tensors with size [1, num_of_prev_nodes or
                num_of_branches]
        """
        node_logits = []
        
        # the first two steps: sample indexs
        query = torch.cat(anchors_w, dim=0)
        assert query.shape == (len(anchors_w), self.config.lstm_size)
        
        for i in range(2):            
            (h, c), index, logits = self._sample_index(inputs, prev_h, prev_c, query)
            prev_h, prev_c, inputs = h, c, anchors[index]
            assert inputs.shape == (1, 1, self.config.lstm_size), "Oops, LSTM input size seems wrong!"
            
            arc_seq.append(index)
            node_logits.append(logits)
            
        # the second two steps: sample operations
        for i in range(2):
            (h, c), op_id, logits = self._sample_opera(inputs, prev_h, prev_c)
            tensor_id = torch.tensor([[op_id]], dtype=torch.long).to(self.device)
            
            prev_h, prev_c, inputs = h, c, self.w_emb(tensor_id)
            assert inputs.shape == (1, 1, self.config.lstm_size), "Oops, LSTM input size seems wrong!"
            
            arc_seq.append(op_id)
            node_logits.append(logits)
        
        # one more step: add the node to anchors
        h, c = self._add_node_to_anchors(inputs, prev_h, prev_c, anchors, anchors_w)
        
        return (h, c), node_logits
        
        
    def _sample_index(self, inputs, h, c, query):
        """Sample index: find out the input node.
        
        Returns:
            (h, c): tensors, the hidden state of the top layer LSTM.
            index: int
            logits: tensor, [1, num_of_existing_nodes]
        """
        # attention
        output, (h, c) = self.lstm(inputs, (h, c))
        index, logits = self.index_sampler(h, query, 
                                           self.config.temperature, 
                                           self.config.tanh_constant)
        return (h, c), index, logits
    
    
    def _sample_opera(self, inputs, h, c):
        """Sample operation.
        
        Args:
            inputs: [1, 1, lstm_size]
            h, c: previous hidden LSTM state.

        Returns:
            (h, c): tensors, the hidden state of the top layer LSTM.
            op_id: int
            logits: tensor, [1, num_of_branches]         
        """
        output, (h, c) = self.lstm(inputs, (h, c))
        op_id, logits = self.opt_sampler(h, 
                                         self.config.temperature, 
                                         self.config.op_tanh_constant)
        return (h, c), op_id, logits

    
    def _add_node_to_anchors(self, inputs, h, c, anchors, anchors_w):
        """The final step of generating a node: one more forward step of LSTM and 
           add the hidden state to anchors.
        """
        output, (h, c) = self.lstm(inputs, (h, c))
        anchors.append(torch.unsqueeze(h[-1], dim=0))
        anchors_w.append(self.w_attn_1(h[-1]))
        
        return (h, c)
    

class OptSampler(nn.Module):
    """Sample operation"""
    
    def __init__(self, lstm_size, num_branches, use_bias=True):
        super(OptSampler, self).__init__()
        
        self.w_soft = nn.Linear(lstm_size, num_branches)
        self.b_soft = nn.Parameter(torch.tensor([[10.0]*2 + [0.]*(num_branches - 2)], 
            requires_grad=True))
        self.logits_proc = TempAndTC()
        if use_bias:
            self.b_soft_no_learn = nn.Parameter(torch.tensor(
                [[0.25]*2 + [-0.25]*(num_branches - 2)], requires_grad=False))
            
        assert self.b_soft.shape == self.b_soft_no_learn.shape == (1, num_branches)
        
    def forward(self, h, temperature=None, tanh_constant=None):
        """
        Args:
            h: tensor, [layer_num, batch_size, lstm_size]
            temperature: float
            tanh_constant: float
            
        Returns:
            index: int
            logits: tensor, [1, num_branches]
        """
        logits = self.w_soft(h[-1]) + self.b_soft
        logits = self.logits_proc(logits, temperature, tanh_constant) # [1, num_branches]
        if self.b_soft_no_learn is not None:
            logits += self.b_soft_no_learn
        
        # sample operation id
        op_id = torch.multinomial(logits.exp(), 1).item()
        
        return op_id, logits


class IndexSampler(nn.Module):
    """Sample index"""
    
    def __init__(self, lstm_size):
        super(IndexSampler, self).__init__()
        self.w_attn_2 = nn.Linear(lstm_size, lstm_size)
        self.v_attn = nn.Linear(lstm_size, 1)
        self.logits_proc = TempAndTC()
        
    def forward(self, h, query, temperature=None, tanh_constant=None):
        """
        Args:
            h: tensor, [layer_num, batch_size, lstm_size]
            query: tensor, [node_num, lstm_size]
            temperature: float
            tanh_constant: float
            
        Returns:
            index: int
            logits: tensor, [1, num_exitsing_nodes]
        """
        logits = self.v_attn(torch.tanh(query + self.w_attn_2(h[-1])))
        logits = self.logits_proc(logits.view(1, -1), temperature, tanh_constant)
        
        # sample index using logits
        index = torch.multinomial(logits.exp(), 1).item()
        
        return index, logits


class TempAndTC(nn.Module):
    """Apply tanh constant and temperature to controller's logits"""
    
    def __init__(self):
        super(TempAndTC, self).__init__()
        
    def forward(self, logits, temperature=None, tanh_constant=None):
        """
        Args:
            logits: tensor
            temperature: float
            tanh_constant: float
        """
        if temperature is not None:
            logits /= temperature
        if tanh_constant is not None:
            logits = tanh_constant * torch.tanh(logits)
        return logits