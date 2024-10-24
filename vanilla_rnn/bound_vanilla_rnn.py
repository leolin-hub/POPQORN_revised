#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 08:36:29 2018

@author: root

"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
from torchvision import datasets
import numpy as np

import get_bound_for_general_activation_function as get_bound
from utils.sample_data import sample_mnist_data
from utils.verify_bound import verify_final_output, verifyMaximumEps

import os
import argparse

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_step, activation):
        # the num_layers here is the number of RNN units
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # Add hidden_size attribute
        self.rnn = nn.RNN(input_size = input_size,hidden_size=hidden_size,
                          num_layers = 1,batch_first=True, nonlinearity = activation)
        self.out = nn.Linear(hidden_size , output_size )
        self.X = None #data attached to the classifier, and is of size (N,self.time_step,self.input_size)
        self.l = [None]*(time_step+1) #l[0] has no use, l[k] is of size (N,n_k), k from 1 to m
        # l[k] is the pre-activation lower bound of the k-th layer  
        self.u = [None]*(time_step+1) #u[0] has no use, u[k] is of size (N,n_k), k from 1 to m
        #u[k] is the pre-activation upper bound of the k-th layer 
        self.kl = [None]*(time_step+1)
        self.ku = [None]*(time_step+1)
        
        self.bl = [None]*(time_step+1)
        self.bu = [None]*(time_step+1)
        
        self.time_step = time_step #number of vanilla layers 
        self.num_neurons = hidden_size #numbers of neurons in each layer, a scalar
        self.input_size = input_size 
        self.output_size = output_size
        
        self.activation = activation
        if activation == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation == 'relu':
            self.activation_function = nn.ReLU()
        else:
            raise Exception(activation+' activation function is not supported')

        self.W_fa = None  # [output_size, num_neurons] 
        self.W_aa = None  # [num_neurons, num_neurons]
        self.W_ax = None  # [num_neurons, input_size]
        self.b_f = None   # [output_size]
        self.b_ax = None  # [num_neurons]
        self.b_aa = None  # [num_neurons]
        self.a_0 = None   # initial hidden state     
        
    def forward(self,X):
        # X is of size (batch, seq_len, input_size)
        r_out, _ = self.rnn(X, self.a_0) # RNN usage: output, hn = rnn(input,h0)
        # r_out is of size (batch, seq_len, hidden_size)
        # h_n is of size (batch, num_layers, hidden_size)
        out = self.out(r_out[:,-1,:])
        return out      
    
    def clear_intermediate_variables(self):
        time_step = self.time_step
        self.l = [None]*(time_step+1) 
        self.u = [None]*(time_step+1) 
        self.kl = [None]*(time_step+1)
        self.ku = [None]*(time_step+1)
        self.bl = [None]*(time_step+1)
        self.bu = [None]*(time_step+1)
        
    def reset_seq_len(self, seq_len):
        self.time_step = seq_len
        self.clear_intermediate_variables()
    
    def get_preactivation_output(self, X):
        #k range from 1 to self.num_layers
        #get the pre-relu output of the k-th layer
        #X is of shape (N, seq_len, in_features)
        with torch.no_grad():
            device = X.device
            N = X.shape[0]
            seq_len = X.shape[1]
            h = torch.zeros([N, seq_len+1, self.hidden_size], device = device)
            pre_h = torch.zeros([N, seq_len+1, self.hidden_size], device = device)
            b_ax = self.b_ax.unsqueeze(0).expand(N,-1)
            b_aa = self.b_aa.unsqueeze(0).expand(N,-1)
            for i in range(seq_len):
                pre_h[:,i+1,:] = (torch.matmul(self.W_ax, X[:,i,:].unsqueeze(2)).squeeze(2)
                    + b_ax +
                    torch.matmul(self.W_aa, h[:,i,:].unsqueeze(2)).squeeze(2) + b_aa) # 下一層的pre-activation = W_aa * a_(t-1) + W_ax * x_t + b_aa + b_ax
                h[:,i+1,:] = self.activation_function(pre_h[:,i+1,:]) # 下一層的activation = f(pre-activation)
        return pre_h[:,1:,:], h[:,1:,:]
                
    def attachData(self, X):
        # X is of size N*C*H*W
        if torch.numel(X) == (X.shape[0] * X.shape[1] * X.shape[2] * X.shape[3]): # number of elements
            X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
            self.X = X
        else:
            raise Exception('The input dimension must be %d' % (X.shape[1] * X.shape[2] * X.shape[3]))
    
    def extractWeight(self, clear_original_model=True):
        with torch.no_grad():
            # torch original method, directly retrieving the weights of hidden and input layer
            self.W_fa = self.out.weight  # [output_size, num_neurons] 
            self.W_aa = self.rnn.weight_hh_l0  # [num_neurons, num_neurons]
            self.W_ax = self.rnn.weight_ih_l0  # [num_neurons, input_size]
            self.b_f = self.out.bias   # [output_size]
            self.b_ax = self.rnn.bias_ih_l0  # [num_neurons]
            self.b_aa = self.rnn.bias_hh_l0  # [num_neurons]
            # a_t = tanh(W_ax x_(t-1) + b_ax + W_aa a_(t-1) + b_aa)
            if clear_original_model:
                self.rnn = None
        return 0
        
    def compute2sideBound(self, eps, p, v, X = None, Eps_idx = None): # add max_splits
        """
        Compute two-sided bounds for a given input using a recurrent neural network.
        Parameters:
            eps (torch.Tensor): Perturbation tensor, shape (N, s) where N is the batch size and s is the hidden size.
            p (float or str): The p-norm to be used for the computation. Can be a float or 'inf'.
            v (int): The current timestep or layer index for which bounds are computed.
            X (torch.Tensor, optional): Input tensor, shape (N, time_step, n) where n is the input size.
            Eps_idx (torch.Tensor, optional): Indices of the timesteps where perturbations occur, shape (k,) where k is the number of perturbations.
            max_splits (int, optional): Maximum number of splits to perform on dimensions, default is 3.
        Returns:
            tuple: A tuple containing:
                - yL (torch.Tensor): Lower bound tensor, shape (N, s).
                - yU (torch.Tensor): Upper bound tensor, shape (N, s).
        Notes:
            - The function initializes bounds and computes them based on the specified norms.
            - It handles ReLU activation differently by applying zero-splits if necessary.
            - The function assumes that self.l and self.u are initialized and have the correct sizes for the given layer index v.
        """

        with torch.no_grad():
            n = self.W_ax.shape[1]  # input_size
            s = self.W_ax.shape[0]  # hidden_size     
            idx_eps = torch.zeros(self.time_step, device=X.device) # 一維tensor，長度time_step
            idx_eps[Eps_idx-1] = 1 # Eps_idx表示在哪個timestep擾動
            if X is None:
                X = self.X
            N = X.shape[0]  # number of images, batch size
            if self.a_0 is None:
                a_0 = torch.zeros(N, s, device = X.device)
            else:
                a_0 = self.a_0
            if type(eps) == torch.Tensor:
                eps = eps.to(X.device)
            # Deal with p-norm and q-norm, when p is 1, q is infinite
            # When p is infinite, q is 1, otherwise, q = p / (p - 1)
            # If 1/p + 1/q = 1, this is a dual-norm
            if p == 1:
                q = float('inf')
            elif p == 'inf' or p==float('inf'):
                q = 1 
            else:
                q = p / (p-1)
        
            yU = torch.zeros(N, s, device = X.device)  # [N,s]
            yL = torch.zeros(N, s, device = X.device)  # [N,s]
        
            W_ax = self.W_ax.unsqueeze(0).expand(N,-1,-1)  # [N, s, n]
            W_aa = self.W_aa.unsqueeze(0).expand(N,-1,-1)  # [N, s, s]    
            b_ax = self.b_ax.unsqueeze(0).expand(N,-1)  # [N, s]
            b_aa = self.b_aa.unsqueeze(0).expand(N,-1)
        
            # 在計算邊界之前應用 zeroSplit
            # v代表當前的timestep，位於第幾層hidden layer
            if self.activation == 'relu' and v > 1:
                # impact = torch.abs(self.u[v-1] - self.l[v-1]).sum(dim=0) # 待確認
                # 使用 any() 來檢查每個維度是否有任何元素需要分割，會是一個需要分割的list
                # 識別所有跨越零的維度
                zero_crossing = (self.l[v-1] < 0) & (self.u[v-1] > 0)
                dimensions_to_split = torch.where(zero_crossing.any(dim=0))[0]
                # dimensions_to_split = torch.argsort(impact, descending=True)[:max_splits]
                abstractions = [(self.l[v-1], self.u[v-1])] # [N, s]
                
                for split_depth in range(1, len(dimensions_to_split) + 1):
                    if len(abstractions) > 0: # 限制抽象的總數
                        print(f"Warning: Reached maximum number of abstractions (100). Stopping further splits.")
                        break
                    #dim = dimensions_to_split.pop(0)
                    new_abstractions = []
                    for l_abs, u_abs in abstractions:
                        new_abstractions.extend(self.zeroSplit((l_abs, u_abs), dimensions_to_split[:split_depth])) # 重複確定需要zeroSplit
                    abstractions = new_abstractions
                    print(f"Split {split_depth}: Number of abstractions = {len(abstractions)}")
                new_yL = torch.full_like(yL, 1e10) # 考慮使用一個大但有限的數字 用1e10取代float('inf')
                new_yU = torch.full_like(yU, -1e10)

                for i, (l_abs, u_abs) in enumerate(abstractions):
                    # 確保 l_abs <= u_abs
                    l_abs, u_abs = torch.min(l_abs, u_abs), torch.max(l_abs, u_abs)
                    temp_yL, temp_yU = self._compute_bounds(eps, p, v, X, Eps_idx, l_abs, u_abs, n, s, N, a_0, W_ax, W_aa, b_ax, b_aa, q)
                    new_yL = torch.min(new_yL, temp_yL)
                    new_yU = torch.max(new_yU, temp_yU)
                    if i % 10 == 0:
                        print(f"Processed {i+1}/{len(abstractions)} abstractions")

                yL, yU = new_yL, new_yU
            else:
                # 對於非 ReLU 激活函數或第一層，使用原來的邊界計算方法
                yL, yU = self._compute_bounds(eps, p, v, X, Eps_idx, self.l[v-1] if v > 1 else None, self.u[v-1] if v > 1 else None, n, s, N, a_0, W_ax, W_aa, b_ax, b_aa, q)
            
            # 處理可能的 NaN 值
            yL = torch.where(torch.isnan(yL), torch.full_like(yL, -float('inf')), yL)
            yU = torch.where(torch.isnan(yU), torch.full_like(yU, float('inf')), yU)

            # 經zeroSplit和compute bound之後的上下界
            self.l[v] = yL
            self.u[v] = yU
            print(f"Layer {v} bounds: min(yL) = {yL.min().item()}, max(yU) = {yU.max().item()}")
            return yL, yU

    def _compute_bounds(self, eps, p, v, X, Eps_idx, l_prev, u_prev, n, s, N, a_0, W_ax, W_aa, b_ax, b_aa, q):
        # 初始化上下界
        yU = torch.zeros(N, s, device = X.device)  # [N,s]
        yL = torch.zeros(N, s, device = X.device)  # [N,s]
    
        # v-th terms, three terms        
        ## first term for epsilon (處理擾動)
        if type(eps) == torch.Tensor:                      
            #eps is a tensor of size N 
            yU = yU + Eps_idx[v-1]*eps.unsqueeze(1).expand(-1, s)*torch.norm(W_ax,p=q,dim=2)  # eps ||A^ {<v>} W_ax||q    
            yL = yL - Eps_idx[v-1]*eps.unsqueeze(1).expand(-1, s)*torch.norm(W_ax,p=q,dim=2)  # eps ||Ou^ {<v>} W_ax||q      
        else:
            yU = yU + Eps_idx[v-1]*eps*torch.norm(W_ax,p=q,dim=2)  # eps ||A^ {<v>} W_ax||q    
            yL = yL - Eps_idx[v-1]*eps*torch.norm(W_ax,p=q,dim=2)  # eps ||Ou^ {<v>} W_ax||q  
        ## second term for current timestep input
        if v == 1:
            X = X.view(N,1,n)
        yU = yU + torch.matmul(W_ax,X[:,v-1,:].view(N,n,1)).squeeze(2)  # A^ {<v>} W_ax x^{<v>}            
        yL = yL + torch.matmul(W_ax,X[:,v-1,:].view(N,n,1)).squeeze(2)  # Ou^ {<v>} W_ax x^{<v>}       
        ## third term for bias
        yU = yU + b_aa + b_ax  # A^ {<v>} (b_a + Delta^{<v>})
        yL = yL + b_aa + b_ax  # Ou^ {<v>} (b_a + Theta^{<v>})
                    
        if v > 1:
            # k from v-1 to 1 terms
            for k in range(v-1,0,-1): 
                # 計算Activation Function的linear bound
                if k == v-1:
                    ## compute A^{<v-1>}, Ou^{<v-1>}, Delta^{<v-1>} and Theta^{<v-1>}
                    ### 1. compute slopes alpha and intercepts beta, kl and ku for lower and upper bound of slopes,
                    ### bl and bu for lower and upper bound of intercepts
                    kl, bl, ku, bu = get_bound.getConvenientGeneralActivationBound(
                        l_prev, u_prev, self.activation)
                
                    # bl = bl/kl
                    # bu = bu/ku
                    epsilon = 1e-10
                    bl = torch.where(kl.abs() > epsilon, bl / kl, torch.zeros_like(bl))
                    bu = torch.where(ku.abs() > epsilon, bu / ku, torch.zeros_like(bu))
                
                    self.kl[k] = kl  # [N, s]
                    self.ku[k] = ku  # [N, s]
                    self.bl[k] = bl  # [N, s]
                    self.bu[k] = bu  # [N, s]
                    alpha_l = kl.unsqueeze(1).expand(-1, s, -1)  # [N, s, s]
                    alpha_u = ku.unsqueeze(1).expand(-1, s, -1)  # [N, s, s]
                    beta_l = bl.unsqueeze(1).expand(-1, s, -1)  # [N, s, s]
                    beta_u = bu.unsqueeze(1).expand(-1, s, -1)  # [N, s, s]
                    ### 2. compute lambda^{<v-1>}, omega^{<v-1>}, Delta^{<v-1>} and Theta^{<v-1>}
                    I = (W_aa >= 0).float()  # [N, s, s]
                    lamida = I*alpha_u + (1-I)*alpha_l                  
                    omiga = I*alpha_l + (1-I)*alpha_u
                    Delta = I*beta_u + (1-I)*beta_l  # [N, s, s], this is the transpose of the delta defined in the paper
                    Theta = I*beta_l + (1-I)*beta_u  # [N, s, s]
                    ### 4. compute A^{<v-1>} and Ou^{<v-1>}
                    A = W_aa * lamida  # [N, s, s]
                    Ou = W_aa * omiga  # [N, s, s]
                else:
                    ## compute A^{<k>}, Ou^{<k>}, Delta^{<k>} and Theta^{<k>}
                    ### 1. compute slopes alpha and intercepts beta
                    alpha_l = self.kl[k].unsqueeze(1).expand(-1, s, -1)
                    alpha_u = self.ku[k].unsqueeze(1).expand(-1, s, -1)
                    beta_l = self.bl[k].unsqueeze(1).expand(-1, s, -1)  # [N, s, s]
                    beta_u = self.bu[k].unsqueeze(1).expand(-1, s, -1)  # [N, s, s]
                    ### 2. compute lambda^{<k>}, omega^{<k>}, Delta^{<k>} and Theta^{<k>}
                    I = (torch.matmul(A,W_aa) >= 0).float()  # [N, s, s]
                    lamida = I*alpha_u + (1-I)*alpha_l                  
                    Delta = I*beta_u + (1-I)*beta_l  # [N, s, s], this is the transpose of the delta defined in the paper
                    I = (torch.matmul(Ou,W_aa) >= 0).float()  # [N, s, s]
                    omiga = I*alpha_l + (1-I)*alpha_u
                    Theta = I*beta_l + (1-I)*beta_u  # [N, s, s]
                    ### 3. compute A^{<k>} and Ou^{<k>}
                    A = torch.matmul(A,W_aa) * lamida  # [N, s, s]
                    Ou = torch.matmul(Ou,W_aa) * omiga  # [N, s, s]
                ## first term
                if type(eps) == torch.Tensor:                
                    #eps is a tensor of size N 
                    yU = yU + Eps_idx[k-1]*eps.unsqueeze(1).expand(-1,
                            s)*torch.norm(torch.matmul(A,W_ax),p=q,dim=2)  # eps ||A^ {<k>} W_ax||q    
                    yL = yL - Eps_idx[k-1]*eps.unsqueeze(1).expand(-1,
                            s)*torch.norm(torch.matmul(Ou,W_ax),p=q,dim=2)  # eps ||Ou^ {<k>} W_ax||q      
                else:
                    yU = yU + Eps_idx[k-1]*eps*torch.norm(torch.matmul(A,W_ax),p=q,dim=2)  # eps ||A^ {<k>} W_ax||q    
                    yL = yL - Eps_idx[k-1]*eps*torch.norm(torch.matmul(Ou,W_ax),p=q,dim=2)  # eps ||Ou^ {<k>} W_ax||q  
                ## second term
                yU = yU + torch.matmul(A,torch.matmul(W_ax,X[:,k-1,:].view(N,n,1))).squeeze(2)  # A^ {<k>} W_ax x^{<k>}            
                yL = yL + torch.matmul(Ou,torch.matmul(W_ax,X[:,k-1,:].view(N,n,1))).squeeze(2)  # Ou^ {<k>} W_ax x^{<k>}       
                ## third term
                yU = yU + torch.matmul(A,(b_aa+b_ax).view(N,s,1)).squeeze(2)+(A*Delta).sum(2)  # A^ {<k>} (b_a + Delta^{<k>})
                yL = yL + torch.matmul(Ou,(b_aa+b_ax).view(N,s,1)).squeeze(2)+(Ou*Theta).sum(2)  # Ou^ {<k>} (b_a + Theta^{<k>})
            # compute A^{<0>}
            A = torch.matmul(A,W_aa)  # (A^ {<1>} W_aa) * lambda^{<0>}
            Ou = torch.matmul(Ou,W_aa)  # (Ou^ {<1>} W_aa) * omega^{<0>}
        else:
            A = W_aa  # A^ {<0>}, [N, s, s]
            Ou = W_aa  # Ou^ {<0>}, [N, s, s]
        yU = yU + torch.matmul(A,a_0.view(N,s,1)).squeeze(2)  # A^ {<0>} * a_0
        yL = yL + torch.matmul(Ou,a_0.view(N,s,1)).squeeze(2)  # Ou^ {<0>} * a_0
        # 確保 yL <= yU
        yL, yU = torch.min(yL, yU), torch.max(yL, yU)
                
        return yL, yU

    
    def computeLast2sideBound(self, eps, p, v, X = None, Eps_idx = None, max_splits=3):
        with torch.no_grad():
            n = self.W_ax.shape[1]  # input_size
            s = self.W_ax.shape[0]  # hidden_size     
            t = self.W_fa.shape[0]  # output_size
            idx_eps = torch.zeros(self.time_step, device=X.device)
            idx_eps[Eps_idx-1] = 1
            if X is None:
                X = self.X
            N = X.shape[0]  # number of images, batch size
            if self.a_0 is None:
                a_0 = torch.zeros(N, s, device = X.device)
            else:
                a_0 = self.a_0
            if type(eps) == torch.Tensor:
                eps = eps.to(X.device)        
            if p == 1:
                q = float('inf')
            elif p == 'inf':
                q = 1
            else:
                q = p / (p-1)
            
            yU = torch.zeros(N, t, device = X.device)  # [N,s]
            yL = torch.zeros(N, t, device = X.device)  # [N,s]
            
            W_ax = self.W_ax.unsqueeze(0).expand(N,-1,-1)  # [N, s, n]
            W_aa = self.W_aa.unsqueeze(0).expand(N,-1,-1)  # [N, s, s]   
            W_fa = self.W_fa.unsqueeze(0).expand(N,-1,-1)  # [N, t, s]  
            b_ax = self.b_ax.unsqueeze(0).expand(N,-1)  # [N, s]
            b_aa = self.b_aa.unsqueeze(0).expand(N,-1)  # [N, s]
            b_f = self.b_f.unsqueeze(0).expand(N,-1)  # [N, t]
                            
            # k from time_step+1 to 1 terms
            for k in range(v-1,0,-1): 
                if k == v-1:
                    ## compute A^{<v-1>}, Ou^{<v-1>}, Delta^{<v-1>} and Theta^{<v-1>}
                    ### 1. compute slopes alpha and intercepts beta
                    kl, bl, ku, bu = get_bound.getConvenientGeneralActivationBound(
                                self.l[k], self.u[k], self.activation)
                    
                    # bl = bl/kl
                    # bu = bu/ku
                    # 在計算 bl/kl 和 bu/ku 時避免除以零
                    epsilon = 1e-8
                    bl = torch.where(kl != 0, bl / (kl + epsilon), bl)
                    bu = torch.where(ku != 0, bu / (ku + epsilon), bu)
                    
                    self.kl[k] = kl  # [N, s]
                    self.ku[k] = ku  # [N, s]
                    self.bl[k] = bl  # [N, s]
                    self.bu[k] = bu  # [N, s]
                    alpha_l = kl.unsqueeze(1).expand(-1, t, -1)  # [N, t, s]
                    alpha_u = ku.unsqueeze(1).expand(-1, t, -1)  # [N, t, s]
                    beta_l = bl.unsqueeze(1).expand(-1, t, -1)  # [N, t, s]
                    beta_u = bu.unsqueeze(1).expand(-1, t, -1)  # [N, t, s]
                    ### 2. compute lambda^{<v-1>}, omega^{<v-1>}, Delta^{<v-1>} and Theta^{<v-1>}
                    epsilon = 1e-8 # to avoid division by zero
                    I = (W_fa >= 0).float()  # [N, t, s]
                    lamida = I*alpha_u + (1-I)*alpha_l                  
                    omiga = I*alpha_l + (1-I)*alpha_u

                    # 在計算 Delta 之後立即添加這些檢查
                    print(f"beta_u range: [{beta_u.min().item()}, {beta_u.max().item()}]")
                    print(f"beta_l range: [{beta_l.min().item()}, {beta_l.max().item()}]")

                    Delta = I*beta_u + (1-I)*beta_l + epsilon  # [N, t, s], this is the transpose of the delta defined in the paper

                    print(f"Delta initial range: [{Delta.min().item()}, {Delta.max().item()}]")

                    # 清理 Delta 中的 NaN 和 Inf
                    #Delta = torch.nan_to_num(Delta, nan=0.0, posinf=1e20, neginf=-1e20)

                    # 使用 clamp 來限制 Delta 的範圍
                    #Delta = torch.clamp(Delta, min=-1e20, max=1e20)

                    print(f"Delta cleaned range: [{Delta.min().item()}, {Delta.max().item()}]")

                    Theta = I*beta_l + (1-I)*beta_u  # [N, t, s]

                    # 對 Theta 進行與 Delta 相同的處理
                    print(f"Theta initial range: [{Theta.min().item()}, {Theta.max().item()}]")
                    #Theta = torch.nan_to_num(Theta, nan=0.0, posinf=1e20, neginf=-1e20)
                    #Theta = torch.clamp(Theta, min=-1e20, max=1e20)
                    #print(f"Theta cleaned range: [{Theta.min().item()}, {Theta.max().item()}]")

                    ### 3. clear l[k] and u[k] to release memory
                    self.l[k] = None
                    self.u[k] = None
                    ### 4. compute A^{<v-1>} and Ou^{<v-1>}
                    A = W_fa * lamida  # [N, t, s]
                    Ou = W_fa * omiga  # [N, t, s]
                else:
                    ## compute A^{<k>}, Ou^{<k>}, Delta^{<k>} and Theta^{<k>}
                    ### 1. compute slopes alpha and intercepts beta
                    alpha_l = self.kl[k].unsqueeze(1).expand(-1, t, -1)
                    alpha_u = self.ku[k].unsqueeze(1).expand(-1, t, -1)
                    beta_l = self.bl[k].unsqueeze(1).expand(-1, t, -1)  # [N, t, s]
                    beta_u = self.bu[k].unsqueeze(1).expand(-1, t, -1)  # [N, t, s]

                    # 在計算 Delta 之後立即添加這些檢查
                    print(f"beta_u range: [{beta_u.min().item()}, {beta_u.max().item()}]")
                    print(f"beta_l range: [{beta_l.min().item()}, {beta_l.max().item()}]")

                    ### 2. compute lambda^{<k>}, omega^{<k>}, Delta^{<k>} and Theta^{<k>}
                    I = (torch.matmul(A,W_aa) >= 0).float()  # [N, t, s]
                    lamida = I*alpha_u + (1-I)*alpha_l                  
                    Delta = I*beta_u + (1-I)*beta_l + epsilon  # [N, s, s], this is the transpose of the delta defined in the paper

                    print(f"Delta initial range: [{Delta.min().item()}, {Delta.max().item()}]")
                    # 清理 Delta 中的 NaN 和 Inf
                    #Delta = torch.nan_to_num(Delta, nan=0.0, posinf=1e20, neginf=-1e20)
                    # 使用 clamp 來限制 Delta 的範圍
                    #Delta = torch.clamp(Delta, min=-1e20, max=1e20)
                    print(f"Delta cleaned range: [{Delta.min().item()}, {Delta.max().item()}]")

                    I = (torch.matmul(Ou,W_aa) >= 0).float()  # [N, t, s]
                    omiga = I*alpha_l + (1-I)*alpha_u
                    Theta = I*beta_l + (1-I)*beta_u  # [N, s, s]

                    # 對 Theta 進行與 Delta 相同的處理
                    print(f"Theta initial range: [{Theta.min().item()}, {Theta.max().item()}]")
                    #Theta = torch.nan_to_num(Theta, nan=0.0, posinf=1e20, neginf=-1e20)
                    #Theta = torch.clamp(Theta, min=-1e20, max=1e20)
                    print(f"Theta cleaned range: [{Theta.min().item()}, {Theta.max().item()}]")

                    ### 3. compute A^{<k>} and Ou^{<k>}
                    A = torch.matmul(A,W_aa) * lamida  # [N, s, s]
                    Ou = torch.matmul(Ou,W_aa) * omiga  # [N, s, s]
                ## first term
                if type(eps) == torch.Tensor:                
                    #eps is a tensor of size N 
                    yU = yU + idx_eps[k-1]*eps.unsqueeze(1).expand(-1,
                               t)*torch.norm(torch.matmul(A,W_ax),p=q,dim=2)  # eps ||A^ {<k>} W_ax||q    
                    yL = yL - idx_eps[k-1]*eps.unsqueeze(1).expand(-1,
                               t)*torch.norm(torch.matmul(Ou,W_ax),p=q,dim=2)  # eps ||Ou^ {<k>} W_ax||q      
                else:
                    yU = yU + idx_eps[k-1]*eps*torch.norm(torch.matmul(A,W_ax),p=q,dim=2)  # eps ||A^ {<k>} W_ax||q    
                    yL = yL - idx_eps[k-1]*eps*torch.norm(torch.matmul(Ou,W_ax),p=q,dim=2)  # eps ||Ou^ {<k>} W_ax||q  
                ## second term
                yU = yU + torch.matmul(A,torch.matmul(W_ax,X[:,k-1,:].view(N,n,1))).squeeze(2)  # A^ {<k>} W_ax x^{<k>}            
                yL = yL + torch.matmul(Ou,torch.matmul(W_ax,X[:,k-1,:].view(N,n,1))).squeeze(2)  # Ou^ {<k>} W_ax x^{<k>}       
                ## third term

                # 轉換為 float64 以提高精度
                # A = A.to(torch.float64)
                # Delta = Delta.to(torch.float64)
                # Ou = Ou.to(torch.float64)
                # Theta = Theta.to(torch.float64)
                # b_aa = b_aa.to(torch.float64)
                # b_ax = b_ax.to(torch.float64)
                # yU = yU.to(torch.float64)

                # 打印出 A 和 Delta 的形狀
                print(f"A shape: {A.shape}")
                print(f"Delta shape: {Delta.shape}")
                print(f"b_aa shape: {b_aa.shape}, dtype: {b_aa.dtype}")
                print(f"b_ax shape: {b_ax.shape}, dtype: {b_ax.dtype}")

                temp1 = torch.matmul(A, (b_aa + b_ax).view(N, s, 1)).squeeze(2) # Seperate the computation to avoid memory error
                epsilon = 1e-20
                A_abs = torch.clamp(torch.abs(A), min=epsilon, max=1e20)
                Delta_abs = torch.clamp(torch.abs(Delta), min=epsilon, max=1e20)

                print(f"A_abs range: [{A_abs.min().item()}, {A_abs.max().item()}]")
                print(f"Delta_abs range: [{Delta_abs.min().item()}, {Delta_abs.max().item()}]")

                log_A = torch.log(A_abs)
                log_Delta = torch.log(Delta_abs)

                print(f"log_A range: [{log_A.min().item()}, {log_A.max().item()}]")
                print(f"log_Delta range: [{log_Delta.min().item()}, {log_Delta.max().item()}]")

                log_prod = log_A + log_Delta
                print(f"log_prod range: [{log_prod.min().item()}, {log_prod.max().item()}]")

                max_val, _ = torch.max(log_prod, dim=2, keepdim=True)
                print(f"max_val range: [{max_val.min().item()}, {max_val.max().item()}]")

                log_diff = log_prod - max_val
                print(f"log_diff range: [{log_diff.min().item()}, {log_diff.max().item()}]")

                exp_diff = torch.exp(torch.clamp(log_diff, min=-20, max=20))
                print(f"exp_diff range: [{exp_diff.min().item()}, {exp_diff.max().item()}]")

                sum_exp = torch.sum(exp_diff, dim=2)
                print(f"sum_exp range: [{sum_exp.min().item()}, {sum_exp.max().item()}]")

                log_sum = torch.log(sum_exp) + max_val.squeeze(2)
                print(f"log_sum range: [{log_sum.min().item()}, {log_sum.max().item()}]")

                sign = torch.sign(A) * torch.sign(Delta)
                sign_prod = torch.prod(sign, dim=2)

                temp2 = sign_prod * torch.exp(torch.clamp(log_sum, min=-20, max=20))

                # 檢查結果
                print(f"temp2 shape: {temp2.shape}, max: {temp2.max().item()}, min: {temp2.min().item()}, mean: {temp2.mean().item()}")

                if torch.isinf(temp2).any() or torch.isnan(temp2).any():
                    print("Warning: temp2 contains inf or nan values")
                #temp2 = (A*Delta).sum(2)
                yU = yU + temp1 + temp2
                #yU = yU + torch.matmul(A,(b_aa+b_ax).view(N,s,1)).squeeze(2)+(A*Delta).sum(2)  # A^ {<k>} (b_a + Delta^{<k>})

                temp1_L = torch.matmul(Ou, (b_aa + b_ax).view(N, s, 1)).squeeze(2)
                Ou_abs = torch.clamp(torch.abs(Ou), min=epsilon, max=1e20)
                Theta_abs = torch.clamp(torch.abs(Theta), min=epsilon, max=1e20)

                log_Ou = torch.log(Ou_abs)
                log_Theta = torch.log(Theta_abs)

                log_prod_L = log_Ou + log_Theta
                log_prod_L = log_Ou + log_Theta
                max_val_L, _ = torch.max(log_prod_L, dim=2, keepdim=True)
                log_diff_L = log_prod_L - max_val_L
                exp_diff_L = torch.exp(torch.clamp(log_diff_L, min=-20, max=20))
                sum_exp_L = torch.sum(exp_diff_L, dim=2)
                log_sum_L = torch.log(sum_exp_L) + max_val_L.squeeze(2)

                sign_L = torch.sign(Ou) * torch.sign(Theta)
                sign_prod_L = torch.prod(sign_L, dim=2)

                temp2_L = sign_prod_L * torch.exp(torch.clamp(log_sum_L, min=-20, max=20))

                print(f"temp2_L shape: {temp2_L.shape}, max: {temp2_L.max().item()}, min: {temp2_L.min().item()}, mean: {temp2_L.mean().item()}")

                if torch.isinf(temp2_L).any() or torch.isnan(temp2_L).any():
                    print("Warning: temp2_L contains inf or nan values")

                yL = yL + temp1_L + temp2_L
                #yL = yL + torch.matmul(Ou,(b_aa+b_ax).view(N,s,1)).squeeze(2)+(Ou*Theta).sum(2)  # Ou^ {<k>} (b_a + Theta^{<k>})
            # compute A^{<0>}
            A = torch.matmul(A,W_aa)  # (A^ {<1>} W_aa) * lambda^{<0>}
            Ou = torch.matmul(Ou,W_aa)  # (Ou^ {<1>} W_aa) * omega^{<0>}
            yU = yU + torch.matmul(A,a_0.view(N,s,1)).squeeze(2)  # A^ {<0>} * a_0
            yL = yL + torch.matmul(Ou,a_0.view(N,s,1)).squeeze(2)  # Ou^ {<0>} * a_0
            yU = yU + b_f
            yL = yL + b_f
        return yL,yU
    
    def zeroSplit(self, abstraction, dimensions):
        l, u = abstraction
        # 確保 l <= u
        l, u = torch.min(l, u), torch.max(l, u)

        # Initialize results list, only the original abstraction are included
        results = [(l.clone(), u.clone())]
    
        # Split the abstraction along each dimension
        for dim in dimensions:
            new_results = []
            for l_i, u_i in results:
                # 使用 any() 來檢查是否有任何元素在這個維度上跨越0
                if (l_i[..., dim] < 0).any() and (u_i[..., dim] > 0).any():
                    # Split at 0

                    # 創建新的下界,保持下界不變,將大於 0 的上界設為 0
                    l_new, u_new = l_i.clone(), u_i.clone()
                    u_new[..., dim] = torch.where(u_new[..., dim] > 0, torch.zeros_like(u_new[..., dim]), u_new[..., dim])
                    new_results.append((l_i, u_new))

                    # 創建新的上界,保持上界不變,將小於 0 的下界設為 0
                    l_new, u_new = l_i.clone(), u_i.clone()
                    l_new[..., dim] = torch.where(l_new[..., dim] < 0, torch.zeros_like(l_new[..., dim]), l_new[..., dim])
                    new_results.append((l_new, u_i))
                else:
                    # 如果這個區間在該維度上不跨越 0,保持不變
                    new_results.append((l_i, u_i))
            # Update the result as the result of this dim
            results = new_results
    
        return results
    
    def getLastLayerBound(self, eps, p, X = None, clearIntermediateVariables=False, Eps_idx = None):
        # need to be checked
        #eps could be a real number, or a tensor of size N
        with torch.no_grad():
            if self.X is None and X is None:
                raise Exception('You must first attach data to the net or feed data to this function')
            if self.W_fa is None or self.W_aa is None or self.W_ax is None or self.b_f is None or self.b_ax is None or self.b_aa is None:
                self.extractWeight()
            if X is None:
                X = self.X
            if Eps_idx is None:
                Eps_idx = torch.arange(1,self.time_step+1)
            for k in range(1,self.time_step+1):
                # k from 1 to self.time_step
                yL,yU = self.compute2sideBound(eps, p, k, X=X[:,0:k,:], Eps_idx = Eps_idx)

                # 檢查NaN
                if torch.isnan(yL).any() or torch.isnan(yU).any():
                    print(f"NaN detected in step {k}")
                    print(f"yL NaN count: {torch.isnan(yL).sum()}")
                    print(f"yU NaN count: {torch.isnan(yU).sum()}")

                # 數值穩定性
                yL = torch.clamp(yL, min=-1e6, max=1e6)
                yU = torch.clamp(yU, min=-1e6, max=1e6)

                # 確保 yL <= yU
                yL, yU = torch.min(yL, yU), torch.max(yL, yU)
                self.l[k], self.u[k] = yL, yU
            yL,yU = self.computeLast2sideBound(eps, p, self.time_step+1, X, Eps_idx = Eps_idx)  
                #in this loop, self.u, l, Il, WD are reused
            if clearIntermediateVariables:
                self.clear_intermediate_variables()
        return yL, yU
        
    def getMaximumEps(self, p, true_label, target_label, eps0 = 1, max_iter = 100, 
                      X = None, acc = 0.001, gx0_trick = False, Eps_idx = None):
        """
        Finds the maximum epsilon value for a given set of parameters.
        Args:
            p (float): The value of p.
            true_label (int): The true label.
            target_label (int): The target label.
            eps0 (float, optional): The initial value of epsilon. Defaults to 1.
            max_iter (int, optional): The maximum number of iterations. Defaults to 100.
            X (torch.Tensor, optional): The input data. Defaults to None.
            acc (float, optional): The accuracy threshold. Defaults to 0.001. = tao
            gx0_trick (bool, optional): Whether to use the gx0 trick. Defaults to True.
            Eps_idx (torch.Tensor, optional): The epsilon index. Defaults to None.
        Returns:
            torch.Tensor: The lower bound of epsilon.
            torch.Tensor: The upper bound of epsilon.
        """
        # 需要check
        #when u_eps-l_eps < acc, we stop searching
        with torch.no_grad():
            if self.X is None and X is None:
                raise Exception('You must first attach data to the net or feed data to this function')
            if X is None:
                X = self.X
            N = X.shape[0]
            #        time_step=X.shape[1]
            if Eps_idx is None:
                Eps_idx = torch.tensor(range(1,self.time_step+1))#, device = X.device)
                # print('Eps_idx.device', Eps_idx.device)
            if max(Eps_idx) > self.time_step:
                raise Exception('The perturbed frame index should not exceed the number of time step')
            idx=torch.arange(N)
            l_eps = torch.zeros(N, device = X.device) #lower bound of eps
            u_eps = torch.ones(N, device = X.device) * eps0 #upper bound of eps
            
            # use gx0_trick: the "equivalent output node" is equal to N (number of samples in one batch)
            if gx0_trick == True:
                print("Using gx0_trick")
                print("W_fa size = {}".format(self.W_fa.shape))
                print("b_f size = {}".format(self.b_f.shape))             
                self.W_fa = Parameter(self.W_fa[true_label,:]-self.W_fa[target_label,:])
                self.b_f = Parameter(self.b_f[true_label]-self.b_f[target_label])
                print("after gx0_trick W_fa size = {}".format(self.W_fa.shape))
                print("after gx0_trick b_f size = {}".format(self.b_f.shape))
            yL, yU = self.getLastLayerBound(u_eps, p, X = X,  
                                             clearIntermediateVariables=True, Eps_idx = Eps_idx)
            if gx0_trick: 
                 lower = yL[idx,idx]
                 increase_u_eps = lower > 0 # Boolean tensor of size N, label the sample need to increase u_eps, True shows continue, False shows stop
                 print("initial batch f_c^L - f_j^U = {}".format(lower))
                 print("increase_u_eps = {}".format(increase_u_eps))
            else:
                 # in the paper, f_c^L = true_lower, f_j^U = target_upper 
                 true_lower = yL[idx,true_label]
                 target_upper = yU[idx,target_label]
                     
                 #indicate whether to further increase the upper bound of eps 
                 increase_u_eps = true_lower > target_upper 
                 print("initial batch f_c^L - f_j^U = {}".format(true_lower - target_upper))
                 
            ## 1. Find initial upper and lower bound for binary search
            # Add iterations limitation
            max_while_iter = 5
            while_iter = 0
            prev_sum = increase_u_eps.sum()
            no_change_count = 0

            while (increase_u_eps.sum()>0) and (while_iter < max_while_iter):
                #find true and nontrivial lower bound and upper bound of eps
                while_iter += 1
                num = increase_u_eps.sum()
                l_eps[increase_u_eps] = u_eps[increase_u_eps]
                u_eps[increase_u_eps ] = u_eps[increase_u_eps ] * 2

                # 檢查u_eps的變化
                if torch.allclose(u_eps, l_eps, rtol=1e-5, atol=1e-5):
                    print("u_eps and l_eps coverged")
                    break
                yL, yU = self.getLastLayerBound(u_eps[increase_u_eps], p, 
                            X=X[increase_u_eps,:],clearIntermediateVariables=True, Eps_idx = Eps_idx)
                #yL and yU only for those equal to 1 in increase_u_eps
                #they are of size (num,_)
                if gx0_trick:

                    # 確保所有的索引都在有效範圍內
                    valid_mask = idx < yL.shape[0]
                    valid_idx = idx[valid_mask]
                    valid_increase_u_eps = increase_u_eps[valid_mask]

                    # 使用有效的索引來獲取 lower 值
                    lower = yL[torch.arange(len(valid_idx)), valid_idx]

                    # 更新 increase_u_eps，但只更新有效的部分
                    new_increase_u_eps = increase_u_eps.clone()
                    new_increase_u_eps[valid_mask] = valid_increase_u_eps & (lower > 0)

                    # 更新 increase_u_eps
                    increase_u_eps = new_increase_u_eps

                    print(f"Updated increase_u_eps shape: {increase_u_eps.shape}")
                    print(f"Number of True values in increase_u_eps: {increase_u_eps.sum().item()}")

                    # lower = yL[torch.arange(num),idx[increase_u_eps]]
                    # increase_u_eps = increase_u_eps & (lower > 0) # 維度不匹配
                    # temp = lower > 0
                    # print("f_c - f_j = {}".format(lower))
                else:
                    true_lower = yL[torch.arange(num),true_label[increase_u_eps]]
                    target_upper = yU[torch.arange(num),target_label[increase_u_eps]]
                    increase_u_eps[increase_u_eps] = true_lower > target_upper
                    #temp = true_lower > target_upper #size num
                    print("f_c - f_j = {}".format(true_lower- target_upper))
                # 使用 clone() 和布爾索引來更新 increase_u_eps, 把temp註解掉換成上面的寫法
                # new_increase_u_eps = increase_u_eps.clone()
                # new_increase_u_eps[increase_u_eps] = temp
                # increase_u_eps = new_increase_u_eps
                # 檢查 increase_u_eps 的變化
                current_sum = increase_u_eps.sum()
                if current_sum == prev_sum:
                    no_change_count += 1
                    if no_change_count > 5: # 如果連續5次沒有變化，就退出循環
                        print('increase_u_eps not changing for 5 iterations')
                        break
                else:
                    no_change_count = 0
                prev_sum = current_sum

                print(f"Iteration {while_iter}, increase_u_eps sum: {current_sum}")
            
            if while_iter >= max_while_iter:
                print("Reached maximum number of iterations in initial bound finding")
                
            print('Finished finding upper and lower bound')
            print('The upper bound we found is \n', u_eps)
            print('The lower bound we found is \n', l_eps)
            
            #search = (u_eps-l_eps) > acc
            search = (u_eps - l_eps) / ((u_eps+l_eps)/2+1e-8) > acc
            #indicate whether to further perform binary search
            
            #for i in range(max_iter):
            iteration = 0 
            while(search.sum()>0) and (iteration < max_iter):
                #perform binary search
                
                print("search = {}".format(search)) # 全為True時代表有問題，沒有找到正確解
                if iteration > max_iter:
                    print('Have reached the maximum number of iterations')
                    break
                #print(search)
                num = search.sum()
                eps = (l_eps[search]+u_eps[search])/2
                yL, yU = self.getLastLayerBound(eps, p, X=X[search,:],
                                clearIntermediateVariables=True, Eps_idx = Eps_idx)
                print("torch.arange(num) = {}".format(torch.arange(num)))
                if gx0_trick:
                     lower = yL[torch.arange(num),idx[search]]
                     temp = lower > 0
                else:
                     true_lower = yL[torch.arange(num),true_label[search]]
                     target_upper = yU[torch.arange(num),target_label[search]]
                     temp = true_lower>target_upper
                # 使用 clone() 和布爾索引來更新 search
                # new_search = search.clone()
                # new_search[search] = temp
                #            print('search ', search.device)
                #            print('temp ', temp.device)
                #set all active units in search to temp
                #original inactive units in search are still inactive
                
                #l_eps[new_search] = eps[temp]
                l_eps[search] = torch.where(temp, eps, l_eps[search])
                #increase active and true_lower>target_upper units in l_eps 
                
                #u_eps[search & (~new_search)] = eps[~temp]
                u_eps[search] = torch.where(~temp, eps, u_eps[search])
                #decrease active and true_lower<target_upper units in u_eps
                
                # search = (u_eps - l_eps) > acc #reset active units in search
                search = (u_eps - l_eps) / ((u_eps+l_eps)/2+1e-8) > acc
                print('----------------------------------------')
                print(f'Iteration {iteration}:')
                print(f'f_c - f_j = {lower if gx0_trick else true_lower - target_upper}')
                print(f'u_eps - l_eps = {u_eps - l_eps}')
                
                iteration = iteration + 1
        return l_eps, u_eps


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Compute Certified Bound for Vanilla RNNs')

    parser.add_argument('--hidden-size', default = 64, type = int, metavar = 'HS',
                        help = 'hidden layer size (default: 64)')
    parser.add_argument('--time-step', default = 7, type = int, metavar = 'TS',
                        help = 'number of slices to cut the 28*28 image into, it should be a factor of 28 (default: 7)')
    parser.add_argument('--activation', default = 'relu', type = str, metavar = 'a',
                        help = 'nonlinearity used in the RNN, can be either tanh or relu (default: tanh)')
    parser.add_argument('--work-dir', default = 'models/mnist_classifier/rnn_7_64_relu/', type = str, metavar = 'WD',
                        help = 'the directory where the pretrained model is stored and the place to save the computed result')
    parser.add_argument('--model-name', default = 'rnn', type = str, metavar = 'MN',
                        help = 'the name of the pretrained model (default: rnn)')
    
    parser.add_argument('--cuda', action='store_true',
                        help='whether to allow gpu for training')
    parser.add_argument('--cuda-idx', default = 0, type = int, metavar = 'CI',
                        help = 'the index of the gpu to use if allow gpu usage (default: 0)')

    parser.add_argument('--N', default = 100, type = int,
                        help = 'number of samples to compute bounds for (default: 100)')
    parser.add_argument('--p', default = 2, type = int,
                        help = 'p norm, if p > 100, we will deem p = infinity (default: 2)')
    parser.add_argument('--eps0', default = 0.1, type = float,
                        help = 'the start value to search for epsilon (default: 0.1)')
    args = parser.parse_args()


    allow_gpu = args.cuda
    
    if torch.cuda.is_available() and allow_gpu:
        device = torch.device('cuda:%s' % args.cuda_idx)
    else:
        device = torch.device('cpu')
    
    N = args.N  # number of samples to handle at a time.
    p = args.p  # p norm
    if p > 100:
        p = float('inf')

    eps0 = args.eps0
    input_size = int(28*28 / args.time_step)
    hidden_size = args.hidden_size
    output_size = 10
    time_step = args.time_step
    activation = args.activation
    work_dir = args.work_dir
    model_name = args.model_name
    base_dir = "C:/Users/leolin9/POPQORN/POPQORN/"
    model_file = base_dir + work_dir + model_name
    # model_file = os.path.join(work_dir, args.model_name)
    save_dir = base_dir + work_dir + '%s_norm_bound/' % str(p)
    
    #load model
    rnn = RNN(input_size, hidden_size, output_size, time_step, activation)
    print(f"Attempting to load model from: {model_file}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Does file exist? {os.path.exists(model_file)}")
    rnn.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
    rnn.to(device)
    
    
    X,y,target_label = sample_mnist_data(N, time_step, device, num_labels=10,
                    data_dir='../data/mnist', train=False, shuffle=True, 
                    rnn=rnn, x=None, y=None)
    
        
    rnn.extractWeight(clear_original_model=False)
    
    l_eps, u_eps = rnn.getMaximumEps(p=p, true_label=y, 
                    target_label=target_label, eps0=eps0,
                      max_iter=10, X=X, 
                      acc=1e-3, gx0_trick = True, Eps_idx = None) # reduce to 10 to save time
    verifyMaximumEps(rnn, X, l_eps, p, y, target_label, 
                        eps_idx = None, untargeted=False, thred=1e-8)

    os.makedirs(save_dir, exist_ok=True)
    torch.save({'l_eps':l_eps, 'u_eps':u_eps, 'X':X, 'true_label':y, 
                'target_label':target_label}, save_dir+'certified_bound')
    print('Have saved the complete result to' + save_dir+'certified_bound')
    print('statistics of l_eps:')
    print('(min, mean, max, std) = (%.4f, %.4f, %.4f, %.4f) ' % (l_eps.min(), l_eps.mean(), l_eps.max(), l_eps.std()))