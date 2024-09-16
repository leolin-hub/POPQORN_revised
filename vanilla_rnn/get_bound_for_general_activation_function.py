#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 09:11:15 2018

@author: root
"""

import torch
import matplotlib.pyplot as plt

def d_tanh(x):
    #the derivative of tanh
    return 1- (torch.tanh(x))**2

def d_atan(x):
    return 1/(1+x**2)

def d_sigmoid(x):
    sx = torch.sigmoid(x)
    return sx*(1-sx)

Activation = {'tanh':[torch.tanh, d_tanh],
              'atan':[torch.atan, d_atan],
              'sigmoid':[torch.sigmoid, d_sigmoid],
              'ba':[torch.sign, 0],
              'relu':[torch.relu, 0],
              'relu_adaptive':[torch.relu, 0]}

def get_bound_for_relu(l, u, adaptive=False):
    '''
    Parameters:
        l: lower bound of the input
        u: upper bound of the input
        adaptive: whether to use adaptive bounds
    Returns:
        kl, bl, ku, bu: lower and upper bounds of the slope (kl and ku) and bias (bl and bu)
    '''
    device = l.device
    ku = torch.zeros(u.shape, device = device)
    bu = torch.zeros(u.shape, device = device)
    kl = torch.zeros(l.shape, device = device)
    bl = torch.zeros(l.shape, device = device)

    idx = l>=0
    kl[idx] = 1
    ku[idx] = 1

    idx = (l<0) * (u>0)

    k = (u / (u-l))[idx]
    # k u + b = u -> b = (1-k) * u
    b = (1-k) * u[idx]

    
    ku[idx] = k
    bu[idx] = b

    if not adaptive:
        kl[idx] = k
    else:
        idx = (l<0) * (u>0) * (u.abs()>=l.abs())
        kl[idx] = 1
        idx = (l<0) * (u>0) * (u.abs()<l.abs())
        kl[idx] = 0
    return kl, bl, ku, bu

def getConvenientGeneralActivationBound(l,u, activation, use_constant=False, epsilon=1e-6):
    """
    Get convenient bounds for general activation functions.
    Parameters:
        l (torch.Tensor): Lower bounds for the activation function.
        u (torch.Tensor): Upper bounds for the activation function.
        activation (str): The type of activation function to use (e.g., 'relu', 'relu_adaptive', 'ba').
        use_constant (bool, optional): If True, uses a constant function for the activation. Defaults to False.
        epsilon (float, optional): A small value to avoid numerical issues. Defaults to 1e-6.
    Returns:
        tuple: A tuple containing:
            - kl (torch.Tensor): Lower bound for the activation function.
            - bl (torch.Tensor): Lower bound for the output of the activation function.
            - ku (torch.Tensor): Upper bound for the activation function.
            - bu (torch.Tensor): Upper bound for the output of the activation function.
    Raises:
        Exception: If any element in l is greater than the corresponding element in u.
    """

    if (l > u + epsilon).any():
        print(f"Warning: l > u detected. Max difference: {(l - u).max().item()}")
        # 修正 l 和 u
        l = torch.min(l, u)
        u = torch.max(l, u)
        
    if (l>u).sum()>0:
        raise Exception('l must be less or equal to u')
        # print(l-u, (l-u).max())
        # if (l-u).max()>1e-4:
        #     raise Exception('l must be less or equal to u')
        # temp = l>u
        # l[temp] = l[temp] - 1e-4
        # u[temp] = u[temp] + 1e-4
    device = l.device
    
    ku = torch.zeros(u.shape, device = device)
    bu = torch.zeros(u.shape, device = device)
    kl = torch.zeros(l.shape, device = device)
    bl = torch.zeros(l.shape, device = device)
    if use_constant:
        #we have assume that the activaiton is monotomic
        function = Activation[activation][0]
        bu = function(u)
        bl = function(l)
        return kl, bl, ku, bu
    if activation == 'relu':
        kl, bl, ku, bu = get_bound_for_relu(l, u, adaptive=False)
        return kl, bl, ku, bu
    if activation == 'relu_adaptive':
        kl, bl, ku, bu = get_bound_for_relu(l, u, adaptive=True)
        return kl, bl, ku, bu
    if activation == 'ba':
        # print(u)
        print('binary activation')
        bu = torch.sign(u)
        bl = torch.sign(l)
        idx = (l<0) * (u>0) * (u.abs() > l.abs())
        kl[idx] = 2/u[idx]

        idx = (l<0) * (u>0) * (u.abs() < l.abs())
        ku[idx] = -2/l[idx]

        # idx = (l>0) * (l>0.8*u)
        # ku[idx] = 1/l[idx]
        # #ku l + bu = 1
        # bu[idx] = 1-ku[idx] * l[idx]
        print('uncertain neurons', ((l<0) * (u>0)).float().mean())
        return kl, bl, ku, bu
    
    idx = (l==u)
    if idx.any()>0: # 有更動 sum -> any
        bu[idx] = l[idx]
        bl[idx] = l[idx]
        
        ku[idx] = 1e-4
        kl[idx] = 1e-4
    
    #valid = (1-idx.int())
    valid = ~idx
    
    if valid.any()>0: # 有更動 sum -> any
        func = Activation[activation][0]
        dfunc = Activation[activation][1]
        kl_temp, bl_temp, ku_temp, bu_temp = getGeneralActivationBound(
                l[valid],u[valid], func, dfunc)
        kl[valid] = kl_temp
        ku[valid] = ku_temp
        bl[valid] = bl_temp
        bu[valid] = bu_temp
    # if (kl==0).sum()>0 or (ku==0).sum()>0:
    #     print(kl,ku)
    #     raise Exception('some elements of kl or ku are 0')
    idx2 = (kl==0)
    if idx2.sum()>0:
        kl[idx2] = 1e-8
    idx3 = (ku==0)
    if idx3.sum()>0:
        ku[idx3] = 1e-8
    return kl, bl, ku, bu

def getGeneralActivationBound(l,u, func, dfunc):
    """
    Calculates the bounds for a general activation function over a given range.
    Parameters:
        l (torch.Tensor): Lower bounds tensor of any shape. Must have the same shape as `u`.
        u (torch.Tensor): Upper bounds tensor of any shape. Must have the same shape as `l`.
        func (callable): The activation function to evaluate.
        dfunc (callable): The derivative of the activation function.
    Returns:
        tuple: A tuple containing four tensors:
            - kl (torch.Tensor): Lower bound slopes for the activation function.
            - bl (torch.Tensor): Lower bound intercepts for the activation function.
            - ku (torch.Tensor): Upper bound slopes for the activation function.
            - bu (torch.Tensor): Upper bound intercepts for the activation function.
    Notes:
        - The first dimension of `l` and `u` is the batch dimension.
        - Ensure that `u` is greater than `l` element-wise.
        - The function handles different cases based on the values of `l` and `u`.
    """

    #l and u are tensors of any shape. l and u must have the same shape
    #the first dimension of l and u is the batch dimension
    #users must make sure that u > l
    
    #initialize the desired variables
    device = l.device
    ku = torch.zeros(u.shape, device = device)
    bu = torch.zeros(u.shape, device = device)
    kl = torch.zeros(l.shape, device = device)
    bl = torch.zeros(l.shape, device = device)
    
    yl = func(l)
    yu = func(u)
    k = (yu - yl) / (u-l)
    b = yl - k * l
    d = (u+l) / 2
    
    func_d = func(d)
    d_func_d = dfunc(d) #derivative of tanh at x=d
    
    #l and u both <=0
    minus = (u <= 0) * (l<=0)
    ku[minus] = k[minus]
    bu[minus] = b[minus]
    kl[minus] = d_func_d[minus]
    bl[minus] = func_d[minus] - kl[minus] * d[minus]
    
    #l and u both >=0
    plus = (l >= 0)
    kl[plus] = k[plus]
    bl[plus] = b[plus]
    ku[plus] = d_func_d[plus]
    bu[plus] = func_d[plus] - ku[plus] * d[plus]
    
    #l < 0 and u>0
    pn = (l < 0) * (u > 0)
    kl[pn], bl[pn] = general_lb_pn(l[pn], u[pn], func, dfunc)
    ku[pn], bu[pn] = general_ub_pn(l[pn], u[pn], func, dfunc)
    
    return kl, bl, ku, bu

def getTanhBound(l,u):
    """
    Calculates the bounds for the hyperbolic tangent (tanh) activation function 
    given lower and upper tensor limits.
    Args:
        l (torch.Tensor): Lower bounds tensor of shape (batch_size, ...).
        u (torch.Tensor): Upper bounds tensor of shape (batch_size, ...).
                      Must have the same shape as l and satisfy u > l.
    Returns:
        tuple: A tuple containing four tensors:
            - kl (torch.Tensor): Lower bound slopes tensor of shape (batch_size, ...).
            - bl (torch.Tensor): Lower bound intercepts tensor of shape (batch_size, ...).
            - ku (torch.Tensor): Upper bound slopes tensor of shape (batch_size, ...).
            - bu (torch.Tensor): Upper bound intercepts tensor of shape (batch_size, ...).
    Notes:
        - The first dimension of l and u is the batch dimension.
        - Ensure that the input tensors l and u are compatible for the operations performed.
    """
    
    #l and u are tensors of any shape. l and u must have the same shape
    #the first dimension of l and u is the batch dimension
    #users must make sure that u > l
    
    #initialize the desired variables
    device = l.device
    ku = torch.zeros(u.shape, device = device)
    bu = torch.zeros(u.shape, device = device)
    kl = torch.zeros(l.shape, device = device)
    bl = torch.zeros(l.shape, device = device)
    
    yl = torch.tanh(l)
    yu = torch.tanh(u)
    k = (yu - yl) / (u-l)
    b = yl - k * l
    d = (u+l) / 2
    tanh_d = torch.tanh(d)
    d_tanh_d = 1 - tanh_d**2 #derivative of tanh at x=d
    
    #l and u both <=0
    minus = (u <= 0) * (l<=0)
    ku[minus] = k[minus]
    bu[minus] = b[minus]
    kl[minus] = d_tanh_d[minus]
    bl[minus] = tanh_d[minus] - kl[minus] * d[minus]
    
    #l and u both >=0
    plus = (l >= 0)
    kl[plus] = k[plus]
    bl[plus] = b[plus]
    ku[plus] = d_tanh_d[plus]
    bu[plus] = tanh_d[plus] - ku[plus] * d[plus]
    
    #l < 0 and u>0
    pn = (l < 0) * (u > 0)
    kl[pn], bl[pn] = general_lb_pn(l[pn], u[pn], torch.tanh, d_tanh)
    ku[pn], bu[pn] = general_ub_pn(l[pn], u[pn], torch.tanh, d_tanh)
    
    return kl, bl, ku, bu

def testGetGeneralActivationBound():
    u = torch.ones(1) * (5)
    l = torch.ones(1) * (-4)
#    kl, bl, ku, bu = getTanhBound(l,u)
    activation = 'relu'
    
    func = Activation[activation][0]#torch.atan#torch.tanh##
#    dfunc = Activation[activation][1]
#    kl, bl, ku, bu = getGeneralActivationBound(l,u, func, dfunc)
    kl, bl, ku, bu = getConvenientGeneralActivationBound(l,u, activation,
                    use_constant=True)
#    kl, bl, ku, bu = getGeneralActivationBound(l,u, torch.atan, d_atan)
    x = torch.rand(1000) * (u-l) + l
    
    func_x = func(x)
    l_func_x = kl * x + bl
    u_func_x = ku * x + bu
    
    plt.plot(x.numpy(), func_x.numpy(), '.')
    plt.plot(x.numpy(), l_func_x.numpy(), '.')
    plt.plot(x.numpy(), u_func_x.numpy(), '.')
    
    # print((l_func_x <= func_x).min())
    # print(x[l_func_x > func_x], l_func_x[l_func_x > func_x], func_x[l_func_x > func_x])
    # print((u_func_x >= func_x).min())
    # print(x[u_func_x < func_x], u_func_x[u_func_x < func_x], func_x[u_func_x < func_x])
    print((l_func_x <= func_x).all()) # tensor(True)
    print((u_func_x >= func_x).all())
    plt.show()
    
    

def get_d_UB(l,u,func,dfunc):
    """
    Calculates an upper bound for a given activation function using a numerical method.
    Parameters:
        l (torch.Tensor): Lower bound tensor of any shape. The first dimension represents the batch size.
        u (torch.Tensor): Upper bound tensor of the same shape as l. 
        func (callable): The activation function for which the upper bound is being calculated.
        dfunc (callable): The derivative of the activation function.
    Returns:
        torch.Tensor: A tensor representing the upper bound for the activation function, calculated based on the provided lower and upper bounds.
    Notes:
        - The function assumes that the activation function is symmetric around zero (i.e., f(x) = f(-x)).
        - The function also assumes that the activation function is convex for x < 0 and concave for x > 0.
        - The maximum number of iterations for the search is set to 1000.
    """

    #l and u are tensor of any shape. Their shape should be the same
    #the first dimension of l and u is batch_dimension
    diff = lambda d,l: (func(d)-func(l))/(d-l) - dfunc(d)
    max_iter = 1000
    # d = u/2
    # d = u
    ub = -l
    d = ub/2
    #use -l as the upper bound as d, it requires f(x) = f(-x)
    #and f to be convex when x<0 and concave when x>0
    
    #originally they use u as the upper bound, it may not always work  
    device = l.device
    lb = torch.zeros(l.shape, device=device)
    keep_search = torch.ones(l.shape, device=device, dtype=torch.bool)

    for i in range(max_iter):
        t = diff(d[keep_search], l[keep_search])
        #idx = (t<0) + (t.abs() > 0.01)
        idx = (t < 0) | (t.abs() > 0.01)
        t = t[idx]
        new_keep_search = torch.zeros_like(keep_search)
        new_keep_search[keep_search] = idx
        keep_search = new_keep_search
        if not keep_search.any(): # keep_search.sum() == 0 to 
            break
        
       
        # idx_pos = t>0
        # keep_search_copy = keep_search.clone()
        # keep_search_copy[keep_search] = idx_pos
        # ub[keep_search_copy] = d[keep_search_copy]
        # d[keep_search_copy] = (d[keep_search_copy] + lb[keep_search_copy]) / 2
      
        # idx_neg = t<0
        # keep_search_copy = keep_search.clone()
        # keep_search_copy[keep_search] = idx_neg
        # lb[keep_search_copy] = d[keep_search_copy]
        # d[keep_search_copy] = (d[keep_search_copy] + ub[keep_search_copy]) / 2
        idx_pos = t > 0
        ub[keep_search] = torch.where(idx_pos, d[keep_search], ub[keep_search])
        
        idx_neg = t < 0
        lb[keep_search] = torch.where(idx_neg, d[keep_search], lb[keep_search])
        
        d[keep_search] = (lb[keep_search] + ub[keep_search]) / 2
        
    # print('Use %d iterations' % i) 
    # print(diff(d,l))
    # print('d:', d)
    return d

def general_ub_pn(l, u, func, dfunc):
    """
    Calculate the upper bound parameters for a general activation function.
    Parameters:
        l (float): The lower bound input value.
        u (float): The upper bound input value.
        func (callable): The activation function to evaluate.
        dfunc (callable): The derivative of the activation function.
    Returns:
        tuple: A tuple containing:
            - k (float): The slope of the line connecting the function values at d_UB and l.
            - b (float): The y-intercept of the line at the point l.
    This function computes the upper bound parameters k and b based on the provided 
    activation function and its derivative, which can be used for further analysis 
    or optimization tasks.
    """

    d_UB = get_d_UB(l,u,func,dfunc)
    # print(d_UB)
    k = (func(d_UB)-func(l))/(d_UB-l)
    b  = func(l) - (l - 0.01) * k
    return k, b

def get_d_LB(l,u,func,dfunc):
    """
    Calculates the lower bound for a general activation function using a numerical method.
    Args:
        l (torch.Tensor): Lower bound tensor of any shape. The first dimension should represent the batch size.
        u (torch.Tensor): Upper bound tensor of the same shape as l. The first dimension should represent the batch size.
        func (callable): The activation function for which the lower bound is being calculated.
        dfunc (callable): The derivative of the activation function.
    Returns:
        d (torch.Tensor): A tensor containing the calculated lower bounds for the activation function.
    Notes:
        - The function assumes that the activation function is symmetric (f(x) = f(-x)) and convex for x < 0, 
          and concave for x > 0.
        - The function iteratively refines the bounds until convergence or until the maximum number of iterations is reached.
    """

    #l and u are tensor of any shape. Their shape should be the same
    #the first dimension of l and u is batch_dimension
    diff = lambda d,u: (func(d)-func(u))/(d-u) - dfunc(d)
    max_iter = 1000
    # d = u/2
    # d = u
    device = l.device
    ub = torch.zeros(l.shape, device=device)
    lb = -u
    d = lb/2
    #use -l as the upper bound as d, it requires f(x) = f(-x)
    #and f to be convex when x<0 and concave when x>0
    
    #originally they use u as the upper bound, it may not always work  

    keep_search = torch.ones(l.shape, device=device, dtype=torch.bool)
    for i in range(max_iter):
        t = diff(d[keep_search], u[keep_search])
        idx = (t<0) | (t.abs() > 0.01)
        t = t[idx]
        new_keep_search = torch.zeros_like(keep_search)
        new_keep_search[keep_search] = idx
        keep_search = new_keep_search
        if not keep_search.any():
            break
        
       
        # idx_pos = t>0
        # keep_search_copy = keep_search.clone()
        # keep_search_copy[keep_search] = idx_pos
        # lb[keep_search_copy] = d[keep_search_copy]
        # d[keep_search_copy] = (d[keep_search_copy] + ub[keep_search_copy]) / 2
      
        # idx_neg = t<0
        # keep_search_copy = keep_search.data.clone()
        # keep_search_copy[keep_search] = idx_neg
        # ub[keep_search_copy] = d[keep_search_copy]
        # d[keep_search_copy] = (d[keep_search_copy] + lb[keep_search_copy]) / 2
        idx_pos = t > 0
        lb[keep_search] = torch.where(idx_pos, d[keep_search], lb[keep_search])
        
        idx_neg = t < 0
        ub[keep_search] = torch.where(idx_neg, d[keep_search], ub[keep_search])
        
        d[keep_search] = (lb[keep_search] + ub[keep_search]) / 2
        
    # print('Use %d iterations' % i) 
    # print(diff(d,l))
    # print('d:', d)
    return d

def general_lb_pn(l, u, func, dfunc):
    """
    Calculates the linear lower bound parameters (k, b) for a given activation function.
    Parameters:
        l (float): The lower bound input value.
        u (float): The upper bound input value.
        func (callable): The activation function for which the bounds are calculated.
                         It should accept a single float input and return a float output.
        dfunc (callable): The derivative of the activation function. It should accept
                          a single float input and return a float output.
    Returns:
        tuple: A tuple containing:
            - k (float): The slope of the linear lower bound.
            - b (float): The y-intercept of the linear lower bound.
    Shapes:
        l: scalar (1D)
        u: scalar (1D)
        func: function mapping R -> R
        dfunc: function mapping R -> R
        k: scalar (1D)
        b: scalar (1D)
    """

    d_LB = get_d_LB(l,u,func,dfunc)
    # print(d_LB)
    k = (func(d_LB)-func(u))/(d_LB-u)
    b  = func(u) - (u + 0.01) * k
    return k, b

def test_general_b_pn():
    u = torch.ones(1) * 1
    l = torch.ones(1) * (-1)
    
    # d = get_d_LB(l,u,torch.tanh,d_tanh)
    ku,bu = general_ub_pn(l, u, torch.tanh,d_tanh)
    kl,bl = general_lb_pn(l, u, torch.tanh,d_tanh)
    
    x = torch.rand(1000) * (u-l) * 1 + l
    y0 = torch.tanh(x)
    yu = ku * x + bu
    yl = kl * x + bl
    # print((yu>y0).min())
    # print((yl<y0).min())
    print((yu>y0).all()) # tensor(True)
    print((yl<y0).all())
    plt.plot(x.numpy(),y0.numpy(),'.')
    plt.plot(x.numpy(),yl.numpy(),'.')
    plt.plot(x.numpy(),yu.numpy(),'.')
    plt.show()
    
if __name__ == '__main__':
   testGetGeneralActivationBound()
   #test_general_b_pn()
