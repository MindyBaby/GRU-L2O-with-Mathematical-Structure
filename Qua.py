#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.nn as nn
from timeit import default_timer as timer

import warnings
warnings.filterwarnings("ignore")


# In[2]:


if torch.cuda.is_available():
    USE_CUDA = True  

print('USE_CUDA = {}'.format(USE_CUDA))


# ## 1.创建二次函数模拟数据

# In[3]:


num_samples = 250
num_features = 10
batchsize = 128

torch.manual_seed(56)
W = torch.randn(batchsize, num_samples, num_features) 
x_gt = torch.randn(batchsize, num_features)
y_gt = torch.matmul(W, x_gt.unsqueeze(-1)).squeeze()


# In[4]:


# 设定噪声的标准差
noise_std = 0.1
# 生成噪声
noise = torch.randn_like(y_gt) * noise_std
# 添加噪声到 Y 中
Y = y_gt + noise


# ## 3.定义二次函数优化问题（光滑凸函数情况）

# In[5]:


def f(W,Y,x):
    """quadratic function : f(\theta) = \|W\theta - y\|_2^2"""
    if USE_CUDA:
        W = W.cuda()
        Y = Y.cuda()
        x = x.cuda()
    loss = ((torch.matmul(W,x.unsqueeze(-1)).squeeze()-Y)**2).sum(dim=1).mean(dim=0)
    return loss


# ## 2.构造LSTM_BlackBox优化器----2016

# In[6]:


class LSTM_BlackBox_Optimizee_Model(nn.Module):   
    def __init__(self,input_size, output_size, hidden_size, num_stacks, batchsize, preprocess = True ,p = 10 ,output_scale = 1):
        super().__init__()
        self.preprocess_flag = preprocess
        self.p = p
        self.input_flag = 2
        if preprocess != True:
             self.input_flag = 1
        self.output_scale = output_scale #论文
        use_bias = True
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_stacks = num_stacks
        self.batchsize = batchsize
        
        self.lstm = nn.LSTM(self.input_size * self.input_flag, self.hidden_size, self.num_stacks)       
        self.Linear = nn.Linear(self.hidden_size, self.output_size, bias=use_bias) #1-> output_size
    
    def LogAndSign_Preprocess_Gradient(self, gradients):
        p  = self.p
        log = torch.log(torch.abs(gradients))
        clamp_log = torch.clamp(log/p , min = -1.0,max = 1.0)
        clamp_sign = torch.clamp(torch.exp(torch.Tensor(p))*gradients, min = -1.0, max =1.0)
        return torch.cat((clamp_log,clamp_sign),dim = -1) #在gradients的最后一维input_dims拼接
    
    def Output_Gradient_Increment_And_Update_LSTM_Hidden_State(self, input_gradients, prev_state, i):
        if prev_state is None: #init_state
            prev_state = (torch.zeros(self.num_stacks,self.batchsize,self.hidden_size),
                            torch.zeros(self.num_stacks,self.batchsize,self.hidden_size))
            if USE_CUDA :
                 prev_state = (torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda(),
                            torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda())
        
        lr = 0.05
        update , next_state = self.lstm(input_gradients, prev_state)
        update = self.Linear(update) * self.output_scale * lr #因为LSTM的输出是当前步的Hidden，需要变换到output的相同形状上 
        return update, next_state
        
    def forward(self, x, input_gradients, prev_state, i):
        if USE_CUDA:
            input_gradients = input_gradients.cuda()
        #LSTM的输入为梯度，pytorch要求torch.nn.lstm的输入为（1，batchsize,input_dim）
        #原gradient.size()=torch.size[5] ->[1,1,5]
        input_gradients = input_gradients.unsqueeze(0)
        
        if self.preprocess_flag == True:
            input_gradients = self.LogAndSign_Preprocess_Gradient(input_gradients)
      
        update , next_state = self.Output_Gradient_Increment_And_Update_LSTM_Hidden_State(input_gradients , prev_state, i)
        # Squeeze to make it a single batch again.[1,1,5]->[5]
        update = update.squeeze().squeeze()
        update.retain_grad() 
        #x = x + update
        x = torch.add(x, update)
        return x , next_state


# In[7]:


Layers = 2
Hidden_nums = 20
Input_DIM = num_features
Output_DIM = num_features
output_scale_value=1


# In[8]:


LSTM_BlackBox_Optimizee = LSTM_BlackBox_Optimizee_Model( Input_DIM, Output_DIM, Hidden_nums ,Layers , batchsize=batchsize,
                preprocess=False,output_scale=output_scale_value)   ###### preprocess=False

print(LSTM_BlackBox_Optimizee)

if USE_CUDA:
    LSTM_BlackBox_Optimizee = LSTM_BlackBox_Optimizee.cuda()


# ## 2.构造Lr_LSTM_BlackBox优化器

# In[9]:


class LSTM_BlackBox_Optimizee_Model_lr(nn.Module):   
    def __init__(self,input_size, output_size, hidden_size, num_stacks, batchsize, preprocess = True ,p = 10 ,output_scale = 1):
        super().__init__()
        self.preprocess_flag = preprocess
        self.p = p
        self.input_flag = 2
        if preprocess != True:
             self.input_flag = 1
        self.output_scale = output_scale #论文
        use_bias = True
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_stacks = num_stacks
        self.batchsize = batchsize
        
        self.lstm = nn.LSTM(self.input_size * self.input_flag, self.hidden_size, self.num_stacks)       
        self.Linear = nn.Linear(self.hidden_size, self.output_size, bias=use_bias) #1-> output_size
    
    def LogAndSign_Preprocess_Gradient(self, gradients):
        p  = self.p
        log = torch.log(torch.abs(gradients))
        clamp_log = torch.clamp(log/p , min = -1.0,max = 1.0)
        clamp_sign = torch.clamp(torch.exp(torch.Tensor(p))*gradients, min = -1.0, max =1.0)
        return torch.cat((clamp_log,clamp_sign),dim = -1) #在gradients的最后一维input_dims拼接
    
    def learn_rate(self, initial_lr, gamma, step, min_lr=1e-6):
        lr = initial_lr * gamma ** step
        return lr
    
    def Output_Gradient_Increment_And_Update_LSTM_Hidden_State(self, input_gradients, prev_state, i):
        if prev_state is None: #init_state
            prev_state = (torch.zeros(self.num_stacks,self.batchsize,self.hidden_size),
                            torch.zeros(self.num_stacks,self.batchsize,self.hidden_size))
            if USE_CUDA :
                 prev_state = (torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda(),
                            torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda())
        
        lr = self.learn_rate(initial_lr = 0.05, gamma = 0.96, step = i)
        update , next_state = self.lstm(input_gradients, prev_state)
        update = self.Linear(update) * self.output_scale * lr #因为LSTM的输出是当前步的Hidden，需要变换到output的相同形状上 
        return update, next_state
        
    def forward(self, x, input_gradients, prev_state, i):
        if USE_CUDA:
            input_gradients = input_gradients.cuda()
        #LSTM的输入为梯度，pytorch要求torch.nn.lstm的输入为（1，batchsize,input_dim）
        #原gradient.size()=torch.size[5] ->[1,1,5]
        input_gradients = input_gradients.unsqueeze(0)
        
        if self.preprocess_flag == True:
            input_gradients = self.LogAndSign_Preprocess_Gradient(input_gradients)
      
        update , next_state = self.Output_Gradient_Increment_And_Update_LSTM_Hidden_State(input_gradients , prev_state, i)
        # Squeeze to make it a single batch again.[1,1,5]->[5]
        update = update.squeeze().squeeze()
        update.retain_grad() 
        #x = x + update
        x = torch.add(x, update)
        return x , next_state


# In[10]:


Layers = 2
Hidden_nums = 20
Input_DIM = num_features
Output_DIM = num_features
output_scale_value=1


# In[11]:


LSTM_BlackBox_Optimizee_lr = LSTM_BlackBox_Optimizee_Model_lr( Input_DIM, Output_DIM, Hidden_nums ,Layers , batchsize=batchsize,
                preprocess=False,output_scale=output_scale_value)   ###### preprocess=False

print(LSTM_BlackBox_Optimizee_lr)

if USE_CUDA:
    LSTM_BlackBox_Optimizee_lr = LSTM_BlackBox_Optimizee_lr.cuda()


# ## 3.构造LSTM_Math优化器---2023

# In[12]:


NORM_FUNC = {
    'exp': torch.exp,
    'eye': nn.Identity(),
    'sigmoid': lambda x: 2.0 * torch.sigmoid(x),
    'softplus': nn.Softplus(),
}


# In[13]:


class LSTM_Math_Optimizee_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_stacks,
                 batchsize, preprocess = True ,p = 10 ,output_scale = 1,
                 p_use=True, p_scale=1.0, p_scale_learned=True, p_norm='eye',
                 b_use=True, b_scale=1.0, b_scale_learned=True, b_norm='eye',
                 ):
        super().__init__()
        self.preprocess_flag = preprocess
        self.p = p
        self.input_flag = 2
        if preprocess != True:
             self.input_flag = 1
        self.output_scale = output_scale #论文
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_stacks = num_stacks
        self.batchsize = batchsize
        
        self.lstm = nn.LSTM(self.input_size * self.input_flag, self.hidden_size, self.num_stacks)       
        self.Linear = nn.Linear(self.hidden_size, self.output_size)
        
        # pre-conditioner
        self.linear_p = nn.Linear(self.hidden_size, self.output_size)
        # bias
        self.linear_b = nn.Linear(self.hidden_size, self.output_size)
        
        self.p_use = p_use
        if p_scale_learned:
            self.p_scale = nn.Parameter(torch.tensor(1.) * p_scale)
        else:
            self.p_scale = p_scale
        self.p_norm = NORM_FUNC[p_norm]
        
        self.b_use = b_use
        if b_scale_learned:
            self.b_scale = nn.Parameter(torch.tensor(1.) * b_scale)
        else:
            self.b_scale = b_scale
        self.b_norm = NORM_FUNC[b_norm]
        
    def LogAndSign_Preprocess_Gradient(self,gradients):
        p  = self.p
        log = torch.log(torch.abs(gradients))
        clamp_log = torch.clamp(log/p , min = -1.0,max = 1.0)
        clamp_sign = torch.clamp(torch.exp(torch.Tensor(p))*gradients, min = -1.0, max =1.0)
        return torch.cat((clamp_log,clamp_sign),dim = -1) #在gradients的最后一维input_dims拼接
    
    def Output_Gradient_Increment_And_Update_LSTM_Hidden_State(self, x, input_gradients, prev_state, i):
        if prev_state is None: #init_state
            prev_state = (torch.zeros(self.num_stacks,self.batchsize,self.hidden_size),
                            torch.zeros(self.num_stacks,self.batchsize,self.hidden_size))
            if USE_CUDA :
                 prev_state = (torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda(),
                            torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda())
        
        output , next_state = self.lstm(input_gradients, prev_state)

        P = self.linear_p(output)* self.output_scale
        B = self.linear_b(output)* self.output_scale
 
        P = self.p_norm(P) * self.p_scale if self.p_use else 1.0
        B = self.b_norm(B) * self.b_scale if self.b_use else 1.0        

        if USE_CUDA:
            input_gradients = input_gradients.cuda()
            P = P.cuda()     
            B = B.cuda() 

        lr = 0.005
        update = x - P * input_gradients* lr - B 
        return update, next_state
    
    def forward(self, x, input_gradients, prev_state, i):
        if USE_CUDA:
            input_gradients = input_gradients.cuda()
        #LSTM的输入为梯度，pytorch要求torch.nn.lstm的输入为（1，batchsize,input_dim）
        #原gradient.size()=torch.size[5] ->[1,1,5]
        input_gradients = input_gradients.unsqueeze(0)
        
        if self.preprocess_flag == True:
            input_gradients = self.LogAndSign_Preprocess_Gradient(input_gradients)
            
        update , next_state = self.Output_Gradient_Increment_And_Update_LSTM_Hidden_State(x, input_gradients, prev_state, i)
        #out , next_state = self.Output_Gradient_Increment_And_Update_LSTM_Hidden_State(x, gradients , prev_state)
        # Squeeze to make it a single batch again.[1,1,5]->[5]
        update = update.squeeze().squeeze()
        update.retain_grad() 
        #x = x + update
        x = update
        return x , next_state


# In[14]:


'''class LSTM_Math_Optimizee_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_stacks,
                 batchsize, preprocess = True ,p = 10 ,output_scale = 1,
                 p_use=True, p_scale=1.0, p_scale_learned=True, p_norm='eye',
                 b_use=True, b_scale=1.0, b_scale_learned=True, b_norm='eye',
                 a_use=True, a_scale=1.0, a_scale_learned=True, a_norm='eye',
                 b1_use=True, b1_scale=1.0, b1_scale_learned=True, b1_norm='eye',
                 b2_use=True, b2_scale=1.0, b2_scale_learned=True, b2_norm='eye',
                 ):
        super().__init__()
        self.preprocess_flag = preprocess
        self.p = p
        self.input_flag = 2
        if preprocess != True:
             self.input_flag = 1
        self.output_scale = output_scale #论文
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_stacks = num_stacks
        self.batchsize = batchsize
        
        self.lstm = nn.LSTM(self.input_size * self.input_flag, self.hidden_size, self.num_stacks)       
        self.Linear = nn.Linear(self.hidden_size, self.output_size)
        
        # pre-conditioner
        self.linear_p = nn.Linear(self.hidden_size, self.output_size)
        # bias
        self.linear_b = nn.Linear(self.hidden_size, self.output_size)
        self.linear_b1 = nn.Linear(self.hidden_size, self.output_size)
        self.linear_b2 = nn.Linear(self.hidden_size, self.output_size)
        # momentum
        self.linear_a = nn.Linear(self.hidden_size, self.output_size)
        
        self.p_use = p_use
        if p_scale_learned:
            self.p_scale = nn.Parameter(torch.tensor(1.) * p_scale)
        else:
            self.p_scale = p_scale
        self.p_norm = NORM_FUNC[p_norm]
        
        self.b_use = b_use
        if b_scale_learned:
            self.b_scale = nn.Parameter(torch.tensor(1.) * b_scale)
        else:
            self.b_scale = b_scale
        self.b_norm = NORM_FUNC[b_norm]
        
        self.b1_use = b1_use
        if b1_scale_learned:
            self.b1_scale = nn.Parameter(torch.tensor(1.) * b1_scale)
        else:
            self.b1_scale = b1_scale
        self.b1_norm = NORM_FUNC[b1_norm]
        
        self.b2_use = b2_use
        if b2_scale_learned:
            self.b2_scale = nn.Parameter(torch.tensor(1.) * b2_scale)
        else:
            self.b2_scale = b2_scale
        self.b2_norm = NORM_FUNC[b2_norm]
        
        self.a_use = a_use
        if a_scale_learned:
            self.a_scale = nn.Parameter(torch.tensor(1.) * a_scale)
        else:
            self.a_scale = a_scale
        self.a_norm = NORM_FUNC[a_norm]

        
    def LogAndSign_Preprocess_Gradient(self,gradients):
        p  = self.p
        log = torch.log(torch.abs(gradients))
        clamp_log = torch.clamp(log/p , min = -1.0,max = 1.0)
        clamp_sign = torch.clamp(torch.exp(torch.Tensor(p))*gradients, min = -1.0, max =1.0)
        return torch.cat((clamp_log,clamp_sign),dim = -1) #在gradients的最后一维input_dims拼接
    

    def Output_Gradient_Increment_And_Update_LSTM_Hidden_State(self, x, input_gradients, prev_state, i):
        if prev_state is None: #init_state
            prev_state = (torch.zeros(self.num_stacks,self.batchsize,self.hidden_size),
                            torch.zeros(self.num_stacks,self.batchsize,self.hidden_size))
            if USE_CUDA :
                 prev_state = (torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda(),
                            torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda())
        
        output , next_state = self.lstm(input_gradients, prev_state)
        
        P = self.linear_p(output)* self.output_scale
        B = self.linear_b(output)* self.output_scale
        A = self.linear_a(output)* self.output_scale
        B1 = self.linear_b1(output)* self.output_scale
        B2 = self.linear_b2(output)* self.output_scale 
        
        P = self.p_norm(P) * self.p_scale if self.p_use else 1.0
        B = self.b_norm(B) * self.b_scale if self.b_use else 1.0
        A = self.a_norm(A) * self.a_scale if self.a_use else 0.0
        B1 = self.b_norm(B1) * self.b1_scale if self.b1_use else 0.0
        B2 = self.b_norm(B2) * self.b2_scale if self.b2_use else 0.0
        
        lr = 0.005
        if USE_CUDA:
            input_gradients = input_gradients.cuda()
            P = P.cuda()        
        updateX = x - P * input_gradients * lr
            
        Z_input = LSTM_Math_learner.get_var('Z') 
        if Z_input.grad is not None:
            Z_grad = torch.clone(Z_input.grad).detach()
        else:
            Z_grad = torch.zeros_like(Z_input)

        if USE_CUDA:
            Z_grad = Z_grad.cuda()
            
        Z_input_clone = torch.clone(Z_input).detach()
        updateZ = Z_input_clone - P * Z_grad * lr
        
        prox_in = (1 - B) * updateX + B * updateZ - B1
        P_in = P * lr
        prox_out = LSTM_Math_learner.prox({'P':P_in , 'X':prox_in}, compute_grad = True)   
        prox_diff = prox_out - x
        Z_new = prox_out + A * prox_diff  + B2
        update = prox_out
        
        LSTM_Math_learner.set_var('Z', Z_new)
        return update, next_state
    
    def forward(self, x, input_gradients, prev_state, i):
        if USE_CUDA:
            input_gradients = input_gradients.cuda()
        #LSTM的输入为梯度，pytorch要求torch.nn.lstm的输入为（1，batchsize,input_dim）
        #原gradient.size()=torch.size[5] ->[1,1,5]
        input_gradients = input_gradients.unsqueeze(0)
        
        if self.preprocess_flag == True:
            input_gradients = self.LogAndSign_Preprocess_Gradient(input_gradients)
            
        update , next_state = self.Output_Gradient_Increment_And_Update_LSTM_Hidden_State(x, input_gradients , prev_state, i)
        #out , next_state = self.Output_Gradient_Increment_And_Update_LSTM_Hidden_State(x, gradients , prev_state)
        # Squeeze to make it a single batch again.[1,1,5]->[5]
        update = update.squeeze().squeeze()
       
        #x = x + update
        #x = torch.add(x, update)
        x = update
        return x , next_state'''


# In[15]:


Layers = 2
Hidden_nums = 20
Input_DIM = num_features
Output_DIM = num_features
output_scale_value=1


# In[16]:


LSTM_Math_Optimizee = LSTM_Math_Optimizee_Model( Input_DIM, Output_DIM, Hidden_nums ,Layers , batchsize=batchsize,
                preprocess=False,output_scale=output_scale_value)   ###### preprocess=False

print(LSTM_Math_Optimizee)

if USE_CUDA:
    LSTM_Math_Optimizee = LSTM_Math_Optimizee.cuda()


# ## 3.构造GRU_Math优化器

# In[17]:


NORM_FUNC = {
    'exp': torch.exp,
    'eye': nn.Identity(),
    'sigmoid': lambda x: 2.0 * torch.sigmoid(x),
    'softplus': nn.Softplus(),
}


# In[18]:


class GRU_Math_Optimizee_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_stacks,
                 batchsize, preprocess = True ,p = 10 ,output_scale = 1,
                 p_use=True, p_scale=1.0, p_scale_learned=True, p_norm='eye',
                 b_use=True, b_scale=1.0, b_scale_learned=True, b_norm='eye',
                 ):
        super().__init__()
        self.preprocess_flag = preprocess
        self.p = p
        self.input_flag = 2
        if preprocess != True:
             self.input_flag = 1
        self.output_scale = output_scale #论文
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_stacks = num_stacks
        self.batchsize = batchsize
        
        self.gru = nn.GRU(self.input_size * self.input_flag, self.hidden_size, self.num_stacks)       
        self.Linear = nn.Linear(self.hidden_size, self.output_size)
        
        # pre-conditioner
        self.linear_p = nn.Linear(self.hidden_size, self.output_size)
        # bias
        self.linear_b = nn.Linear(self.hidden_size, self.output_size)
        
        self.p_use = p_use
        if p_scale_learned:
            self.p_scale = nn.Parameter(torch.tensor(1.) * p_scale)
        else:
            self.p_scale = p_scale
        self.p_norm = NORM_FUNC[p_norm]
        
        self.b_use = b_use
        if b_scale_learned:
            self.b_scale = nn.Parameter(torch.tensor(1.) * b_scale)
        else:
            self.b_scale = b_scale
        self.b_norm = NORM_FUNC[b_norm]

        
    def LogAndSign_Preprocess_Gradient(self,gradients):
        p  = self.p
        log = torch.log(torch.abs(gradients))
        clamp_log = torch.clamp(log/p , min = -1.0,max = 1.0)
        clamp_sign = torch.clamp(torch.exp(torch.Tensor(p))*gradients, min = -1.0, max =1.0)
        return torch.cat((clamp_log,clamp_sign),dim = -1) #在gradients的最后一维input_dims拼接
    
    def learn_rate(self, initial_lr, gamma, step, min_lr=1e-6):
        lr = initial_lr * gamma ** step
        return lr
    
    def Output_Gradient_Increment_And_Update_GRU_Hidden_State(self, x, input_gradients, prev_state, i):
        if prev_state is None: #init_state
            prev_state = torch.zeros(self.num_stacks,self.batchsize,self.hidden_size)
            if USE_CUDA :
                 prev_state = torch.zeros(self.num_stacks,self.batchsize,self.hidden_size).cuda()
        
        output , next_state = self.gru(input_gradients, prev_state)

        P = self.linear_p(output)* self.output_scale
        B = self.linear_b(output)* self.output_scale
 
        P = self.p_norm(P) * self.p_scale if self.p_use else 1.0
        B = self.b_norm(B) * self.b_scale if self.b_use else 1.0        

        if USE_CUDA:
            input_gradients = input_gradients.cuda()
            P = P.cuda()     
            B = B.cuda() 
            
        lr = self.learn_rate(initial_lr = 0.01, gamma = 0.96, step = i)
        update = x - P * input_gradients* lr - B * lr
        return update, next_state
    
    def forward(self, x, input_gradients, prev_state, i):
        if USE_CUDA:
            input_gradients = input_gradients.cuda()
        #LSTM的输入为梯度，pytorch要求torch.nn.lstm的输入为（1，batchsize,input_dim）
        #原gradient.size()=torch.size[5] ->[1,1,5]
        input_gradients = input_gradients.unsqueeze(0)
        
        if self.preprocess_flag == True:
            input_gradients = self.LogAndSign_Preprocess_Gradient(input_gradients)
            
        update , next_state = self.Output_Gradient_Increment_And_Update_GRU_Hidden_State(x, input_gradients , prev_state, i)
        #out , next_state = self.Output_Gradient_Increment_And_Update_LSTM_Hidden_State(x, gradients , prev_state)
        # Squeeze to make it a single batch again.[1,1,5]->[5]
        update = update.squeeze().squeeze()
        update.retain_grad() 
        #x = x + update
        x = update
        return x , next_state


# In[19]:


Layers = 2
Hidden_nums = 20
Input_DIM = num_features
Output_DIM = num_features
output_scale_value=1


# In[20]:


GRU_Math_Optimizee = GRU_Math_Optimizee_Model(Input_DIM, Output_DIM, Hidden_nums ,Layers , batchsize=batchsize,
                preprocess=False,output_scale=output_scale_value)   ###### preprocess=False

print(GRU_Math_Optimizee)

if USE_CUDA:
    GRU_Math_Optimizee = GRU_Math_Optimizee.cuda()


# ## 7.优化问题目标函数的学习过程

# In[21]:


class Learner( object ):
    def __init__(self,    
                 f ,  
                 W,
                 Y,
                 optimizee,  
                 train_steps,  
                 eval_flag = False,
                 retain_graph_flag=False,
                 reset_theta = False ,
                 reset_function_from_IID_distirbution = True,
                 **options)-> None:
        
        self.f = f
        self.W = W
        self.Y = Y
        self.optimizee = optimizee
        self.train_steps = train_steps
        self.eval_flag = eval_flag
        self.retain_graph_flag = retain_graph_flag
        self.reset_theta = reset_theta
        self.reset_function_from_IID_distirbution = reset_function_from_IID_distirbution  
        self.state = None

        self.global_loss_graph = 0 #这个是为LSTM优化器求所有loss相加产生计算图准备的
        self.losses = []   # 保存每个训练周期的loss值
        self.x = torch.zeros(batchsize, num_features, requires_grad=True)
        self.sqr_grads = 0
        self.vars = dict()
        self.initialize()
                          
    def Reset_Or_Reuse(self , x , W , Y , state, num_roll):      
        if num_roll == 0:
            self.W = W 
            self.Y = Y    
            self.x = x
            state = None
            
        if USE_CUDA:
            W = W.cuda()
            Y = Y.cuda()
            x = x.cuda()
            x.retain_grad()                      
        return  x , W , Y , state    
    
    def initialize(self):
        prox_out = torch.zeros(batchsize, num_features, requires_grad=True)  
        if USE_CUDA:
            prox_out = prox_out.cuda()
        self.set_var('Z', prox_out) 

    def get_var(self, var_name):
        return self.vars[var_name]
           
    def set_var(self, var_name, var_value):
        self.vars[var_name] = var_value
        '''if var_name in self.vars:
            self.vars[var_name].append(var_value)
        else:
            self.vars[var_name] = [var_value]'''
        
    def prox(self, inputs: dict, compute_grad: bool = False, **kwargs):
        P = inputs['P']
        X = inputs['X']
        mag = nn.functional.relu(torch.abs(X) - P)
        return torch.sign(X) * mag
        
    def __call__(self, num_roll=0) :  #全局训练
        f  = self.f 
        x , W , Y , state =  self.Reset_Or_Reuse(self.x , self.W , self.Y , self.state , num_roll )
        sqr_grads = self.sqr_grads
        self.global_loss_graph = 0   #每个unroll的开始需要重新置零
        optimizee = self.optimizee
        print('state is None = {}'.format(state == None))
        
        if optimizee == LSTM_BlackBox_Optimizee:            
            for i in range(self.train_steps):    
                loss = f(W,Y,x)
                self.global_loss_graph += loss
                loss.backward(retain_graph = self.retain_graph_flag) # 默认为False,当优化LSTM设置为True
          
                grad = torch.clone(x.grad).detach()
                x, state = optimizee(x, grad, state, i)

                self.losses.append(loss)
                x.retain_grad()
            if state is not None:
                self.state = (state[0].detach(),state[1].detach())
            return self.losses ,self.global_loss_graph 
         
        if optimizee == LSTM_BlackBox_Optimizee_lr:            
            for i in range(self.train_steps):    
                loss = f(W,Y,x)
                self.global_loss_graph += loss
                loss.backward(retain_graph = self.retain_graph_flag) # 默认为False,当优化LSTM设置为True
          
                grad = torch.clone(x.grad).detach()
                x, state = optimizee(x, grad, state, i)

                self.losses.append(loss)
                x.retain_grad()
            if state is not None:
                self.state = (state[0].detach(),state[1].detach())
            return self.losses ,self.global_loss_graph
        
        if optimizee == LSTM_Math_Optimizee:            
            for i in range(self.train_steps):    
                loss = f(W,Y,x)
                self.global_loss_graph += loss
                loss.backward(retain_graph = self.retain_graph_flag) # 默认为False,当优化LSTM设置为True
          
                grad = torch.clone(x.grad).detach()
                x, state = optimizee(x, grad, state, i)

                self.losses.append(loss)
                x.retain_grad()
            if state is not None:
                self.state = (state[0].detach(),state[1].detach())
            return self.losses ,self.global_loss_graph 
        
        if optimizee == GRU_Math_Optimizee: 
            for i in range(self.train_steps):    
                loss = f(W,Y,x)
                self.global_loss_graph += loss
                loss.backward(retain_graph = self.retain_graph_flag) # 默认为False,当优化LSTM设置为True
                    
                grad = torch.clone(x.grad).detach()
                x, state = optimizee(x, grad, state, i)
          
                self.losses.append(loss)
                x.retain_grad()
            if state is not None:
                self.state = state.detach()
            return self.losses ,self.global_loss_graph


# In[22]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

STEPS = 100
x = np.arange(STEPS)


#for _ in range(1): 
for loop_count in range(1):  # 在这里设置循环次数
   
    LSTM_BlackBox_learner = Learner(f , W, Y, LSTM_BlackBox_Optimizee, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    LSTM_BlackBox_lr_learner = Learner(f , W, Y, LSTM_BlackBox_Optimizee_lr, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    LSTM_Math_learner = Learner(f , W, Y, LSTM_Math_Optimizee, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    GRU_Math_learner = Learner(f , W, Y, GRU_Math_Optimizee, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    
    lstm_blackbox_losses, lstm_blackbox_sum_loss = LSTM_BlackBox_learner()
    lstm_blackbox_lr_losses, lstm_blackbox_lr_sum_loss = LSTM_BlackBox_lr_learner()
    lstm_math_losses, lstm_math_sum_loss = LSTM_Math_learner()
    gru_math_losses, gru_math_sum_loss = GRU_Math_learner()

    lstm_blackbox_losses_tensor = torch.tensor(lstm_blackbox_losses)
    lstm_blackbox_lr_losses_tensor = torch.tensor(lstm_blackbox_lr_losses)
    lstm_math_losses_tensor = torch.tensor(lstm_math_losses)
    gru_math_losses_tensor = torch.tensor(gru_math_losses)
    
    p1, = plt.plot(x, lstm_blackbox_losses_tensor.numpy(), label='LSTM_BlackBox')
    p2, = plt.plot(x, lstm_blackbox_lr_losses_tensor.numpy(), label='LSTM_BlackBox_Lr')
    p3, = plt.plot(x, lstm_math_losses_tensor.numpy(), label='LSTM_Math')
    p4, = plt.plot(x, gru_math_losses_tensor.numpy(), label='GRU_Math') 

    
    plt.yscale('log')
    plt.legend(handles=[p1, p2, p3, p4])
    plt.title('Losses')
    plt.show()
    print("lstm_black={},lstm_black_lr={},lstm_math={}, gru_math={}".format(lstm_blackbox_sum_loss,lstm_blackbox_lr_sum_loss, lstm_math_sum_loss, 
                                                                                            gru_math_sum_loss  ))


# ## 9.自动学习的LSTM优化器Learning to learn

# In[23]:


from timeit import default_timer as timer
def Learning_to_learn_global_training(optimizee, global_taining_steps, Optimizee_Train_Steps, UnRoll_STEPS, 
                                      Evaluate_period ,optimizer_lr=0.1):
    global_loss_list = []
    Total_Num_Unroll = Optimizee_Train_Steps // UnRoll_STEPS
    adam_global_optimizer = torch.optim.Adam(optimizee.parameters(),lr = optimizer_lr)

    Optimizer_Learner = Learner(f, W, Y, optimizee, UnRoll_STEPS, retain_graph_flag=True, reset_theta=True,)

    best_sum_loss = 999999999
    best_final_loss = 999999999
    best_flag = False
    start = timer()
    for i in range(Global_Train_Steps): 
        print('global training steps(全局步数): {}'.format(i))
        
        total_time = timer()
        for num in range(Total_Num_Unroll):
            
            start = timer()
            _,global_loss = Optimizer_Learner(num)   

            adam_global_optimizer.zero_grad()
            global_loss.backward() 
       
            adam_global_optimizer.step()
            global_loss_list.append(global_loss.detach_())
            
            time = timer() - start
            print(f'Epoch [{(num +1)* UnRoll_STEPS}/{Optimizee_Train_Steps}], Time: {time:.2f}, Global_Loss: {global_loss:.4f}')

        if (i + 1) % Evaluate_period == 0:
            
            best_sum_loss, best_final_loss, best_flag  = evaluate(best_sum_loss, best_final_loss, best_flag, optimizer_lr)
        
        end_time = total_time/ 3600
        print('总时间：{:.2f}h'.format(end_time))
    return global_loss_list, best_flag


# ## 2.构造LSTM_BlackBox优化器----2016

# In[24]:


def evaluate(best_sum_loss, best_final_loss, best_flag, lr):
    print('evalute the model(评估模型)')
    STEPS = 100
    learner = Learner(f , W, Y, LSTM_BlackBox_Optimizee, STEPS, eval_flag=True,reset_theta=True, retain_graph_flag=True)
    losses, sum_loss = learner()
    try:
        best = torch.load('QuadraticFunction/LSTM_BlackBox_best_loss.txt')
    except IOError:
        print ('can not find LSTM_BlackBox_best_loss.txt')
        pass
    else:
        best_sum_loss = best[0]
        best_final_loss = best[1]
        print('当前损失[{}], 当前总损失[{}]'.format(losses[-1], sum_loss))
        
    if losses[-1] < best_final_loss and  sum_loss < best_sum_loss:
        best_final_loss = losses[-1]
        best_sum_loss =  sum_loss      
        print('最佳损失[{}], 最佳总损失[{}]'.format(best_final_loss, best_sum_loss))
        
        torch.save(LSTM_BlackBox_Optimizee.state_dict(),'QuadraticFunction/LSTM_BlackBox_best_optimizer.pth')
        torch.save([best_sum_loss ,best_final_loss,lr ],'QuadraticFunction/LSTM_BlackBox_best_loss.txt')
        best_flag = True
        
    return best_sum_loss, best_final_loss, best_flag


# In[25]:


Global_Train_Steps = 50 #可修改1000
Optimizee_Train_Steps = 100#######100
UnRoll_STEPS = 20
Evaluate_period = 1 #可修改
optimizer_lr = 0.1 #可修改
global_loss_list ,flag = Learning_to_learn_global_training( LSTM_BlackBox_Optimizee,
                                                            Global_Train_Steps,
                                                            Optimizee_Train_Steps,
                                                            UnRoll_STEPS,
                                                            Evaluate_period,
                                                            optimizer_lr)


# ## 2.构造Lr_LSTM_BlackBox优化器

# In[26]:


def evaluate(best_sum_loss, best_final_loss, best_flag, lr):
    print('evalute the model(评估模型)')
    STEPS = 100
    learner = Learner(f , W, Y, LSTM_BlackBox_Optimizee_lr, STEPS, eval_flag=True,reset_theta=True, retain_graph_flag=True)
    losses, sum_loss = learner()
    try:
        best = torch.load('QuadraticFunction/LSTM_BlackBox_best_loss_lr.txt')
    except IOError:
        print ('can not find LSTM_BlackBox_best_loss_lr.txt')
        pass
    else:
        best_sum_loss = best[0]
        best_final_loss = best[1]
        print('当前损失[{}], 当前总损失[{}]'.format(losses[-1], sum_loss))
        
    if losses[-1] < best_final_loss and  sum_loss < best_sum_loss:
        best_final_loss = losses[-1]
        best_sum_loss =  sum_loss      
        print('最佳损失[{}], 最佳总损失[{}]'.format(best_final_loss, best_sum_loss))
        
        torch.save(LSTM_BlackBox_Optimizee_lr.state_dict(),'QuadraticFunction/LSTM_BlackBox_best_optimizer_lr.pth')
        torch.save([best_sum_loss ,best_final_loss,lr ],'QuadraticFunction/LSTM_BlackBox_best_loss_lr.txt')
        best_flag = True
        
    return best_sum_loss, best_final_loss, best_flag


# In[27]:


Global_Train_Steps = 50 #可修改1000
Optimizee_Train_Steps = 100#######100
UnRoll_STEPS = 20
Evaluate_period = 1 #可修改
optimizer_lr = 0.1 #可修改
global_loss_list ,flag = Learning_to_learn_global_training( LSTM_BlackBox_Optimizee_lr,
                                                            Global_Train_Steps,
                                                            Optimizee_Train_Steps,
                                                            UnRoll_STEPS,
                                                            Evaluate_period,
                                                            optimizer_lr)


# ## 11.LSTM_Math

# In[28]:


def evaluate(best_sum_loss, best_final_loss, best_flag, lr):
    print('evalute the model(评估模型)')
    STEPS = 100
    learner = Learner(f , W, Y, LSTM_Math_Optimizee, STEPS, eval_flag=True,reset_theta=True, retain_graph_flag=True)
    losses, sum_loss = learner()
    try:
        best = torch.load('QuadraticFunction/LSTM_Math_best_loss.txt')
    except IOError:
        print ('can not find LSTM_Math_best_loss.txt')
        pass
    else:
        best_sum_loss = best[0]
        best_final_loss = best[1]
        print('当前损失[{}], 当前总损失[{}]'.format(losses[-1], sum_loss))
        
    if losses[-1] < best_final_loss and  sum_loss < best_sum_loss:
        best_final_loss = losses[-1]
        best_sum_loss =  sum_loss      
        print('最佳损失[{}], 最佳总损失[{}]'.format(best_final_loss, best_sum_loss))
        
        torch.save(LSTM_Math_Optimizee.state_dict(),'QuadraticFunction/LSTM_Math_best_optimizer.pth')
        torch.save([best_sum_loss ,best_final_loss,lr ],'QuadraticFunction/LSTM_Math_best_loss.txt')
        best_flag = True
        
    return best_sum_loss, best_final_loss, best_flag


# In[29]:


Global_Train_Steps = 50 #可修改1000
Optimizee_Train_Steps = 100#######100
UnRoll_STEPS = 20
Evaluate_period = 1 #可修改
optimizer_lr = 0.1 #可修改
global_loss_list ,flag = Learning_to_learn_global_training( LSTM_Math_Optimizee,
                                                            Global_Train_Steps,
                                                            Optimizee_Train_Steps,
                                                            UnRoll_STEPS,
                                                            Evaluate_period,
                                                            optimizer_lr)


# ## 12.GRU_Math

# In[30]:


def evaluate(best_sum_loss, best_final_loss, best_flag, lr):
    print('evalute the model(评估模型)')
    STEPS = 100
    learner = Learner(f , W, Y, GRU_Math_Optimizee, STEPS, eval_flag=True,reset_theta=True, retain_graph_flag=True)
    losses, sum_loss = learner()
    try:
        best = torch.load('QuadraticFunction/GRU_Math_best_loss.txt')
    except IOError:
        print ('can not find GRU_Math_best_loss.txt')
        pass
    else:
        best_sum_loss = best[0]
        best_final_loss = best[1]
        print('当前损失[{}], 当前总损失[{}]'.format(losses[-1], sum_loss))
        
    if losses[-1] < best_final_loss and  sum_loss < best_sum_loss:
        best_final_loss = losses[-1]
        best_sum_loss =  sum_loss      
        print('最佳损失[{}], 最佳总损失[{}]'.format(best_final_loss, best_sum_loss))
        
        torch.save(GRU_Math_Optimizee.state_dict(),'QuadraticFunction/GRU_Math_best_optimizer.pth')
        torch.save([best_sum_loss ,best_final_loss,lr ],'QuadraticFunction/GRU_Math_best_loss.txt')
        best_flag = True
        
    return best_sum_loss, best_final_loss, best_flag


# In[31]:


Global_Train_Steps = 50 #可修改1000
Optimizee_Train_Steps = 100#######100
UnRoll_STEPS = 20
Evaluate_period = 1 #可修改
optimizer_lr = 0.1 #可修改
global_loss_list ,flag = Learning_to_learn_global_training( GRU_Math_Optimizee,
                                                            Global_Train_Steps,
                                                            Optimizee_Train_Steps,
                                                            UnRoll_STEPS,
                                                            Evaluate_period,
                                                            optimizer_lr)


# In[32]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

STEPS = 100
x = np.arange(STEPS)

Adam = 'Adam' #因为这里Adam使用Pytorch
SGD = 'SGD'
RMS = 'RMS'

#for _ in range(1): 
for loop_count in range(1):  # 在这里设置循环次数
   
    LSTM_BlackBox_learner = Learner(f , W, Y, LSTM_BlackBox_Optimizee, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    LSTM_BlackBox_lr_learner = Learner(f , W, Y, LSTM_BlackBox_Optimizee_lr, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    LSTM_Math_learner = Learner(f , W, Y, LSTM_Math_Optimizee, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    GRU_Math_learner = Learner(f , W, Y, GRU_Math_Optimizee, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
 
    lstm_blackbox_losses, lstm_blackbox_sum_loss = LSTM_BlackBox_learner()
    lstm_blackbox_lr_losses, lstm_blackbox_lr_sum_loss = LSTM_BlackBox_lr_learner()
    lstm_math_losses, lstm_math_sum_loss = LSTM_Math_learner()
    gru_math_losses, gru_math_sum_loss = GRU_Math_learner()
    
    llstm_blackbox_losses_tensor = torch.tensor(lstm_blackbox_losses)
    lstm_blackbox_lr_losses_tensor = torch.tensor(lstm_blackbox_lr_losses)
    lstm_math_losses_tensor = torch.tensor(lstm_math_losses)
    gru_math_losses_tensor = torch.tensor(gru_math_losses)
    
    p1, = plt.plot(x, lstm_blackbox_losses_tensor.numpy(), label='LSTM_BlackBox')
    p2, = plt.plot(x, lstm_blackbox_lr_losses_tensor.numpy(), label='LSTM_BlackBox_Lr')
    p3, = plt.plot(x, lstm_math_losses_tensor.numpy(), label='LSTM_Math') 
    p4, = plt.plot(x, gru_math_losses_tensor.numpy(), label='GRU_Math') 
    
    
    plt.yscale('log')
    plt.legend(handles=[p1, p2, p3, p4])
    plt.title('Losses')
    plt.show()
    print("lstm_black={},lstm_black_lr={},lstm_math={}, gru_math={}".format(lstm_blackbox_sum_loss,lstm_blackbox_lr_sum_loss, 
                                                                            lstm_math_sum_loss, gru_math_sum_loss ))


# In[33]:


import matplotlib.pyplot as plt
import numpy as np
import torch

# 定义颜色列表
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# 定义线条样式列表
linestyles = ['-', '--', '-.', ':', '-', '--']

# 定义标记样式列表
markers = ['*', 'o', 's', '^', 'D', 'x']

# Monte Carlo实验函数
def monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, optimizee):
    all_losses = []
    
    for _ in range(num_experiments):
        # 生成随机数据
        W = torch.randn(batchsize, num_samples, num_features)
        x_gt = torch.randn(batchsize, num_features)
        y_gt = torch.matmul(W, x_gt.unsqueeze(-1)).squeeze()
        noise_std = 0.5
        noise = torch.randn_like(y_gt) * noise_std
        Y = y_gt + noise
    
        if optimizee == LSTM_BlackBox_Optimizee:
            learner = Learner(f, W, Y, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
        elif optimizee == LSTM_BlackBox_Optimizee_lr:
            learner = Learner(f, W, Y, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
        
        elif optimizee == LSTM_Math_Optimizee:
            learner = Learner(f, W, Y, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
        elif optimizee == GRU_Math_Optimizee:
            learner = Learner(f, W, Y, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
        losses, _ = learner()
        losses_tensor = torch.tensor(losses)
        all_losses.append(losses_tensor)        
    
    # 计算所有实验的平均损失、标准差和95%置信区间
    all_losses_array = np.array(all_losses)
    avg_losses = np.mean(all_losses_array, axis=0)
    std_losses = np.std(all_losses_array, axis=0)
    sem_losses = std_losses / np.sqrt(num_experiments)  # 标准误差
    conf_interval = 1.96 * sem_losses  # 95%置信区间
    return avg_losses, conf_interval

# 设置参数
num_experiments = 10  # 蒙特卡洛实验次数
num_samples = 250
num_features = 10
batchsize = 128
num_steps = 100

# 进行蒙特卡洛实验
avg_losses_lstm_blackbox, conf_interval_lstm_blackbox = monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, LSTM_BlackBox_Optimizee)
avg_losses_lstm_blackbox_lr, conf_interval_lstm_blackbox_lr = monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, LSTM_BlackBox_Optimizee_lr)
avg_losses_lstm_math, conf_interval_lstm_math = monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, LSTM_Math_Optimizee)
avg_losses_gru_math_lr, conf_interval_gru_math_lr = monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, GRU_Math_Optimizee)

# 绘制结果
x = np.arange(num_steps)

plt.figure(figsize=(10, 6))
for i, (avg_losses, conf_interval, label) in enumerate([
    (avg_losses_lstm_blackbox, conf_interval_lstm_blackbox, 'LSTM-DM'),
    (avg_losses_lstm_blackbox_lr, conf_interval_lstm_blackbox_lr, 'LSTM-LR'),
    (avg_losses_lstm_math, conf_interval_lstm_math, 'LSTM-Math'),
    (avg_losses_gru_math_lr, conf_interval_gru_math_lr, 'GRU-Math-LR')
]):
    plt.plot(x, avg_losses, label=label, linestyle=linestyles[i % len(linestyles)], 
             marker=markers[i % len(markers)], markersize=6, color=colors[i % len(colors)], 
             markeredgewidth=0, markevery=10)  # 去掉标记边框并减少标记密度
    plt.fill_between(x, avg_losses - conf_interval, avg_losses + conf_interval, color=colors[i % len(colors)], alpha=0.2)

plt.yscale('log')
plt.title('QuadraticFunction-Tesing Loss with Monte Carlo')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# 添加图例并将其放在右上角
plt.legend(loc='upper right')
plt.show()


# In[34]:


import numpy as np
import matplotlib.pyplot as plt
import torch

# 定义颜色列表
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# 定义线条样式列表
linestyles = ['-', '--', '-.', ':', '-', '--']

# 定义标记样式列表
markers = ['*', 'o', 's', '^', 'D', 'x']

STEPS = 100
x = np.arange(STEPS)

Adam = 'Adam'  # 因为这里Adam使用Pytorch
SGD = 'SGD'
RMS = 'RMS'
AdaGrad = 'AdaGrad'

# 在这里设置循环次数
for loop_count in range(1):  
    LSTM_BlackBox_learner = Learner(f, W, Y, LSTM_BlackBox_Optimizee, STEPS, eval_flag=True, reset_theta=True, retain_graph_flag=True)
    LSTM_BlackBox_lr_learner = Learner(f, W, Y, LSTM_BlackBox_Optimizee_lr, STEPS, eval_flag=True, reset_theta=True, retain_graph_flag=True)
    LSTM_Math_learner = Learner(f , W, Y, LSTM_Math_Optimizee, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    GRU_Math_learner = Learner(f , W, Y, GRU_Math_Optimizee, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)
    
    lstm_blackbox_losses, lstm_blackbox_sum_loss = LSTM_BlackBox_learner()
    lstm_blackbox_lr_losses, lstm_blackbox_lr_sum_loss = LSTM_BlackBox_lr_learner()
    lstm_math_losses, lstm_math_sum_loss = LSTM_Math_learner()
    gru_math_losses, gru_math_sum_loss = GRU_Math_learner()
    
    lstm_blackbox_losses_tensor = torch.tensor(lstm_blackbox_losses)
    lstm_blackbox_lr_losses_tensor = torch.tensor(lstm_blackbox_lr_losses)
    lstm_math_losses_tensor = torch.tensor(lstm_math_losses)
    gru_math_losses_tensor = torch.tensor(gru_math_losses)
    
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(x, lstm_blackbox_losses_tensor.numpy(), label='LSTM-DM', linestyle=linestyles[0], marker=markers[0], markersize=6, color=colors[0], markeredgewidth=0, markevery=10)
    plt.plot(x, lstm_blackbox_lr_losses_tensor.numpy(), label='LSTM-LR', linestyle=linestyles[1], marker=markers[1], markersize=6, color=colors[1], markeredgewidth=0, markevery=10)
    plt.plot(x, lstm_math_losses_tensor.numpy(), label='LSTM-Math', linestyle=linestyles[2], marker=markers[2], markersize=6, color=colors[2], markeredgewidth=0, markevery=10)
    plt.plot(x, gru_math_losses_tensor.numpy(), label='GRU-Math-LR', linestyle=linestyles[3], marker=markers[3], markersize=6, color=colors[3], markeredgewidth=0, markevery=10)
    
    plt.yscale('log')
    plt.title('QuadraticFunction Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)  # 添加网格线
    # 添加图例并将其放在右上角
    plt.legend(loc='upper right')


    # Monte Carlo实验函数
    def monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, optimizee):
        all_losses = []
    
        for _ in range(num_experiments):
            # 生成随机数据
            W = torch.randn(batchsize, num_samples, num_features)
            x_gt = torch.randn(batchsize, num_features)
            y_gt = torch.matmul(W, x_gt.unsqueeze(-1)).squeeze()
            noise_std = 0.5
            noise = torch.randn_like(y_gt) * noise_std
            Y = y_gt + noise
    
            if optimizee == LSTM_BlackBox_Optimizee:
                learner = Learner(f, W, Y, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
            elif optimizee == LSTM_BlackBox_Optimizee_lr:
                learner = Learner(f, W, Y, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
        
            elif optimizee == LSTM_Math_Optimizee:
                learner = Learner(f, W, Y, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
            elif optimizee == GRU_Math_Optimizee:
                learner = Learner(f, W, Y, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
            losses, _ = learner()
            losses_tensor = torch.tensor(losses)
            all_losses.append(losses_tensor)        
    
        # 计算所有实验的平均损失、标准差和95%置信区间
        all_losses_array = np.array(all_losses)
        avg_losses = np.mean(all_losses_array, axis=0)
        std_losses = np.std(all_losses_array, axis=0)
        sem_losses = std_losses / np.sqrt(num_experiments)  # 标准误差
        conf_interval = 1.96 * sem_losses  # 95%置信区间
        return avg_losses, conf_interval

    # 设置参数
    num_experiments = 10  # 蒙特卡洛实验次数
    num_samples = 250
    num_features = 10
    batchsize = 128
    num_steps = 100

    # 进行蒙特卡洛实验
    avg_losses_lstm_blackbox, conf_interval_lstm_blackbox = monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, LSTM_BlackBox_Optimizee)
    avg_losses_lstm_blackbox_lr, conf_interval_lstm_blackbox_lr = monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, LSTM_BlackBox_Optimizee_lr)
    avg_losses_lstm_math, conf_interval_lstm_math = monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, LSTM_Math_Optimizee)
    avg_losses_gru_math_lr, conf_interval_gru_math_lr = monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, GRU_Math_Optimizee)


    # 绘制蒙特卡洛实验结果
    plt.subplot(1, 2, 2)
    # 绘制结果
    x = np.arange(num_steps)

    for i, (avg_losses, conf_interval, label) in enumerate([
        (avg_losses_lstm_blackbox, conf_interval_lstm_blackbox, 'LSTM-DM'),
        (avg_losses_lstm_blackbox_lr, conf_interval_lstm_blackbox_lr, 'LSTM-LR'),
        (avg_losses_lstm_math, conf_interval_lstm_math, 'LSTM-Math'),
        (avg_losses_gru_math_lr, conf_interval_gru_math_lr, 'GRU-Math-LR')
    ]):
        plt.plot(x, avg_losses, label=label, linestyle=linestyles[i % len(linestyles)], 
                 marker=markers[i % len(markers)], markersize=6, color=colors[i % len(colors)], 
                 markeredgewidth=0, markevery=10)  # 去掉标记边框并减少标记密度
        plt.fill_between(x, avg_losses - conf_interval, avg_losses + conf_interval, color=colors[i % len(colors)], alpha=0.2)

    plt.yscale('log')
    plt.title('QuadraticFunction Tesing Loss with Monte Carlo')
    plt.xlabel('Epochs')
    plt.grid(True)  # 添加网格线
    # 添加图例并将其放在右上角
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[38]:


import matplotlib.pyplot as plt
import numpy as np
import torch

# 定义颜色列表
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# 定义线条样式列表
linestyles = ['-', '--', '-.', ':', '-', '--']

# 定义标记样式列表
markers = ['*', 'o', 's', '^', 'D', 'x']

# Monte Carlo实验函数
def monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, optimizee):
    all_losses = []
    
    for _ in range(num_experiments):
        # 生成随机数据
        W = torch.randn(batchsize, num_samples, num_features)
        x_gt = torch.randn(batchsize, num_features)
        y_gt = torch.matmul(W, x_gt.unsqueeze(-1)).squeeze()
        noise_std = 1.0
        noise = torch.randn_like(y_gt) * noise_std
        Y = y_gt + noise
    
        if optimizee == LSTM_BlackBox_Optimizee:
            learner = Learner(f, W, Y, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
        elif optimizee == LSTM_BlackBox_Optimizee_lr:
            learner = Learner(f, W, Y, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
        
        elif optimizee == LSTM_Math_Optimizee:
            learner = Learner(f, W, Y, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
        elif optimizee == GRU_Math_Optimizee:
            learner = Learner(f, W, Y, optimizee, num_steps, eval_flag=True, reset_theta=True, retain_graph_flag=True)
            
        losses, _ = learner()
        losses_tensor = torch.tensor(losses)
        all_losses.append(losses_tensor)        
    
    # 计算所有实验的平均损失、标准差和95%置信区间
    all_losses_array = np.array(all_losses)
    avg_losses = np.mean(all_losses_array, axis=0)
    std_losses = np.std(all_losses_array, axis=0)
    sem_losses = std_losses / np.sqrt(num_experiments)  # 标准误差
    conf_interval = 1.96 * sem_losses  # 95%置信区间
    return avg_losses, conf_interval

# 设置参数
num_experiments = 10  # 蒙特卡洛实验次数
num_samples = 250
num_features = 10
batchsize = 128
num_steps = 100

# 进行蒙特卡洛实验
avg_losses_lstm_blackbox, conf_interval_lstm_blackbox = monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, LSTM_BlackBox_Optimizee)
avg_losses_lstm_blackbox_lr, conf_interval_lstm_blackbox_lr = monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, LSTM_BlackBox_Optimizee_lr)
avg_losses_lstm_math, conf_interval_lstm_math = monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, LSTM_Math_Optimizee)
avg_losses_gru_math_lr, conf_interval_gru_math_lr = monte_carlo_experiment(num_experiments, num_samples, num_features, batchsize, num_steps, GRU_Math_Optimizee)

# 绘制结果
x = np.arange(num_steps)

plt.figure(figsize=(10, 6))
for i, (avg_losses, conf_interval, label) in enumerate([
    (avg_losses_lstm_blackbox, conf_interval_lstm_blackbox, 'LSTM-DM'),
    (avg_losses_lstm_blackbox_lr, conf_interval_lstm_blackbox_lr, 'LSTM-LR'),
    (avg_losses_lstm_math, conf_interval_lstm_math, 'LSTM-Math'),
    (avg_losses_gru_math_lr, conf_interval_gru_math_lr, 'GRU-Math-LR')
]):
    plt.plot(x, avg_losses, label=label, linestyle=linestyles[i % len(linestyles)], 
             marker=markers[i % len(markers)], markersize=6, color=colors[i % len(colors)], 
             markeredgewidth=0, markevery=10)  # 去掉标记边框并减少标记密度
    plt.fill_between(x, avg_losses - conf_interval, avg_losses + conf_interval, color=colors[i % len(colors)], alpha=0.2)

plt.yscale('log')
plt.title('QuadraticFunction-Tesing Loss with Monte Carlo')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)  # 添加网格线
# 添加图例并将其放在右上角
plt.legend(loc='upper right')
plt.show()


# In[ ]:




