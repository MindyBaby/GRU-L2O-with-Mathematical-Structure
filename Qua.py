import torch
import numpy as np
import torch.nn as nn
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    USE_CUDA = True  
print('USE_CUDA = {}'.format(USE_CUDA))

num_samples = 250
num_features = 10
batchsize = 128
torch.manual_seed(56)
W = torch.randn(batchsize, num_samples, num_features) 
x_gt = torch.randn(batchsize, num_features)
y_gt = torch.matmul(W, x_gt.unsqueeze(-1)).squeeze()
noise_std = 0.1
noise = torch.randn_like(y_gt) * noise_std
Y = y_gt + noise

def f(W,Y,x):
    """quadratic function : f(\theta) = \|W\theta - y\|_2^2"""
    if USE_CUDA:
        W = W.cuda()
        Y = Y.cuda()
        x = x.cuda()
    loss = ((torch.matmul(W,x.unsqueeze(-1)).squeeze()-Y)**2).sum(dim=1).mean(dim=0)
    return loss


# ## 1.LSTM_BlackBox----2016
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
Layers = 2
Hidden_nums = 20
Input_DIM = num_features
Output_DIM = num_features
output_scale_value=1
LSTM_BlackBox_Optimizee = LSTM_BlackBox_Optimizee_Model( Input_DIM, Output_DIM, Hidden_nums ,Layers , batchsize=batchsize,
                preprocess=False,output_scale=output_scale_value)   ###### preprocess=False
print(LSTM_BlackBox_Optimizee)
if USE_CUDA:
    LSTM_BlackBox_Optimizee = LSTM_BlackBox_Optimizee.cuda()


# ## 2.Lr_LSTM_BlackBox
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
Layers = 2
Hidden_nums = 20
Input_DIM = num_features
Output_DIM = num_features
output_scale_value=1
LSTM_BlackBox_Optimizee_lr = LSTM_BlackBox_Optimizee_Model_lr( Input_DIM, Output_DIM, Hidden_nums ,Layers , batchsize=batchsize,
                preprocess=False,output_scale=output_scale_value)   ###### preprocess=False
print(LSTM_BlackBox_Optimizee_lr)
if USE_CUDA:
    LSTM_BlackBox_Optimizee_lr = LSTM_BlackBox_Optimizee_lr.cuda()


# ## 3.LSTM_Math---2023
NORM_FUNC = {
    'exp': torch.exp,
    'eye': nn.Identity(),
    'sigmoid': lambda x: 2.0 * torch.sigmoid(x),
    'softplus': nn.Softplus(),
}
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
Layers = 2
Hidden_nums = 20
Input_DIM = num_features
Output_DIM = num_features
output_scale_value=1
LSTM_Math_Optimizee = LSTM_Math_Optimizee_Model( Input_DIM, Output_DIM, Hidden_nums ,Layers , batchsize=batchsize,
                preprocess=False,output_scale=output_scale_value)   ###### preprocess=False
print(LSTM_Math_Optimizee)
if USE_CUDA:
    LSTM_Math_Optimizee = LSTM_Math_Optimizee.cuda()


# ## 4.GRU_Math——Our model
NORM_FUNC = {
    'exp': torch.exp,
    'eye': nn.Identity(),
    'sigmoid': lambda x: 2.0 * torch.sigmoid(x),
    'softplus': nn.Softplus(),
}
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
Layers = 2
Hidden_nums = 20
Input_DIM = num_features
Output_DIM = num_features
output_scale_value=1
GRU_Math_Optimizee = GRU_Math_Optimizee_Model(Input_DIM, Output_DIM, Hidden_nums ,Layers , batchsize=batchsize,
                preprocess=False,output_scale=output_scale_value)   ###### preprocess=False
print(GRU_Math_Optimizee)
if USE_CUDA:
    GRU_Math_Optimizee = GRU_Math_Optimizee.cuda()


# ## 5.优化问题目标函数的学习过程
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

# ## 6.自动学习的LSTM优化器Learning to learn
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


# ## 7.LSTM_BlackBox----2016
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
Global_Train_Steps = 50 
Optimizee_Train_Steps = 100
UnRoll_STEPS = 20
Evaluate_period = 1 
optimizer_lr = 0.1 
global_loss_list ,flag = Learning_to_learn_global_training( LSTM_BlackBox_Optimizee,
                                                            Global_Train_Steps,
                                                            Optimizee_Train_Steps,
                                                            UnRoll_STEPS,
                                                            Evaluate_period,
                                                            optimizer_lr)


# ## 8.Lr_LSTM_BlackBox
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
Global_Train_Steps = 50 
Optimizee_Train_Steps = 100
UnRoll_STEPS = 20
Evaluate_period = 1 
optimizer_lr = 0.1
global_loss_list ,flag = Learning_to_learn_global_training( LSTM_BlackBox_Optimizee_lr,
                                                            Global_Train_Steps,
                                                            Optimizee_Train_Steps,
                                                            UnRoll_STEPS,
                                                            Evaluate_period,
                                                            optimizer_lr)


# ## 8.LSTM_Math
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
Global_Train_Steps = 50 
Optimizee_Train_Steps = 100
UnRoll_STEPS = 20
Evaluate_period = 1 
optimizer_lr = 0.1 
global_loss_list ,flag = Learning_to_learn_global_training( LSTM_Math_Optimizee,
                                                            Global_Train_Steps,
                                                            Optimizee_Train_Steps,
                                                            UnRoll_STEPS,
                                                            Evaluate_period,
                                                            optimizer_lr)


# ## 9.GRU_Math
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
Global_Train_Steps = 50 
Optimizee_Train_Steps = 100
UnRoll_STEPS = 20
Evaluate_period = 1 
optimizer_lr = 0.1 
global_loss_list ,flag = Learning_to_learn_global_training( GRU_Math_Optimizee,
                                                            Global_Train_Steps,
                                                            Optimizee_Train_Steps,
                                                            UnRoll_STEPS,
                                                            Evaluate_period,
                                                            optimizer_lr)
