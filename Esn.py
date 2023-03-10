import numpy as np
from scipy import linalg 
import torch
from sklearn.utils import check_array
'''
1. 전체적으로 ESN을 학습에 있어서는 먼저 ESN에 알맞은 parameters를 입력하고 정의한다
2. train values를 esn모델에 learning을 통해 input w, W를 learning 하고 fix시킨다
3. train values의 learning된 weight들과 구하고자 하는 output을 linear regression에 fit 한다
3. test values를 esn모델에 learning 하지말고 통과 시킨다
4. test values의 learning된 weight들을 linear regression에 predict를 통해 구하고자 하는 target을 알 수 있다
'''

class ESN():
    def __init__(self, n_readout, 
                 resSize, damping=0.3, spectral_radius=None,
                 weight_scaling=1.25,initLen=100, random_state=42,inter_unit=torch.tanh):
        
        self.resSize=resSize
        self.n_readout=n_readout # 마지막에 연결된 노드 갯수
        self.damping = damping  # 소실하는 정도로 모든 노드를 사용하지 않는다
        self.spectral_radius=spectral_radius # 1보다 작아야한다
        self.weight_scaling=weight_scaling
        self.initLen=initLen # 처음에 버릴 길이
        self.random_state=random_state
        self.inter_unit=inter_unit
        self.Win=None # 학습하여 input weight가 있다면 넣어준다
        self.W=None # 학습하여 weight가 있다면 넣어준다
        torch.manual_seed(random_state) # torch에서 random값 고정
        self.out=None
        
        
    def init_fit(self,input):
        if input.ndim==1:
            input=input.reshape(1,-1)
        input = check_array(input, ensure_2d=True)
        n_feature, n_input = input.shape
        W = torch.rand(self.resSize,self.resSize, dtype=torch.double) - 0.5
        self.Win = (torch.rand(self.resSize,1+n_feature, dtype=torch.double) - 0.5) * 1
        print('Computing spectral radius...')
        #spectral_radius = max(abs(linalg.eig(W)[0]))  default
        print('done.')
         # 가중치 업데이트 과정 -> weight_scaling 값으로 나눈 값으로 가중치를 업데이트함. -> weight_scaling은 가중치 학습률이다.
        rhoW = max(abs(linalg.eig(W)[0]))
        if self.spectral_radius == None:
            self.W= W*(self.weight_scaling/rhoW)
        else:
            self.W= W*(self.weight_scaling/self.spectral_radius)
        
        Yt=torch.DoubleTensor(input[:,self.initLen+1:])
        
        X = torch.zeros((1+n_feature+self.resSize,n_input-self.initLen-1)).type(torch.double) # X의 크기는 n_레저버 * 1
        x = torch.zeros((self.resSize,1)).type(torch.double)    # x의 크기는 n_레저버 * 1
        
        
        for t in range(n_input-1):
            u=torch.DoubleTensor(np.array(input[:,t],ndmin=2)) # input에서 값을 하나씩 들고온다
            
            x = (1-self.damping)*x + self.damping*self.inter_unit(torch.matmul(self.Win, torch.vstack([torch.DoubleTensor([1]),u])) + torch.matmul( self.W, x ))
            # x에 전체노드에서 소실률에 의거해 위의 식에 따라 계산된 weight값을 저장한다 
            if t >= self.initLen:
                X[:,t-self.initLen] = torch.vstack([torch.DoubleTensor([1]),u,x])[:,0]  # X에 1,u,x를 쌓아 저장한다 
        self.X=X
        self.x=x
        self.out = input[:,n_input-1] #generative mode를 위한 input의 last value를 저장
       
        #### train the output by ridge regression
        # reg = 1e-8  # regularization coefficient
        #### direct equations from texts:
        # X_T = X.T
        # Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
        # reg*np.eye(1+inSize+resSize) ) )
        # using scipy.linalg.solve:
        reg = 1e-8
        Wout = linalg.solve(torch.matmul(self.X,self.X.T) + reg*torch.eye(1+n_feature+self.resSize), torch.matmul(X,Yt.T)).T
        
        Wout=np.array(Wout)
        Wout=torch.DoubleTensor(Wout)
        self.Wout=torch.DoubleTensor(Wout)
       
        return self
        
        

    def fit(self,input): # 처음 학습을 시킬 때 사용 
        self=self.init_fit(input)
        return self.X[2:,:].T # 계산된 weight들을 들고와서 regression에 사용한다
    
    def pre_fit(self,input): # 이미 학습을 시킨 후 w와 input w가 있을 때 사용
        input = check_array(input, ensure_2d=True)
        n_input, n_feature = input.shape
        
        if self.Win == None:    # 앞에서 학습을 안 시켰을 경우 아래 적용
            self.Win=(torch.rand(self.resSize,1+n_feature, dtype=torch.double) - 0.5) * 1
        if self.W == None:      # 앞에서 학습을 안 시켰을 경우 아래 적용
            self.W=torch.rand(self.resSize,self.resSize, dtype=torch.double) - 0.5
            self.W=self.weight*(self.weight_scaling/self.spectral_radius)
        
        X = torch.zeros((1+n_feature+self.resSize,n_input-self.initLen)).type(torch.double)
        x = torch.zeros((self.resSize,1)).type(torch.double)
        
        for t in range(n_input):
            u=torch.DoubleTensor(np.array(input[t,:],ndmin=2)).T
            x = (1-self.damping)*x + self.damping*self.inter_unit(torch.matmul(self.Win, torch.vstack([torch.DoubleTensor([1]),u])) + torch.matmul( self.W, x ))
            if t >= self.initLen:
                X[:,t-self.initLen] = torch.vstack([torch.DoubleTensor([1]),u,x])[:,0]    
        return self.X[2:,:].T  # 계산된 weight들을 들고와서 regression에 사용한다
           
    def predict(self,outLen):    #gerative mode
        # run the trained ESN in a generative mode. no need to initialize here, 
        # because x is initialized with training data and we continue from there.
        x=self.x
       
        Y = torch.zeros((self.n_readout,outLen))
        u = torch.DoubleTensor(self.out)
     
        for t in range(outLen):
            
            x = (1-self.damping)*x + self.damping*self.inter_unit( torch.matmul( self.Win, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W, x ) )
            y = torch.matmul( self.Wout, torch.vstack([torch.DoubleTensor([1]),u,x])) 
            Y[:,t] = y
            # generative mode:
            u = y
        self.Y=Y
        return Y


