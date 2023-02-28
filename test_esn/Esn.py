import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg 
import torch

trainLen = 2000
testLen = 2000
initLen = 100
data = np.loadtxt('MackeyGlass_t17.txt')

class ESN():
    def __init__(self, inSize, outSize, resSize, a=3,
                 spectral_radius=1.25,
                 random_state=42):
       
        self.inSize=inSize
        self.resSize=resSize
        self.outSize=outSize
        self.a=a
        self.spectral_radius=spectral_radius
        self.random_state=random_state
        torch.manual_seed(random_state)
        self.initmodel()
    def initmodel(self):
        self.a = 0.3
        W = torch.rand(self.resSize,self.resSize, dtype=torch.double) - 0.5
        self.Win = (torch.rand(self.resSize,1+self.inSize, dtype=torch.double) - 0.5) * 1
        print('Computing spectral radius...')
        rhoW = max(abs(linalg.eig(W)[0]))
        print('done.')
        self.W= W*(self.spectral_radius/rhoW)
        
    def fit(self,data):
        X = torch.zeros((1+self.inSize+self.resSize,trainLen-initLen)).type(torch.double)
        # set the corresponding target matrix directly
        Yt = torch.DoubleTensor(data[None,initLen+1:trainLen+1])

        # run the reservoir with the data and collect X
        x = torch.zeros((self.resSize,1)).type(torch.double)
        for t in range(trainLen):
            u = torch.DoubleTensor([(data[t])])
            x = (1-self.a)*x + self.a*torch.tanh( torch.matmul( self.Win, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W, x )) 
            if t >= initLen:
                X[:,t-initLen] = torch.vstack([torch.Tensor([1]),u,x])[:,0]
        self.x=x
        self.X=X
        # train the output by ridge regression
        reg = 1e-8  # regularization coefficient
        # direct equations from texts:
        #X_T = X.T
        #Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
        #    reg*np.eye(1+inSize+resSize) ) )
        # using scipy.linalg.solve:
        reg = 1e-8
        Wout = linalg.solve( torch.matmul(self.X,self.X.T) + reg*torch.eye(1+self.inSize+self.resSize), torch.matmul(self.X,Yt.T)).T
        Wout=np.array(Wout)
        Wout=torch.DoubleTensor(Wout)
        self.Wout=torch.DoubleTensor(Wout)
    def predict(self,data):    
        # run the trained ESN in a generative mode. no need to initialize here, 
        # because x is initialized with training data and we continue from there.
        Y = torch.zeros((self.outSize,testLen))
        u = torch.DoubleTensor([data[trainLen]])
        x=self.x

        for t in range(testLen):
            
            x = (1-self.a)*x + self.a*torch.tanh( torch.matmul( self.Win, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W, x ) )
            y = torch.matmul( self.Wout, torch.vstack([torch.DoubleTensor([1]),u,x])) 

            Y[:,t] = y
            # generative mode:
            u = y
        self.Y=Y
        return Y
    def mse(self):
        errorLen = 500
        mse = sum( np.square( data[trainLen+1:trainLen+errorLen+1] - 
            np.array(self.Y[0,0:errorLen]) ) ) / errorLen
        print('MSE = ' + str(mse) )

e=ESN(1,1,1000)
e.fit(data)
print(e.predict(data))