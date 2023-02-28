# -*- coding: utf-8 -*-
"""
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
in "plain" scientific Python.
from https://mantas.info/code/simple_esn/
(c) 2012-2020 Mantas Lukoševičius
Distributed under MIT license https://opensource.org/licenses/MIT
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg 
import torch
# numpy.linalg is also an option for even fewer dependencies

# load the data
trainLen = 2000
testLen = 3000
initLen = 100
data = np.loadtxt('MackeyGlass_t17.txt')
#data=torch.tensor(data)
# plot some of it
plt.figure(10).clear()
plt.plot(data[:1000])
plt.title('A sample of data')

# generate the ESN reservoir
inSize = outSize = 1
resSize = 1000
a = 0.3 # leaking rate
torch.manual_seed(42)
Win = (torch.rand(resSize,1+inSize, dtype=torch.double) - 0.5) * 1
W = torch.rand(resSize,resSize, dtype=torch.double) - 0.5 
# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0]))
print('done.')
W *= 1.25 / rhoW

# allocated memory for the design (collected states) matrix
X = torch.zeros((1+inSize+resSize,trainLen-initLen)).type(torch.double)
# set the corresponding target matrix directly
Yt = torch.DoubleTensor(data[None,initLen:trainLen])

# run the reservoir with the data and collect X
x = torch.zeros((resSize,1), dtype=torch.double)
for t in range(trainLen):
    u = torch.DoubleTensor([(data[t])])
    x = (1-a)*x + a*torch.tanh( torch.matmul( Win, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( W, x )) 
    if t >= initLen:
        X[:,t-initLen] = torch.vstack([torch.Tensor([1]),u,x])[:,0]
   
# train the output by ridge regression
reg = 1e-8  # regularization coefficient
# direct equations from texts:
#X_T = X.T
#Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
#    reg*np.eye(1+inSize+resSize) ) )
# using scipy.linalg.solve:

Wout = linalg.solve( torch.matmul(X,X.T) + reg*torch.eye(1+inSize+resSize), torch.matmul(X,Yt.T)).T
# run the trained ESN in a generative mode. no need to initialize here, 
# because x is initialized with training data and we continue from there.
Y = torch.zeros((outSize,testLen))
u = torch.DoubleTensor([data[trainLen]])
Wout=np.array(Wout)
Wout=torch.DoubleTensor(Wout)

for t in range(testLen):
    
    x = (1-a)*x + a*torch.tanh( torch.matmul( Win, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( W, x ) )
    y = torch.matmul( Wout, torch.vstack([torch.DoubleTensor([1]),u,x])) 

    Y[:,t] = y
    # generative mode:
    u = y
  
    ## this would be a predictive mode:
    #u = data[trainLen+t+1] 

# compute MSE for the first errorLen time steps
errorLen = 500
mse = sum( np.square( data[trainLen:trainLen+errorLen] - 
    np.array(Y[0,0:errorLen]) ) ) / errorLen
print('MSE = ' + str(mse) )

# plot some signals
'''
plt.figure(1).clear()
plt.plot( data[trainLen+1:trainLen+testLen+1], 'g' )
plt.plot( np.array(Y.T), 'b' )
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.legend(['Target signal', 'Free-running predicted signal'])

plt.figure(2).clear()
plt.plot( np.array(X[0:20,0:200].T) )
plt.title(r'Some reservoir activations $\mathbf{x}(n)$')

plt.figure(3).clear()
plt.bar( np.arange(1+inSize+resSize), np.array(Wout[0]).T )
plt.title(r'Output weights $\mathbf{W}^{out}$')

plt.show()
'''
