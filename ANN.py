'''
Created on 21 Oct 2018

@author: sean
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
# Loading the dataset
os.getcwd()
#with open('fashion-mnist_train.csv') as train_file, open('fashion-mnist_test.csv') as test_file:
with open('temp.csv') as train_file, open('fashion-mnist_test.csv') as test_file:
    train_data = list(csv.reader(train_file)) 
    test_data = list(csv.reader(test_file))

train_data = np.delete(train_data, (0), axis=0)
trainSetX = np.array(train_data.T, dtype="float64")
trainSetY = np.array(trainSetX[0], dtype="int")
trainSetXN = np.delete(trainSetX, (0), axis=0)/255


test_data = np.delete(test_data, (0), axis=0)
testSetX = np.array(test_data.T, dtype="float64")
testSetY = np.array(testSetX[0], dtype="int")
testSetXN = np.delete(testSetX, (0), axis=0)/255


iterations = 0
#learning rate
alpha = 0.2
#number of iterations to perform
limit = 2000
#number of inputs
inputUnits = trainSetXN.shape[0]
hiddenUnits = 32
outputUnits = 10
#number of train samples
m = trainSetXN.shape[1]
#number of test samples
m_test = testSetXN.shape[1]

tmpSetY = np.zeros((outputUnits,trainSetY.shape[0]))
for i in range(trainSetY.shape[0]):
    tmpSetY[trainSetY[i],i]= 1
trainSetY = tmpSetY

#initialisation of weights and biases
W1 = np.random.rand(hiddenUnits, inputUnits)/2 - 0.25
W2 = np.random.rand(outputUnits, hiddenUnits)/2 - 0.25
b1 = np.zeros((hiddenUnits,1))
b2 = np.zeros((outputUnits,1))

costValues = np.zeros(limit)


def sig(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    exps = np.exp(x)
    return exps/np.sum(exps,axis=0)

def get_cost(Y,A):
    return -np.sum(Y*np.log(A))
while(iterations<limit):
    #feedforward
    Z1 = np.dot(W1,trainSetXN) + b1
    A1 = sig(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    
    #compute loss
    L = get_cost(trainSetY,A2)
    J = L/m
    costValues[iterations] = J
    #compute gradients
    dZ2 = A2-trainSetY
    dW2 = np.dot(dZ2,A1.T)
    dW2 /= m
    db2 = np.sum(dZ2,axis=1).reshape(outputUnits,1)
    db2 /= m
    
    dZ1 = np.dot(W2.T,dZ2)*(sig(Z1)*(1-sig(Z1)))
    dW1 = np.dot(dZ1,trainSetXN.T)
    dW1 /= m
    db1 = (np.sum(dZ1,axis=1)).reshape(hiddenUnits,1)
    db1 /= m
    #gradient descent
    W1 -= alpha*dW1
    W2 -= alpha*dW2
    b1 -= alpha*db1
    b2 -= alpha*db2
    #print every 100 iterations
    if iterations % 100 == 0:
        print (iterations)
    iterations += 1

print(J)

plt.plot(costValues)
plt.title('train set cost')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

Z1 = np.dot(W1,testSetXN) + b1
A1 = sig(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = softmax(Z2)

correct = 0
for i in range(m_test):
    if(np.argmax(A2[:,i])==testSetY[i]):
        correct += 1

accuracy = correct/(m_test)*100
print ("ANN outputted %d percent of test set correctly " % accuracy)