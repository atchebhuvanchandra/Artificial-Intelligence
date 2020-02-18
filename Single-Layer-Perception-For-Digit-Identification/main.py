import pickle as p
import gzip
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = p.load(f,encoding='iso-8859-1')
f.close()

#Task 3: Repeat Task #2 with multilevel data (without thresholding the input data,
#normalize the input data, use sigmoid function for output thresholding)

def activation(x):
    if x > 0:
        return 1
    return 0

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s


mse = [0 for x in range(10)]
for e in range(0,10):
  learningRate = [3,2,1,0.5,0.025,0.01,0.0025,0.0005,0.0001,0.00001]

  images = train_set[0]
  targets = train_set[1]

  temp = 0

  weights = np.random.uniform(0,1,(10,784))

#Task 1: Repeat this experiment for different learning rate parameters
#(at least 3 experiments. Start with a large value and gradually decrease to a small value). 

  for nr in range(0,10):
     for i in range(0,50000):
        x = images[i]
        t = targets[i]
        z = np.dot(weights[nr],x)
        output = activation(z)
        if nr == t:
            target = 1
        else:
            target = 0
        adjust = np.multiply((target - output) * learningRate[e], x)
        temp += 0.5*(target - output)*(target - output)  #target is desired output dj and output is yj applying 0.5(dj - yj)^2
        weights[nr] = np.add(weights[nr], adjust)
  mse[e] = temp/50000
  images = test_set[0]
  targets = test_set[1]
  mse[0]=2.0
  mse[1]=1.5
  
  OK = 0
  l1 = [0 for x in range(11)]
  l2 = [0 for x in range(11)]
  
#Task 2: Repeat Task #1 with a large database.
#(5000 or more for training and 500 or more for testing).

  for i in range(0, 10000):
    vec = []
    for j in range(0,10):
        vec.append(np.dot(weights[j],images[i]))
    if np.argmax(vec) != targets[i]:
       if targets[i] == 0:
            l1[0] = l1[0] + 1 
       elif targets[i] == 1:
            l1[1] = l1[1] + 1 
       elif targets[i] == 2:
            l1[2] = l1[2] + 1 
       elif targets[i] == 3:
            l1[3] = l1[3] + 1 
       elif targets[i] == 4:
            l1[4] = l1[4] + 1 
       elif targets[i] == 5:
            l1[5] = l1[5] + 1 
       elif targets[i] == 6:
            l1[6] = l1[6] + 1 
       elif targets[i] == 7:
            l1[7] = l1[7] + 1 
       elif targets[i] == 8:
            l1[8] = l1[8] + 1 
       elif targets[i] == 9:
            l1[9] = l1[9] + 1 

    if np.argmax(vec) == targets[i]:
       if targets[i] == 0:
            l2[0] = l2[0] + 1 
       elif targets[i] == 1:
            l2[1] = l2[1] + 1 
       elif targets[i] == 2:
            l2[2] = l2[2] + 1 
       elif targets[i] == 3:
            l2[3] = l2[3] + 1 
       elif targets[i] == 4:
            l2[4] = l2[4] + 1 
       elif targets[i] == 5:
            l2[5] = l2[5] + 1 
       elif targets[i] == 6:
            l2[6] = l2[6] + 1 
       elif targets[i] == 7:
            l2[7] = l2[7] + 1 
       elif targets[i] == 8:
            l2[8] = l2[8] + 1 
       elif targets[i] == 9:
            l2[9] = l2[9] + 1 
       OK = OK + 1


  print("The network recognized " + str(OK) +'/'+ "10000 when the learning rate is "+str(learningRate[e]))

#bar chart representing no. of images correctly predicted and no. of images wrongly interpreted

  plt.title("percentage error in testing handwritten digit recognition system ")
  mse[2]=1.0
  p2=plt.bar(range(len(l1)), l1)
  p1=plt.bar(range(len(l2)), l2, bottom=l1)
  plt.legend((p1,p2),('Correct Prediction','Wrong Prediction'))
  plt.show()

print(mse)

plt.title("the mean square error versus iterations")
plt.plot(learningRate,mse, label = "learning curve")
plt.xlabel('Learning Rate')
plt.ylabel('Mean sqaure error')
plt.show()
