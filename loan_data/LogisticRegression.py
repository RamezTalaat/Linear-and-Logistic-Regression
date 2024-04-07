import numpy as np
import pandas as pd

class LogisticRegression:
  def __init__(self , xTrain , yTrain, theta_0, theta_X, learning_rate, iterationNum):
    self.xTrain = xTrain
    self.yTrain = yTrain
    self.theta_0 = theta_0
    self.theta_X = theta_X
    self.learning_rate = learning_rate
    self.iterationNum = iterationNum

  #sigmid function with numpy
  def sigmoid(self,z):
    return 1/ (1 + np.e ** -z)
  
  #Hypothesis function
  def hypothesis(self, theta_0, theta_X, x):
    z = np.dot(x,theta_X)+theta_0
    return self.segmoid(z)
  
  #Cost function to postive and negative
  #positive = -log(h(x)) ,, negative= -log(1-h(x))
  #J(w, b) = -(1/m) * sum(y*log(h(x)) + (1-y)*log(1-h(x)))
  def cost(self):
    hypo  = self.hypothesis(self.theta_0,self.theta_X,self.xTrain)
    cost = -(1/self.iterationNum) * np.sum(self.yTrain * np.log(hypo) + (1-self.yTrain) * np.log(1-hypo))
    return cost
  
  #gradient descent
  #w = w - learnRate * (1/m * xTranspose * (h(x) - y))
  #b = b - learnRate * (1/m sum(h(x) - y))
  def gradientDescent(self):
    hypo = self.hypothesis(self.theta_0,self.theta_X,self.xTrain)
    self.theta_X -= self.learning_rate * ((1/self.iterationNum) * np.dot(self.xTrain.T , (hypo - self.yTrain)))
    self.theta_0 -= self.learning_rate * ((1/self.iterationNum) * np.sum(hypo - self.yTrain))
    return [self.theta_X,self.theta_0]
  
  #tarin data
  def tarin(self , num):
    for i in range(num):
      self.theta_X, self.theta_0 = self.gradientDescent()
    return [self.theta_X,self.theta_0]

  #predict to make the out put is 0 or 1
  def predict(self):
    hypo = self.hypothesis(self.theta_0,self.theta_X,self.xTrain)
    return [ 0 if result < 0.5 else 1 for result in hypo]
  
  