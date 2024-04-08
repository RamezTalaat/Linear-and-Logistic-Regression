import numpy as np
import pandas as pd

class LogisticRegression:
  def __init__(self , xTrain , yTrain, theta_0, theta_X, learning_rate, numOfItems):
    self.xTrain = xTrain
    self.yTrain = yTrain
    self.theta_0 = np.array(theta_0, dtype=np.float64)  # Ensure theta_0 is float64
    self.theta_X = np.array(theta_X, dtype=np.float64)  # Ensure theta_X is float64
    self.learning_rate = learning_rate
    self.numOfItems = numOfItems

  #sigmoid function with numpy
  def sigmoid(self,z):
    return 1/ (1 + np.e ** -z)
  
  #Hypothesis function
  def hypothesis(self, theta_0, theta_X, x):
    z = np.dot(x,theta_X) + theta_0
    return self.sigmoid(z)
  
  #Cost function to postive and negative
  #positive = -log(h(x)) ,, negative= -log(1-h(x))
  #J(w, b) = -(1/m) * sum(y*log(h(x)) + (1-y)*log(1-h(x)))
  def cost(self):
    hypo  = self.hypothesis(self.theta_0,self.theta_X,self.xTrain)
    cost = -(1/self.numOfItems) * np.sum(self.yTrain * np.log(hypo) + (1-self.yTrain) * np.log(1-hypo))
    return cost
  
  #gradient descent
  #w = w - learnRate * (1/m * xTranspose * (h(x) - y))
  #b = b - learnRate * (1/m sum(h(x) - y))
  def gradientDescent(self):
    hypo = self.hypothesis(self.theta_0,self.theta_X,self.xTrain)
    self.theta_X -= self.learning_rate * ((1/self.numOfItems) * np.dot(self.xTrain.T , (hypo - self.yTrain)))
    self.theta_0 -= self.learning_rate * ((1/self.numOfItems) * np.sum(hypo - self.yTrain))
    return [self.theta_X,self.theta_0]
  
  #train data
  def train(self, num):
    for i in range(num):
      self.theta_X, self.theta_0 = self.gradientDescent()
    return [self.theta_X,self.theta_0]

  #predict to make the out put is 0 or 1
  def predict(self, xtest):
    hypo = self.hypothesis(self.theta_0,self.theta_X,xtest)
    return [ 0 if result < 0.5 else 1 for result in hypo]

  
  def accuracy(self):
    predicted_labels = self.predict()
    correct_predictions = np.sum(predicted_labels == self.yTrain)
    total_samples = len(self.yTrain)
    accuracy = correct_predictions / total_samples
    return accuracy