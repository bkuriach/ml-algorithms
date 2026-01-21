
import numpy as np

class LogisticRegression:
  def __init__(self, learning_rate, epochs, regularization=None, alpha=None):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.regularization = regularization
    self.alpha = alpha        
    self.weights = None
    self.bias = None
 
  def _sigmoid(self, z):
    return 1/(1+(np.exp(-1*z)))

  def fit(self, X, y):
    self.num_samples, self.num_features = X.shape
    self.weights = np.random.normal(0,0.2, self.num_features)
    self.bias = 0
 
    for i in range(self.epochs):
      z = np.dot(X, self.weights) + self.bias
      y_pred = self._sigmoid(z)
 
      loss = -(1/self.num_samples) * np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
      if i%50==0:
        print(f'Loss: {loss}, Epoch: {i}')

      dw = (1/self.num_samples)*np.dot(X.T, (y_pred-y))
      db = (1/self.num_samples) * np.sum(y_pred-y)

      self.weights -= self.learning_rate*dw
      self.bias -= self.learning_rate*db

 
  def predict(self, X, threshold = 0.5):
    z = np.dot(X, self.weights)+ self.bias
    y_pred = self._sigmoid(z)
    predicted_classes = [1 if i>threshold else 0 for i in y_pred]
    return y_pred, np.array(predicted_classes)
 

if __name__=="__main__":
  np.random.seed(42)
  X_class_0 = np.random.randn(100,2)*1.5 + np.array([1,1])
  X_class_1 = np.random.randn(100,2)*1.5 + np.array([5,5])
  X = np.vstack((X_class_0, X_class_1))
  y = np.array([0]*len(X_class_0) + [1]*len(X_class_1))
  logistic_regression = LogisticRegression(learning_rate=0.01,epochs=500)
  logistic_regression.fit(X,y)
  print(logistic_regression.predict([[1,1],[5,5], [0,0], [6,6],[3,3]]))   
  print(f'Weights: {logistic_regression.weights}, Bias: {logistic_regression.bias}')      
                                                                                                          

