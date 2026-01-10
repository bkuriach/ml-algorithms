
import numpy as np


class LinearRegression:
    def __init__(
        self,
        learning_rate,
        epochs,
        regularization=None,
        alpha=0.01

    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.alpha = alpha
        self.weights = None
        self.bias = None  
    
    def fit(
            self,
            X,
            y
    ):
        
        self.num_samples, self.num_features = X.shape
        self.weights = np.random.normal(0, 0.2, self.num_features)
        self.bias = 0
        self.loss = []

        for i in range(self.epochs):
            y_predict = self.predict(X)
            loss = self._mse(
                y_actual=y,
                y_predict=y_predict
            )
            self.loss.append(loss)

            self._update_weights(
                X,
                y,
                y_predict
            )

            if i % 100 == 0:
                print(f" Epoch {i}, Loss :{loss:.4f}")
        
    def _mse(
        self,
        y_actual,
        y_predict
    ):
        loss = (1/self.num_samples) * np.sum((y_actual-y_predict)**2)
        reg_loss = 0
        if self.regularization == 'l1':
            reg_loss = self.alpha * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            reg_loss = self.alpha * np.sum(self.weights**2)

        return loss + reg_loss
    
    def _update_weights(
        self,
        X,
        y,
        y_pred
    ):
        dw = (1/self.num_samples) * np.dot(X.T, (y_pred-y))
        db = (1/self.num_samples) * np.sum(y_pred-y)

        if self.regularization == "l1":
            dw += self.alpha*np.sign(self.weights)
        elif self.regularization == "l2":
            dw += self.alpha * 2 * (self.weights)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(
        self,
        X
    ):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


if __name__ == "__main__":
    np.random.seed(42)

    X = np.random.rand(1000, 2)
    y = 3*X[:, 0] + 5*X[:, 1] + np.random.randn(1000)*0.1

    linear_regression = LinearRegression(
        learning_rate=0.01,
        epochs=2000,
        regularization='l2', 
        alpha=0.01
    )
    linear_regression.fit(X, y)
    print("Weights", linear_regression.weights)
    print("Bias ", linear_regression.bias)

    print("Prediction (first five samples) ", linear_regression.predict(X[:5]))
    print("Actual ", y[:5])
