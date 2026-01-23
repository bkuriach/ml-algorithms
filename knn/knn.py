"Implement K-Nearest Neighbors algorithm for classification and regression."

import numpy as np 
from collections import Counter

class KNN:
	def __init__(self, k, task = "classification"):
		self.k = k 
		self.task=task

	def fit(self, X_train, y_train):
		"""
		Stores training data
		"""
		self.X_train= np.asarray(X_train)
		self.y_train = np.asarray(y_train)

	def _euclidean_distance(self, a, b):
		"""
		Compute eclidean distance, useful when b is a single point
		"""
		return np.sqrt(np.sum((a-b)**2))

	def _compute_distance(self, X_train, x_test):
		"""
		calculate euclidean distance in a vectorized way
		"""
		return np.sqrt(np.sum((X_train-x_test)**2, axis=1))

	def _get_k_neighbors(self, x_test):
		"""
		Compute K Nearest Neibhors using the distance calculation
		"""
		distance = self._compute_distance(self.X_train, x_test)
		idx = np.argsort(distance)[:self.k]
		return idx
	
	def predict(self,X_test):
		"""
		Loop based implementation of predict
		"""
		X_test = np.asarray(X_test)
		preds = []

		for x_test in X_test:
			idx = self._get_k_neighbors(x_test)
			neighbors = self.y_train[idx]

			if self.task == "classification":
				prediction = Counter(neighbors).most_common(1)[0][0]
				preds.append(prediction)
			elif self.task == "regression":
				prediction = np.mean(neighbors)
				preds.append(prediction)
			else:
				preds.append("Not supported")

		return np.array(preds)

	def predict_vectorized(self, X_test):
		"""
		Vectorized implementation
		"""
		preds = []
		distance = np.sqrt(np.sum((X_test[:,np.newaxis]-self.X_train)**2, axis=2))
		nearest_idx = np.argsort(distance, axis=1)[:,:self.k]
		nearest_labels = self.y_train[nearest_idx]

		if self.task=='classification':
			preds = [Counter(row).most_common(1)[0][0] for row in nearest_labels]
		elif self.task == 'regression':
			preds = np.mean(nearest_labels, axis=1)
		
		return np.array(preds)


if __name__ == "__main__":
	training_data = np.array([[1,2], [2,3], [3,4], [6,7], [7,8] , [8,9]])
	training_labels = np.array([0,0,0,1,1,1])

	X_test = np.array([[6,8], [2,2]])

	knn = KNN(k = 3, task='classification')
	knn.fit(training_data, training_labels)

	print("Loop prediction: ",knn.predict(X_test))

	print("Vectorized prediciton:", knn.predict_vectorized(X_test))




