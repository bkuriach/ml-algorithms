import numpy as np 
import matplotlib.pyplot as plt

class KMeans:
	def __init__(self, k, max_iters, tolerance = 1e-4):
		self.k=k 
		self.max_iters = max_iters
		self.tolerance = tolerance
		self.centroids = None

	def fit(self, X):
		np.random.seed(42)
		random_idx = np.random.choice(X.shape[0], self.k , replace = False)
		self.centroids = X[random_idx]

		for _ in range(self.max_iters):
			clusters = self._assign_clusters(X)

			new_centroids = np.array([X[clusters==i].mean(axis=0) for i in range(self.k)])

			if np.linalg.norm(new_centroids-self.centroids) < self.tolerance: 
			# or use np.all(np.abs(new_centroids-self.centroids)< self.tolerance) for stricter checking
				break
			self.centroids = new_centroids

	def predict(self, X):
		return self._assign_clusters(X)

	def _assign_clusters(self, X):		
		distances = np.linalg.norm(X[:,np.newaxis] - self.centroids, axis = 2)
		return np.argmin(distances, axis=1)


if __name__ == "__main__":
	np.random.seed(42)

	x1 = np.random.randn(33,2)+[2,2]
	x2 = np.random.randn(33,2)+[6,6]
	x3 = np.random.randn(34,2)+[9,9]
	X = np.vstack([x1,x2,x3])

	kmeans = KMeans(3,100)
	kmeans.fit(X)

	print("centroids ", kmeans.centroids)
	X_test = np.array([[2,3],[6,5],[8,9]])
	print("Predictions ", kmeans.predict(X_test))
	plt.scatter(X[:, 0], X[:, 1], c=kmeans.predict(X), cmap='viridis')
	plt.show()