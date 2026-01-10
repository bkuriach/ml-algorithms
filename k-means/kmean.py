
import numpy as np

class KMeans:
    def __init__(
        self,
        n_clusters=3,
        max_iters = 100,
        tolerance=1e-4    
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None
        
    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        for i in range(self.max_iters):
            clusters = self._assign_clusters(X)
            new_centroids = np.array([X[clusters==k].mean(axis=0) for k in range(self.n_clusters)])
            
            if np.linalg.norm(new_centroids - self.centroids) < self.tolerance:
                break
            self.centroids = new_centroids           
    
    def predict(self, X):
        return self._assign_clusters(X)
    
    def _initialize_centroids(self, X):
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_indices]
    
    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    
if __name__ == "__main__":
    
    np.random.seed(42)
    X = np.random.rand(100,2) * 10
    
    c1 = np.random.randn(33, 2) + [2, 2]   # Cluster around (2,2)
    c2 = np.random.randn(33, 2) + [6, 6]   # Cluster around (6,6)  
    c3 = np.random.randn(34, 2) + [2, 8]   # Cluster around (2,8)
    X = np.vstack([c1, c2, c3])
    kmeans = KMeans(n_clusters=3, max_iters=100)
    kmeans.fit(X)

    print("Centroids:\n", kmeans.centroids)
    kmeans.predict(X)
    
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.predict(X), cmap='viridis')
    plt.show()

