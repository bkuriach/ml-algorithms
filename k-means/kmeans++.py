import numpy as np

class KMeansPP:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def _init_centroids(self, X):
        n_samples = X.shape[0]
        centroids = []

        # Step 1: pick first centroid randomly
        first_idx = np.random.randint(0, n_samples)
        centroids.append(X[first_idx])

        # Step 2: pick remaining centroids
        for _ in range(1, self.k):
            # compute distances to nearest centroid
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2), axis=1)
            probs = distances**2 / np.sum(distances**2)
            next_idx = np.random.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def fit(self, X):
        self.centroids = self._init_centroids(X)

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        return labels

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(X)
    
    
if __name__ == "__main__":
    np.random.seed(42)

    x1 = np.random.randn(33, 2) + [2, 2]
    x2 = np.random.randn(33, 2) + [6, 6]
    x3 = np.random.randn(34, 2) + [9, 9]
    X = np.vstack([x1, x2, x3])

    kmeans_pp = KMeansPP(k=3, max_iters=100)
    kmeans_pp.fit(X)

    print("Centroids:", kmeans_pp.centroids)
    X_test = np.array([[2, 3], [6, 5], [8, 9]])
    print("Predictions:", kmeans_pp.predict(X_test))