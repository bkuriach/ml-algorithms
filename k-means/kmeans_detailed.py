import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(
        self,
        k=3,
        max_iterations=100,
        random_state=None,
        tolerance=1e-6
    ):
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.n_iter_ = None
    
    def fit(self, X):
        """
        Fit K-means clustering to the data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        """
        if self.random_state:
            np.random.seed(self.random_state)
        
        self.num_samples, self.num_features = X.shape
        
        # Initialize centroids randomly
        self.centroids = self._initialize_centroids(X)
        
        # Store previous centroids to check for convergence
        prev_centroids = np.zeros_like(self.centroids)
        
        for iteration in range(self.max_iterations):
            # Assign points to closest centroids
            self.labels = self._assign_clusters(X)
            
            # Update centroids
            prev_centroids = self.centroids.copy()
            self.centroids = self._update_centroids(X)
            
            # Check for convergence
            if self._has_converged(prev_centroids):
                print(f"Converged after {iteration + 1} iterations")
                self.n_iter_ = iteration + 1
                break
            
            if iteration % 10 == 0:
                inertia = self._calculate_inertia(X)
                print(f"Iteration {iteration}, Inertia: {inertia:.4f}")
        else:
            print(f"Reached maximum iterations ({self.max_iterations})")
            self.n_iter_ = self.max_iterations
        
        # Calculate final inertia
        self.inertia_ = self._calculate_inertia(X)
        
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster for new data points.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            labels: Cluster labels for each point
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            labels: Cluster labels for each point
        """
        self.fit(X)
        return self.labels
    
    def _initialize_centroids(self, X):
        """
        Initialize centroids randomly within the data range.
        """
        centroids = np.zeros((self.k, self.num_features))
        
        for i in range(self.num_features):
            min_val = np.min(X[:, i])
            max_val = np.max(X[:, i])
            centroids[:, i] = np.random.uniform(min_val, max_val, self.k)
        
        return centroids
    
    def _assign_clusters(self, X):
        """
        Assign each point to the closest centroid.
        """
        labels = np.zeros(X.shape[0])
        
        for i, point in enumerate(X):
            distances = [self._euclidean_distance(point, centroid) 
                        for centroid in self.centroids]
            labels[i] = np.argmin(distances)
        
        return labels.astype(int)
    
    def _update_centroids(self, X):
        """
        Update centroids based on the mean of assigned points.
        """
        new_centroids = np.zeros((self.k, self.num_features))
        
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If no points assigned to cluster, keep the old centroid
                new_centroids[i] = self.centroids[i]
        
        return new_centroids
    
    def _euclidean_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def _calculate_inertia(self, X):
        """
        Calculate within-cluster sum of squared distances (inertia).
        """
        inertia = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                centroid = self.centroids[i]
                inertia += np.sum((cluster_points - centroid) ** 2)
        
        return inertia
    
    def _has_converged(self, prev_centroids):
        """
        Check if centroids have converged.
        """
        distances = np.sqrt(np.sum((self.centroids - prev_centroids) ** 2, axis=1))
        return np.all(distances < self.tolerance)
    
    def plot_clusters(self, X, title="K-Means Clustering"):
        """
        Plot the clusters (works for 2D data).
        """
        if X.shape[1] != 2:
            print("Plotting only works for 2D data")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Plot data points
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i % len(colors)], alpha=0.6, label=f'Cluster {i}')
        
        # Plot centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                   c='black', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data with 3 distinct clusters
    cluster1 = np.random.normal(2, 0.5, (100, 2))
    cluster2 = np.random.normal(6, 0.5, (100, 2))
    cluster3 = np.random.normal([2, 6], 0.5, (100, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Apply K-means clustering
    kmeans = KMeans(
        k=3,
        max_iterations=100,
        random_state=42,
        tolerance=1e-6
    )
    
    labels = kmeans.fit_predict(X)
    
    print(f"Final centroids:\n{kmeans.centroids}")
    print(f"Final inertia: {kmeans.inertia_:.4f}")
    print(f"Number of iterations: {kmeans.n_iter_}")
    
    # Plot the results
    kmeans.plot_clusters(X, "K-Means Clustering Results")
    
    # Test prediction on new data
    new_points = np.array([[2.5, 2.5], [6.5, 6.5], [1.5, 6.5]])
    new_labels = kmeans.predict(new_points)
    print(f"New points predictions: {new_labels}")