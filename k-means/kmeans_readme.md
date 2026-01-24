# K-Means Clustering

A simple implementation of the K-Means clustering algorithm from scratch using NumPy.

## Overview

K-Means is an unsupervised learning algorithm that groups data points into `k` clusters based on their similarity. The algorithm iteratively refines cluster assignments until convergence.

## Algorithm Steps

### 1. **Initialization**
```python
random_idx = np.random.choice(X.shape[0], self.k, replace=False)
self.centroids = X[random_idx]
```
- Randomly select `k` data points from the dataset as initial centroids
- These centroids represent the center of each cluster

### 2. **Assignment Step** (`_assign_clusters` method)
```python
distances = np.linalg.norm(X[:,np.newaxis] - self.centroids, axis=2)
return np.argmin(distances, axis=1)
```
**Calculation breakdown:**
- `X[:,np.newaxis]` reshapes X to enable broadcasting with centroids
- `X[:,np.newaxis] - self.centroids` calculates the difference between each point and each centroid
- `np.linalg.norm(..., axis=2)` computes Euclidean distance: $\sqrt{\sum(x_i - c_i)^2}$
- `np.argmin(distances, axis=1)` assigns each point to the nearest centroid

### 3. **Update Step**
```python
new_centroids = np.array([X[clusters==i].mean(axis=0) for i in range(self.k)])
```
- For each cluster, calculate the mean of all points assigned to it
- This mean becomes the new centroid position
- Formula: $c_i = \frac{1}{|C_i|} \sum_{x \in C_i} x$

### 4. **Convergence Check** (Tolerance)
```python
if np.linalg.norm(new_centroids - self.centroids) < self.tolerance:
    break
```
**How it works:**
- Calculates the Euclidean distance between old and new centroids
- If the movement is smaller than `tolerance` (default: 1e-4), the algorithm has converged
- This prevents unnecessary iterations once centroids stabilize
- Alternative: Check each centroid individually with `np.all(np.abs(new_centroids-self.centroids) < self.tolerance)`

## Usage

```python
# Create sample data
X = np.random.randn(100, 2)

# Initialize K-Means with 3 clusters, max 100 iterations
kmeans = KMeans(k=3, max_iters=100, tolerance=1e-4)

# Fit the model
kmeans.fit(X)

# Get cluster assignments for new data
predictions = kmeans.predict(X_test)

# Access final centroids
print(kmeans.centroids)
```

## Parameters

- **k**: Number of clusters to form
- **max_iters**: Maximum number of iterations (prevents infinite loops)
- **tolerance**: Convergence threshold - stops when centroid movement is below this value

## Methods

- **fit(X)**: Trains the model on data X
- **predict(X)**: Returns cluster assignments for data X
- **_assign_clusters(X)**: Internal method that assigns points to nearest centroids

## Time Complexity

- **Per iteration**: O(n × k × d)
  - n = number of data points
  - k = number of clusters
  - d = number of dimensions
- **Total**: O(iterations × n × k × d)


## Cluster Evaluation Metrics

These metrics help assess the quality of clustering results and determine the optimal number of clusters.

### 1. **Within-Cluster Sum of Squares (WCSS/Inertia)**
**Formula:** $WCSS = \sum_{i=1}^{k}\sum_{x \in C_i}||x - \mu_i||^2$

- Measures compactness/tightness within clusters
- Lower values indicate better clustering
- Used in the **Elbow Method** to find optimal k

**Example:**
```python
def calculate_wcss(X, clusters, centroids):
    wcss = 0
    for i in range(len(centroids)):
        cluster_points = X[clusters == i]
        wcss += np.sum((cluster_points - centroids[i])**2)
    return wcss
```

### 2. **Silhouette Score** (Most Popular)
**Formula:** $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$

Where:
- $a(i)$: average distance to points in the same cluster
- $b(i)$: average distance to points in the nearest neighboring cluster

**Interpretation:**
- Range: [-1, 1]
- **1**: Point is well-clustered
- **0**: Point is on the boundary between clusters
- **-1**: Point is likely in the wrong cluster
- Higher average silhouette score = better clustering

### 3. **Davies-Bouldin Index**
**Formula:** $DB = \frac{1}{k}\sum_{i=1}^{k}\max_{j \neq i}\frac{\sigma_i + \sigma_j}{d(c_i, c_j)}$

- Measures ratio of within-cluster to between-cluster distances
- Lower values indicate better clustering
- 0 is the best possible score
- Evaluates cluster separation and compactness simultaneously

### 4. **Calinski-Harabasz Index** (Variance Ratio Criterion)
**Formula:** $CH = \frac{SS_B/(k-1)}{SS_W/(n-k)}$

Where:
- $SS_B$: Between-cluster variance
- $SS_W$: Within-cluster variance

**Characteristics:**
- Ratio of between-cluster dispersion to within-cluster dispersion
- Higher values indicate better-defined clusters
- Fast to compute
- No bounded range

### 5. **Dunn Index**
**Formula:** $D = \frac{\min_{i \neq j} d(C_i, C_j)}{\max_k \text{diam}(C_k)}$

Where:
- Numerator: minimum distance between any two clusters
- Denominator: maximum diameter of any cluster

**Characteristics:**
- Higher values indicate better clustering
- Identifies well-separated and compact clusters
- Computationally expensive for large datasets

### 6. **Elbow Method** (Visual Technique)
- Plot WCSS against different values of k
- Look for the "elbow point" where the rate of decrease sharply changes
- The elbow represents a good trade-off between cluster count and compactness

**Example:**
```python
# Find optimal k using Elbow Method
wcss_values = []
for k in range(1, 11):
    kmeans = KMeans(k=k, max_iters=100)
    kmeans.fit(X)
    wcss = calculate_wcss(X, kmeans.predict(X), kmeans.centroids)
    wcss_values.append(wcss)

# Plot to find elbow
plt.plot(range(1, 11), wcss_values, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()
```

### Which Metric to Use?

- **WCSS + Elbow Method**: Best for choosing optimal k
- **Silhouette Score**: Most intuitive, good for overall quality assessment
- **Calinski-Harabasz**: Fast and reliable for well-separated clusters
- **Davies-Bouldin**: Good when you need a single number to compare different clusterings
- **Dunn Index**: Use when cluster separation is critical (but slower)


## K-Means++ Initialization

K-Means++ is an improved initialization method for K-Means that addresses one of its major weaknesses: poor initial centroid placement can lead to suboptimal clustering results.

### The Problem with Random Initialization

Standard K-Means randomly selects k data points as initial centroids. This can cause:
- Slow convergence
- Getting stuck in local minima
- Poor final clustering results
- High variance in results across different runs

### How K-Means++ Works

K-Means++ uses a **probabilistic approach** to spread out initial centroids, increasing the likelihood of finding better clusters.

**Algorithm:**

**Step 1:** Choose the first centroid randomly from the data points
```python
first_idx = np.random.randint(0, n_samples)
centroids.append(X[first_idx])
```

**Step 2:** For each remaining centroid (repeat k-1 times):

1. **Calculate distances** to the nearest existing centroid for each data point:
```python
distances = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2), axis=1)
```

2. **Compute selection probabilities** proportional to squared distances:
```python
probs = distances**2 / np.sum(distances**2)
```

3. **Select next centroid** using weighted random selection:
```python
next_idx = np.random.choice(n_samples, p=probs)
centroids.append(X[next_idx])
```

### Why Square the Distances?

The probability formula $P(x) = \frac{D(x)^2}{\sum D(x)^2}$ ensures:
- Points **farther** from existing centroids have **higher probability** of being selected
- This spreads centroids across the data space
- Reduces the chance of centroids clustering in one region

### Mathematical Intuition

For each unselected point $x$:
$$P(x) = \frac{D(x)^2}{\sum_{i=1}^{n} D(x_i)^2}$$

Where:
- $D(x)$ = distance to nearest existing centroid
- Points with $D(x) = 0$ (already selected) have probability 0
- Points far from all centroids have higher selection probability

### Benefits of K-Means++

1. **Better convergence**: Often converges faster than random initialization
2. **More consistent results**: Lower variance across multiple runs
3. **Theoretical guarantee**: Provably O(log k) competitive with optimal clustering
4. **Minimal overhead**: Initialization cost is O(nkd), same order as one K-Means iteration

### Comparison Example

```python
# Standard K-Means
kmeans = KMeans(k=3, max_iters=100)
kmeans.fit(X)  # Random initialization

# K-Means++
kmeans_pp = KMeansPP(k=3, max_iters=100)
kmeans_pp.fit(X)  # Smart initialization
```

### When to Use K-Means++

- **Always preferred** over random initialization
- Especially important for:
  - Large datasets
  - High-dimensional data
  - When k is large
  - When consistency across runs matters
  - Production systems requiring reliable results

### Implementation Note

After initialization, K-Means++ uses the **same iterative refinement** as standard K-Means (assignment → update → convergence check). The only difference is the initialization step.
