import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False,parser='auto')
X, y = mnist.data / 255., mnist.target.astype(int)

# Only using labels 2, 3, and 4
labels = [2, 3, 4]
X_filtered = np.vstack([X[y == label] for label in labels])
y_filtered = np.hstack([y[y == label] for label in labels])

# Binarize function
def binarize(X):
    return 1. * (X >= 0.5)

# Binarize the filtered data
X_binarized = binarize(X_filtered)

# EM Algorithm
def EM(X, K, epochs):
    # Initialize parameters
    mu = np.random.uniform(low=0.15, high=0.85, size=(K, X.shape[1]))
    pi = np.ones(K) / K

    # Iterate EM steps
    for _ in range(epochs):
        # E-step
        eps = 1e-10  # for stability in log operations
        gamma = np.exp(np.log(pi + eps) + (X @ np.log(mu.T + eps)) + ((1 - X) @ np.log(1 - mu.T + eps)))
        gamma /= gamma.sum(axis=1)[:, np.newaxis]

        # M-step
        pi = gamma.mean(axis=0)
        mu = (gamma.T @ X) / gamma.sum(axis=0)[:, np.newaxis]

    return mu, pi

# Run EM algorithm
K = len(labels)
mu, pi = EM(X_binarized, K=K, epochs=20)
print("Centroids:")
print(mu)
print("Pi's:")
print(pi)

# Visualize results
fig, ax = plt.subplots(nrows=1, ncols=K, figsize=(15, 15), dpi=100)
for i in range(K):
    ax[i].imshow(mu[i].reshape(28, 28), cmap='gray')
    ax[i].set_title('Parameters class: {}\n pi = {:0.3f}'.format(labels[i], pi[i]), fontsize=K ** (-1) // 0.02)
    ax[i].axis('off')
plt.show()