import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd



X_train = np.genfromtxt(fname = "hw02_data_points.csv", delimiter = ",", dtype = float)
y_train = np.genfromtxt(fname = "hw02_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    n_samples = len(y)
    class_priors = np.array([np.sum(y == k) / n_samples for k in np.unique(y)])
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_class_means(X, y):
    # your implementation starts below
    K = np.unique(y).size
    D = X.shape[1]
    sample_means = np.zeros((K, D))
    for k in np.unique(y):
        sample_means[k-1, :] = np.mean(X[y == k], axis=0)
    # your implementation ends above
    return(sample_means)

sample_means = estimate_class_means(X_train, y_train)
print(sample_means)



# STEP 5
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_class_covariances(X, y):
    # your implementation starts below
    K = np.unique(y).size
    D = X.shape[1]
    sample_covariances = np.zeros((K, D, D))

    for k in range(1, K+1):
        X_k = X[y == k]
        mean_k = X_k.mean(axis=0)
        cov_k = np.cov(X_k, rowvar=False, bias=True)
        sample_covariances[k-1] = cov_k
    # your implementation ends above
    return(sample_covariances)

sample_covariances = estimate_class_covariances(X_train, y_train)
print(sample_covariances)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, class_means, class_covariances, class_priors):
    # your implementation starts below
    N, D = X.shape
    K = class_means.shape[0]
    score_values = np.zeros((N, K))

    for k in range(K):
        diff = X - class_means[k]
        inv_cov = np.linalg.inv(class_covariances[k])
        log_det_cov = np.log(np.linalg.det(class_covariances[k]))
        log_prior = np.log(class_priors[k])
        term1 = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
        score_values[:, k] = -0.5 * (D * np.log(2 * np.pi) + log_det_cov + term1) + log_prior

    
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    y_pred = np.argmax(scores, axis=1) + 1  
    K = np.unique(y_truth).size
    confusion_matrix = np.zeros((K, K), dtype=int)

    for i in range(len(y_truth)):
        confusion_matrix[y_truth[i]-1, y_pred[i]-1] += 1
    confusion_matrix=confusion_matrix.T
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)



def draw_classification_result(X, y, class_means, class_covariances, class_priors):
    class_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#6a3d9a"])
    K = np.max(y)

    x1_interval = np.linspace(-75, +75, 151)
    x2_interval = np.linspace(-75, +75, 151)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T
    scores_grid = calculate_score_values(X_grid, class_means, class_covariances, class_priors)

    score_values = np.zeros((len(x1_interval), len(x2_interval), K))
    for c in range(K):
        score_values[:,:,c] = scores_grid[:, c].reshape((len(x1_interval), len(x2_interval)))

    L = np.argmax(score_values, axis = 2)

    fig = plt.figure(figsize = (6, 6))
    for c in range(K):
        plt.plot(x1_grid[L == c], x2_grid[L == c], "s", markersize = 2, markerfacecolor = class_colors[c], alpha = 0.25, markeredgecolor = class_colors[c])
    for c in range(K):
        plt.plot(X[y == (c + 1), 0], X[y == (c + 1), 1], ".", markersize = 4, markerfacecolor = class_colors[c], markeredgecolor = class_colors[c])
    plt.xlim((-75, 75))
    plt.ylim((-75, 75))
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    return(fig)
    
fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_different_covariances.pdf", bbox_inches = "tight")



# STEP 8
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_shared_class_covariance(X, y):
    # your implementation starts below
    N, D = X.shape  
    K = np.unique(y).size
    shared_covariance = np.zeros((D, D))  

    # Calculate the overall mean
    overall_mean = X.mean(axis=0)

    # Calculate shared covariance
    for k in range(1, K+1):
        X_k = X[y == k]  
        diff = X_k - overall_mean
        shared_covariance += np.dot(diff.T, diff)
    shared_covariance /= N

    sample_covariances = np.array([shared_covariance for _ in range(K)])
    
    # your implementation ends above
    return(sample_covariances)

sample_covariances = estimate_shared_class_covariance(X_train, y_train)
print(sample_covariances)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_shared_covariance.pdf", bbox_inches = "tight")


