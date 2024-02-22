import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df

# K-Means on the Old Faithful dataset
# Load the dataset
data = pd.read_csv('faithful.csv') # Load the dataset
data = data.drop('Unnamed: 0', axis=1) # Drop the first column
data.columns = ['eruptions', 'waiting'] # Rename the columns

print("The first 5 rows of the dataset are:")
print(data.head())

# Plot the data
plt.scatter(data['eruptions'], data['waiting']) 
plt.xlabel('Eruption time (min)') 
plt.ylabel('Waiting time (min)')
plt.title('Old Faithful Eruptions')
plt.show()

# K-Means (without using sklearn)
# Define the number of clusters
k = 2
# Initialize the centroids
np.random.seed(42)

# Now we will randomly initialize the centroids by selecting random points from the dataset
# We are creating a dictionary with the keys as the cluster number
# and the values as the coordinates of the centroids in the 2D space
# and the reason for the range(1,6) is to ensure that the centroids are within the range of the data
# because the eruption time and waiting time are both within the range of 1 to 5
centroids = {
    i+1: [np.random.randint(1, 6), np.random.randint(1, 6)] 
    for i in range(k) 
} 

print("The initial centroids are:")
print(centroids)

# Plot the data with the initial centroids
plt.scatter(data['eruptions'], data['waiting'], c='black') 

# For the k number of clusters, we will plot the centroids with different colors to distinguish them

colmap = {1: 'r', 2: 'g'} # Define the colors for the clusters which will help us to visualize the clusters
for i in centroids.keys(): # Using the keys of the dictionary to iterate through the centroids
    plt.scatter(*centroids[i], color=colmap[i]) # Here we are using the * operator to unpack the list of coordinates
plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting time (min)')
plt.title('Old Faithful Eruptions')
plt.show()

#Here we are going to assign the data to the clusters:
#We are going to calculate the distance of each data point from the centroids
#then assign the data to the cluster with the closest centroid
#and we will also color the data points based on the cluster they belong to
#We are going to add 3 new columns to the dataset:
#The first column will be the distance of each data point from the first centroid
#The second column will be the distance of each data point from the second centroid
#The third column will be the cluster to which the data point belongs
#A fourth column will be the color of the data point
#The color of the data point will be based on the cluster to which the data point belongs
#The color map is defined above

# Assign the data to the clusters
def assignment(data, centroids):
    for i in centroids.keys(): # Iterate through the centroids
        # format(i) is used to format (replace as placeholder) the string with the value of i
        # so basically we are creating a new column for each centroid by the name 
        # 'distance_from_i' where i is the number of the centroid
        # and then we calculate the distance of each data point from the centroid 
        # and then store the distance in the new column
        data['distance_from_{}'.format(i)] = (
            np.sqrt(
                (data['eruptions'] - centroids[i][0]) ** 2 +
                (data['waiting'] - centroids[i][1]) ** 2
            )
        ) 
    # In the above lines, we calculate distance of each data point from each centroid
    # where eruptions and waiting are the columns of the dataset
    # and [i][0] and [i][1] are the coordinates of the i-th centroid 
        
    #Below we are going to create a new column 'closest' which will store the number of the closest centroid
    #We are going to use the idxmin() function, which returns the index of the minimum value in each row
    #to find the closest centroid for each data point
    #We are going to use the map() function to assign the data to the closest centroid
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    # Create a list of the distance columns, this will help us to find the closest centroid
    # Because let's say i is 1 then 'distance_from_{}'.format(i) will be 'distance_from_1'
    # and the for loop will iterate through the centroids and create a list of the distance columns
    # and then we will use this list to find the closest centroid

    data['closest'] = data.loc[:, centroid_distance_cols].idxmin(axis=1)
    # loc is used to access a group of rows and columns by label(s) or a boolean array
    # it takes two arguments, the first one is the row and the second one is the column
    # in this case we are using the colon to access all the rows
    # and the list centroid_distance_cols to access the columns
    # and then we use idxmin() to find the index of the minimum value in each row denoted by axis=1

    data['closest'] = data['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    # map() is used to map the data to the closest centroid
    # lambda x: int(x.lstrip('distance_from_')) is used to remove the 'distance_from_' from the column name
    # where lstrip is used to remove the characters from the left side of the string
    # and then convert the string to an integer
    # This is done to assign the data to the closest centroid

    data['color'] = data['closest'].map(lambda x: colmap[x]) # Assign the color to the data based on the cluster
    return data
data = assignment(data, centroids)

print("The first 5 rows of the dataset with the closest centroids are:")
print(data.head())

# Plot the data with the initial clusters
plt.scatter(data['eruptions'], data['waiting'], color=data['color'], alpha=0.5, edgecolor='k') 
for i in centroids.keys(): 
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting time (min)')
plt.title('Old Faithful Eruptions')
plt.show()

# Update the centroids
import copy
old_centroids = copy.deepcopy(centroids) # Deep copy is used to create a new object with the same value
def update(k): 
    for i in centroids.keys(): # Calculate the mean of the data points in the cluster which are closest to the centroid
        centroids[i][0] = np.mean(data[data['closest'] == i]['eruptions'])
        centroids[i][1] = np.mean(data[data['closest'] == i]['waiting'])
    return k
centroids = update(centroids)

print("The updated centroids are:")
print(centroids)

# Plot the data with the updated centroids
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(data['eruptions'], data['waiting'], color=data['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting time (min)')
plt.title('Old Faithful Eruptions')
plt.show()

# Repeat the assignment of the data to the clusters
# This is done to ensure that the data points are assigned to the closest centroid
data = assignment(data, centroids) 

print("The first 5 rows of the dataset with the closest centroids are:")
print(data.head())

# Plot the data with the updated clusters
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(data['eruptions'], data['waiting'], color=data['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting time (min)')
plt.title('Old Faithful Eruptions')
plt.show()

# Continue the process until the clusters do not change
while True:
    closest_centroids = data['closest'].copy(deep=True) # Create a deep copy of the closest centroids
    centroids = update(centroids) # Update the centroids
    data = assignment(data, centroids) # Assign the data to the clusters
    if closest_centroids.equals(data['closest']): 
    # Check if the closest centroids are the same as the previous iteration
        break

print("The first 5 rows of the dataset with the closest centroids are:")
print(data.head())

# Plot the final clusters
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(data['eruptions'], data['waiting'], color=data['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting time (min)')
plt.title('Old Faithful Eruptions')
plt.show()

# Calculate the accuracy of the K-Means algorithm
def accuracy(data, centroids):
    acc = 0
    for i in centroids.keys():
        acc += np.sum((data[data['closest'] == i]['eruptions'] - centroids[i][0]) ** 2 + 
                          (data[data['closest'] == i]['waiting'] - centroids[i][1]) ** 2)
    return np.sqrt(acc)
acc = accuracy(data, centroids)

accuracy_rate = 1/(acc/len(data))

print(f"The accuracy of the K-Means algorithm is: {accuracy_rate:.2f} %")
print(accuracy_rate)

# Now generate data for the K-Means algorithm using Gaussian distributions

#Initial parameters
pi1 = 1/3 # Mixing coefficient for the first cluster
pi2 = 1/3 # Mixing coefficient for the second cluster
pi3 = 1/3 # Mixing coefficient for the third cluster

# Given prior probabilities
pi1 = pi2 = pi3 = 1/3
# Number of data points
n = 1000

#Question 2a
# Generating the data {(Xi, Zi)} where Xi is the data and Zi is the cluster to which the data belongs

# Generating the data for the first cluster
mu1 = np.array([0, 0])  # Mean of the first cluster
sigma1 = np.array([[0.05, 0], [0, 0.5]])  # Covariance matrix of the first cluster
data1 = np.random.multivariate_normal(mu1, sigma1, size=int(pi1 * n))  # Generating the data for the first cluster

# Generating the data for the second cluster
mu2 = np.array([1, 1])  # Mean of the second cluster
sigma2 = np.array([[0.5, 0], [0, 0.05]])  # Covariance matrix of the second cluster
data2 = np.random.multivariate_normal(mu2, sigma2, size=int(pi2 * n))  # Generating the data for the second cluster

# Generating the data for the third cluster
mu3 = np.array([1, 0])  # Mean of the third cluster
sigma3 = np.array([[0.5, 0.5], [0, 0.05]])  # Covariance matrix of the third cluster
data3 = np.random.multivariate_normal(mu3, sigma3, size=int(pi3 * n))  # Generating the data for the third cluster

# Combining the data
data = np.concatenate((data1, data2, data3), axis=0)  # Combining the data for the three clusters

# Generating the cluster data
cluster1 = np.ones(int(pi1 * n))  # Cluster data for the first cluster
cluster2 = 2 * np.ones(int(pi2 * n))  # Cluster data for the second cluster
cluster3 = 3 * np.ones(int(pi3 * n))  # Cluster data for the third cluster
cluster_data = np.concatenate((cluster1, cluster2, cluster3), axis=0)  # Combining the cluster data

# Displaying the first 5 rows of the dataset
print("The first 5 rows of the dataset are:")
print(pd.DataFrame(data, columns=['X1', 'X2']).head())

# Plotting the data
plt.figure(figsize=(5, 5))
plt.scatter(data[:, 0], data[:, 1], c=cluster_data)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data with Clusters where the True Parameters are Known')
plt.show()

# Here's the multivariate_normal_pdf() function which is used to calculate the probability density function of the multivariate normal distribution
# That is denoted by p(x,z|theta) in the formula
def multivariate_normal_pdf(data, mu, sigma):
    return np.diag(1. / (np.sqrt(2 * np.pi * np.linalg.det(sigma))) * np.exp(-0.5 * np.dot(np.dot((data - mu), np.linalg.inv(sigma)), (data - mu).T)))


#Question 2b
# After having generated the dataset {(Xi)}, now we'll calculate posterior and see which is maximum to assign to it

# Calculating the posterior probabilities
posterior1 = pi1 * multivariate_normal_pdf(data, mu1, sigma1)  # Posterior probability for the first cluster
posterior2 = pi2 * multivariate_normal_pdf(data, mu2, sigma2)  # Posterior probability for the second cluster
posterior3 = pi3 * multivariate_normal_pdf(data, mu3, sigma3)  # Posterior probability for the third cluster

# Assigning the data to the clusters for the maximum posterior
# this r is the cluster to which the data belongs
r = np.argmax(np.array([posterior1, posterior2, posterior3]), axis=0) + 1  # Assigning the data to the clusters

# Plotting the data with the clusters
plt.figure(figsize=(5, 5))
plt.scatter(data[:, 0], data[:, 1], c=r)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data with maximised posterior probabilities')
plt.show()

def gammaCalculation(data, pi, mu, sigma):
    #Calculating the responsibilities
    gamma = np.zeros((data.shape[0], k)) # Initialising the responsibilities
    for i in range(k):
        gamma[:,i] = pi[i] * multivariate_normal_pdf(data, mu[i], sigma[i])
    gamma = gamma / np.sum(gamma, axis=1, keepdims=True) # Normalising the responsibilities so that they sum to 1
    return gamma

def Q(data, pi, mu, sigma, gamma):
    #Calculating the Q function
    Q = 0
    for i in range(k):
        Q += np.sum(gamma[:,i] * np.log(pi[i] * multivariate_normal_pdf(data, mu[i], sigma[i])))
    return Q

def H(gamma):
    #Calculating the entropy
    return -np.sum(gamma * np.log(gamma))

#Now we'll optimise gamma using the fact that the lower bound increases with each iteration
#We'll use the EM algorithm to optimise the lower bound
#We'll start by initialising the parameters

#Initial parameters
k = 3 # Number of clusters
n = data.shape[0] # Number of data points
d = data.shape[1] # Dimension of the data
pi = np.ones(k) / k # Mixing coefficients
mu = np.random.rand(k, d) # Means
sigma = np.array([np.eye(d)] * k) # Covariance matrices
gamma = gammaCalculation(data, pi, mu, sigma) # Responsibilities
Q_old = Q(data, pi, mu, sigma, gamma) # Initial value of the Q function
H_old = H(gamma) # Initial value of the entropy
print("Initial value of the Q function:", Q_old)
print("Initial value of the entropy:", H_old)

#Now we'll optimise the lower bound using the EM algorithm
#EM-Algorithm Implementation

Q_list = []

def EM_Algorithm(data, pi, mu, sigma, gamma, Q_old, H_old):
    #EM-Algorithm
    Q_new = Q_old
    H_new = H_old
    while True:
        #E-Step
        gamma = gammaCalculation(data, pi, mu, sigma)
        #M-Step
        pi = np.sum(gamma, axis=0) / n
        mu = np.dot(gamma.T, data) / np.sum(gamma, axis=0, keepdims=True).T
        for i in range(k):
            sigma[i] = np.dot((gamma[:,i].reshape(-1,1) * (data - mu[i])).T, (data - mu[i])) / np.sum(gamma[:,i])
        Q_new = Q(data, pi, mu, sigma, gamma)
        H_new = H(gamma)

        #Here we will create a list of the different values of the Q function
        #This will help us to visualise the log likelihood
        #We will use this list to plot the log likelihood

        if Q_new - Q_old < 1e-9:
            break

        Q_list.append(Q_new)

        Q_old = Q_new
        H_old = H_new
    return pi, mu, sigma, gamma, Q_new, H_new

pi, mu, sigma, gamma, Q_new, H_new = EM_Algorithm(data, pi, mu, sigma, gamma, Q_old, H_old)
print("Optimised value of the Q function:", Q_new)
print("Optimised value of the entropy:", H_new)

#Now we'll assign the data to the clusters for the maximum posterior
r = np.argmax(gamma, axis=1) + 1 # Assigning the data to the clusters

#Plotting the data with the clusters
plt.figure(figsize=(5, 5))
plt.scatter(data[:, 0], data[:, 1], c=r)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data with maximised posterior probabilities using the EM algorithm')
plt.show()

#Plotting the log likelihood
plt.figure(figsize=(5, 5))
plt.plot(Q_list)
plt.xlabel('Iterations')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood')
plt.show()
