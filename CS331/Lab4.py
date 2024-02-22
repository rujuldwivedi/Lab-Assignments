import numpy as np
import matplotlib.pyplot as plt
import math

# Here I am implementing the Binomial Mixtures using the Expectation Maximization algorithm

p1 = np.random.random() #this is the probability of success for the first coin
p2 = np.random.random() #this is the probability of success for the second coin
p = [p1, p2] #this is the list of probabilities of success for the two coins

print("Initial values of p:", p) #this is the initial list of the probabilities of success

pi1 = np.random.random() #this is the probability of choosing the first coin
pi2 = 1 - pi1 #this is the probability of choosing the second coin
pi = [pi1, pi2] #this is the list of probabilities of choosing the two coins

print("Initial values of pi:", pi) #this is the initial list of the prior probabilities

print("\nNow for the Special Case:")

n = 100 #this is the number of trials
N = 10 #this is the number of samples 
k = 2 #this is the number of coins
X = [] #this is the list of samples

# Basically I am randomly generating the samples and then using the parameters
# to calculate the posterior probabilities and then using the posterior probabilities
# to update the parameters and then checking for convergence

# The parameters should converge to the true parameters which are given by the initial values
# and the X's are generated from the true parameters and then used to estimate the true parameters
# using the EM algorithm

for i in range(N): #this is the loop to generate the samples
    X.append(np.random.randint(0, n)) #this is the random number generator for the samples

posterior = [] #this is the list of the posterior probabilities

epsilon = 1e-3 #this is the threshold for the convergence

while True: #this is the loop for the EM algorithm
    for i in range(N): #this is the loop for the samples
        L = [] #this is the list of the likelihoods
        deno = 0 #this is the denominator for the posterior probability
        for j in range(k): #this is the loop for the coins
            deno = deno + pi[j] * (math.comb(n, X[i])) * (p[j]**X[i]) * ((1 - p[j])**(n - X[i]))
        # Here I am calculating the denominator for the posterior probability by summing over the coins
        # the product of the prior probability of the coin, the likelihood of the sample given the coin,
        # and the likelihood of the complement of the sample given the coin
        for j in range(k): #this is the loop for the coins
            gamma = (pi[j] * (math.comb(n, X[i])) * (p[j]**X[i]) * ((1 - p[j])**(n - X[i]))) / deno #this is the posterior probability
            L.append(gamma)
        posterior.append(L) #this is the list of the posterior probabilities

# Note:
# pi[j] is the prior probability of the coin and (math.comb(n, X[i])) * (p[j]**X[i]) * ((1 - p[j])**(n - X[i]))
# is the likelihood or the class conditional density of the sample given the coin. 
# Gamma is the posterior probability of the coin given the sample.

    new_pi = [] #this is the list of the new prior probabilities
    new_p = [] #this is the list of the new probabilities of success

    for j in range(k): #this is the loop for the coins
        pi_val = 0 #this is the new prior probability
        for i in range(N): #this is the loop for the samples
            pi_val = pi_val + posterior[i][j] / N #this is the sum of the posterior probabilities
        new_pi.append(pi_val) #this is the new prior probability

    for j in range(k): #this is the loop for the coins
        deno = 0 #this is the denominator for the new probability of success
        for i in range(N): #this is the loop for the samples
            deno = deno + posterior[i][j] #this is the sum of the posterior probabilities
        deno = deno*n #this is the product of the sum of the posterior probabilities and the number of trials
        num = 0 #this is the numerator for the new probability of success
        for i in range(N): #this is the loop for the samples
            num = num + posterior[i][j] * X[i] #this is the sum of the product of the posterior probabilities and the samples
        new_p.append(num / deno) #this is the new probability of success

# For convergence I am checking if the difference between the old and new probabilities of success and the prior probabilities is less than the threshold
# If it is then I am updating the probabilities of success and the prior probabilities and breaking the loop
# Otherwise I am updating the probabilities of success and the prior probabilities
        
    b = True #this is the flag for the convergence
    for j in range(k): #this is the loop for the coins
        if abs(p[j] - new_p[j]) > epsilon or abs(pi[j] - new_pi[j]) > epsilon: #this is the condition for the convergence
            b = False
            break
    if b: 
        p = new_p
        pi = new_pi
        break
    p = new_p #this is the update of the probability of success
    pi = new_pi #this is the update of the prior probability

print("Final values of p:", p) #this is the final list of the probabilities of success
print("Final values of pi:", pi) #this is the final list of the prior probabilities
plt.hist(X, bins = 100) #this is the histogram of the samples
plt.show() #this is the command to show the histogram

print("\nNow for the General Case:")

# Here I am implementing the Binomial Mixtures using the Expectation Maximization algorithm
# as a function for a general case
def binomial_EM(k, N, n, X): #this is the function for the EM algorithm
  p = [] #this is the list of the probabilities of success
  epsilon = 1e-6 #this is the threshold for the convergence
  for i in range(k): #this is the loop for the coins
    p.append(np.random.random()) #this is the random number generator for the probabilities of success
  pi = np.random.dirichlet(np.ones(k)) #this is the list of the prior probabilities

  posterior = [] #this is the list of the posterior probabilities
  while True: #this is the loop for the EM algorithm
    for i in range(N): #this is the loop for the samples
        L = [] #this is the list of the likelihoods
        deno = 0
        for j in range(k): #this is the loop for the coins
            deno = deno + pi[j] * (math.comb(n, X[i])) * (p[j]**X[i]) * ((1 - p[j])**(n - X[i])) 
        # Here I am calculating the denominator for the posterior probability by summing over the coins
        # the product of the prior probability of the coin, the likelihood of the sample given the coin,
        # and the likelihood of the complement of the sample given the coin
        for j in range(k): #this is the loop for the coins
            gamma = (pi[j] * (math.comb(n, X[i])) * (p[j]**X[i]) * ((1 - p[j])**(n - X[i]))) / deno #this is the posterior probability
            L.append(gamma)
        posterior.append(L) #this is the list of the posterior probabilities

    new_pi = [] #this is the list of the new prior probabilities
    new_p = [] #this is the list of the new probabilities of success

    for j in range(k): #this is the loop for the coins
        pi_val = 0
        for i in range(N): #this is the loop for the samples
            pi_val = pi_val + posterior[i][j]/N #this is the sum of the posterior probabilities

        new_pi.append(pi_val) #this is the new prior probability

    for j in range(k): #this is the loop for the coins
        deno = 0
        for i in range(N): #this is the loop for the samples
            deno = deno + posterior[i][j] #this is the sum of the posterior probabilities
        deno = deno*n #this is the product of the sum of the posterior probabilities and the number of trials
        num = 0
        for i in range(N): #this is the loop for the samples
            num = num + posterior[i][j] * X[i] #this is the sum of the product of the posterior probabilities and the samples
        new_p.append(num / deno) #this is the new probability of success

    b = True #this is the flag for the convergence
    for j in range(k): #this is the loop for the coins
        if abs(p[j] - new_p[j]) > epsilon or abs(pi[j] - new_pi[j]) > epsilon: #this is the condition for the convergence
            b = False
            break
    if b:
        p = new_p
        pi = new_pi
        break
    p = new_p #this is the update of the probability of success
    pi = new_pi #this is the update of the prior probability

  return (p,pi) #this is the final list of the probabilities of success and the prior probabilities

k = 10 #this is the number of coins
N = 1000 #this is the number of samples
n = 100 #this is the number of trials
X = [] #this is the list of samples
for i in range(N): #this is the loop to generate the samples
    X.append(np.random.randint(0, n)) #this is the random number generator for the samples

(p,pi) = binomial_EM(2,10,100,X) #this is the call to the function for the EM algorithm
print("Final values of p:", p) #this is the final list of the probabilities of success
print("Final values of pi:", pi) #this is the final list of the prior probabilities
plt.hist(X, bins = 100) #this is the histogram of the samples
plt.show() #this is the command to show the histogram
