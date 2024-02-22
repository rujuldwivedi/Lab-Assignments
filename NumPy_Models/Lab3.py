import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
X = np.random.binomial(n=1, p=0.4, size=100)

def likelihood(p, X):
    n = len(X)
    zeros = np.sum(X == 0)
    ones = n - zeros
    return p**ones * (1-p)**zeros

p = np.linspace(0, 1, 100)
plt.plot(p, likelihood(p, X))
plt.xlabel("p")
plt.ylabel("Likelihood")
plt.show()

def loss(p, X):
    return np.log(likelihood(p, X))

plt.plot(p, loss(p, X))
plt.xlabel("p")
plt.ylabel("Loss")
plt.show()

def p_hat(X):
    return np.sum(X) / len(X)

print("p_hat = ", p_hat(X))

p0 = 0.9
p1 = 0.1

# h0 always reports 0
h0_values = np.zeros_like(X)

# h1 always reports 1
h1_values = np.ones_like(X)

loss_h0 = loss(p0, h0_values)
loss_h1 = loss(p1, h1_values)

print("Loss for h0(X):", loss_h0)
print("Loss for h1(X):", loss_h1)

data = np.random.binomial(n=1,p=p1,size=100)

n = len(data)
ones = np.sum(data)

h0 = ones/n
h1 = 1-h0

def loss(arr1, arr2):
	ans = 0
	for i in range(len(arr1)):
		ans+=abs(arr1[i] - arr2[i])
	return ans

p_values = np.linspace(0,1,1000)

h_values = np.empty(len(p_values))

j=0
for i in p_values: 
	h_arr = np.random.binomial(n=1, p=i, size=100)
	h_values[j] = loss(data, h_arr)
	j+=1

print(h0,h1)
print(p0*h1 + p1*h0)
print(h_values)

plt.title("h(X)")
plt.plot(p_values,h_values)
plt.show()

def posterior(priors, ClassConditionals, i, j):
    numerator = priors[i] * ClassConditionals[i][j] #ClassConditionals[i][j] is fX_Y which means P(X|Y)
    denominator = sum(priors[k] * ClassConditionals[k][j] for k in range(len(priors)))
    return numerator / denominator

def predict(posteriors, X):
    q0x = posteriors[0][X]
    q1x = posteriors[1][X]
    return 0 if q0x >= q1x else 1

def optimal_classifier_and_error_discrete_bayes_posterior(priors, ClassConditionals):
    posteriors = np.zeros_like(ClassConditionals)

    for i in range(len(ClassConditionals)):
        for j in range(len(ClassConditionals[0])):
            posteriors[i][j] = posterior(priors, ClassConditionals, i, j) #Posterior[i][j] is P(Y|X) = P(X|Y)P(Y)/P(X) = fX_Y*P(Y)/P(X)

    error = 0
    typeoneerror = 0
    typetwoerror = 0

    for k in range(len(priors)): # different values of Y
        for l in range(len(ClassConditionals[0])): # different values of X
            if predict(posteriors, l) != k: # if this isn't predicting correctly
                error += priors[k] * ClassConditionals[k][l]
                if predict(posteriors, l) == 1:#positive
                    typeoneerror += priors[k] * ClassConditionals[k][l]
                else: # negative
                    typetwoerror += priors[k] * ClassConditionals[k][l]

    return posteriors, error, typeoneerror, typetwoerror

# Example usage:
priors = [0.9, 0.1]
class_conditionals = [[0.95, 0.05], [0.1, 0.9]] #fX_Y as 00,01,10,11

optimal_classifier, error, typeoneerror, typetwoerror = optimal_classifier_and_error_discrete_bayes_posterior(
    priors, class_conditionals)

print("Optimal Classifier Posteriors:")
print(np.max(optimal_classifier)," coming from ",optimal_classifier)
print("Total Error:", error)
print("Type One Error:", typeoneerror)
print("Type Two Error:", typetwoerror)
print("Risk = ", typeoneerror*priors[0] + typetwoerror*priors[1])

from scipy.stats import norm
from scipy.integrate import quad
def gaussian_distribution(x, mean, std_dev):
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)
mean1 = -1
mean2 = 1
lower_limit1 = -np.inf
lower_limit2 = np.inf
upper_limit = (mean1 + mean2)/2
std_dev = 1
TypeI_Error, _ = quad(gaussian_distribution, lower_limit1, upper_limit, args=(mean1, std_dev))
TypeII_Error, _ = quad(gaussian_distribution, upper_limit,lower_limit2, args=(mean2, std_dev))
Total_Error = TypeI_Error + TypeII_Error
print("Total Error:", Total_Error)
print("Type I Error:", TypeI_Error)
print("Type II Error:", TypeII_Error)
print("Risk = ", TypeI_Error*priors[0] + TypeII_Error*priors[1])