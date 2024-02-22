import numpy as np
import matplotlib.pyplot as plt

def func(x, a, b, N):
    Y = a * x + b + N
    return Y

def Inputs(a, b, n):
    arrX = np.empty(n)
    arrY = np.empty(n)
    np.random.seed(42)
    for i in range(n):
        X = np.random.uniform(0, 10)
        N = np.random.normal(0, 1)
        arrX[i] = X
        arrY[i] = func(X, a, b, N)
    arr = np.array([arrX, arrY]) 
    return arr

def y_hat(w, b, x, n): 
    y = np.empty(n)
    for i in range(len(x)):
        y[i] = w * x[i] + b
    return y

def Loss(y, y_hat):
    q = y_hat - y
    L = np.sum(q**2) / (2 * len(y))
    return L

def gradientDescent(x, y, eta, epochs):
    w = 0
    b = 0
    n = len(y)
    for i in range(epochs):
        y_pred = y_hat(w, b, x, n)
        q = 2 * (y_pred - y)
        L = Loss(y, y_pred)
        w = w - eta * np.dot(x, q) / n
        b = b - eta * np.sum(q) / n
        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {L}")
    return [w, b]

def plot_regression_line(x, y, w, b):
    plt.scatter(x, y, color='blue', marker='o', label='Data points')
    plt.plot(x, y_hat(w,b,x,1000), color='red', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

S = Inputs(3, 4, 1000)
weights = gradientDescent(S[0], S[1], 0.001, 10000)
print(weights)
plot_regression_line(S[0], S[1], weights[0], weights[1])