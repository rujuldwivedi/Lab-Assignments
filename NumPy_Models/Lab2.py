import numpy as np

a=3
b=4
n = 1000
eta = 0.0000001

def func (X,N):
	Y = a*X + b + N	
	return Y

def Inputs(n):
	arrX = np.empty(n)
	arrY = np.empty(n)
	np.random.seed(42)
	for i in range(n):
		X = np.random.uniform(0, 10)
		N = np.random.normal(0, 1)
		arrX[i]=X
		arrY[i]=func(X,N)
	arr = np.array([arrX,arrY])
	return arr

def Y_cap(w,B,X,n):
	y_h = np.empty(n)
	for i in range(len(X)):
		y_h[i] = w*X[i] + B
	return y_h

def meanSquaredLoss(y_hat,y):
	q = y_hat - y
	L = (q.T@q)/2
	return L
		
def gradientDescent(X,y,epochs):
	w = 0
	B = 0
	for i in range(epochs):
		y_hat = Y_cap(w,B,X,n)
		q = 2 * (y_hat - y)
		L = meanSquaredLoss(y_hat,y)
		w = w - eta * X.T @ q 
		B = B - eta * np.sum(q)
		print(L)
	return [w,B]

S = Inputs(n)
weights = gradientDescent(S[0],S[1],10000)
print(weights)