import numpy as np

def generate_data_set(n,a,b,c):
    X = np.random.uniform(0,5,n)
    Y = a*(X**2) + b*X + c + np.random.normal(0,1,n)
    return X,Y

print(generate_data_set(3,100,10,1))

def g_D(X,Y,hypothesis):
    if hypothesis == 1:#constant hypothesis
        return np.mean(Y)
    elif hypothesis == 2:#linear hypothesis
        X_bar = np.mean(X)
        Y_bar = np.mean(Y)
        a = np.sum((X-X_bar)*(Y-Y_bar))/np.sum((X-X_bar)**2)
        b = Y_bar - a*X_bar
        return a*X + b
    elif hypothesis == 3:#quadratic hypothesis
        X_bar = np.mean(X)
        Y_bar = np.mean(Y)
        a = (np.sum((X-X_bar)*(Y-Y_bar)))/(np.sum((X-X_bar)**2))
        b = Y_bar - a*X_bar
        c = np.mean(Y - b - a*X)
        return a*X**2 + b*X + c
    
def g_X_bar(X,Y,hypothesis):
    g_X_bar_ = 0
    for i in range(100):
        X,Y = generate_data_set(3,100,10,1)
        g_X_bar_ += g_D(X,Y,hypothesis)
    return g_X_bar_/100

def ED_E_out(X,Y,hypothesis):
    g_X_bar_ = g_X_bar(X,Y,hypothesis)
    f_X = 100*(X**2) + 10*X + 1
    return np.mean((f_X - g_X_bar_)**2) + np.mean((g_X_bar_ - g_D(X,Y,hypothesis))**2) + 1

X,Y = generate_data_set(3,100,10,1)

print('ED_E_out for constant hypothesis:',ED_E_out(X,Y,1))
print('ED_E_out for linear hypothesis:',ED_E_out(X,Y,2))
print('ED_E_out for quadratic hypothesis:',ED_E_out(X,Y,3))
print("So the best hypothesis is the constant hypothesis")
print('\n')

def generate_data_set_sin(n):
    X = np.random.uniform(-1,1,n)
    Y = np.sin(np.pi*X) + np.random.normal(0,0.1,n)
    return X,Y

def g_D_sin(X,Y,hypothesis):
    if hypothesis == 1:#constant hypothesis
        return np.mean(Y)
    elif hypothesis == 2:#linear hypothesis
        X_bar = np.mean(X)
        Y_bar = np.mean(Y)
        a = np.sum((X-X_bar)*(Y-Y_bar))/np.sum((X-X_bar)**2)
        b = Y_bar - a*X_bar
        return a*X + b
    
def g_X_bar_sin(X,Y,hypothesis):
    g_X_bar_ = 0
    for i in range(100):
        X,Y = generate_data_set_sin(2)
        g_X_bar_ += g_D_sin(X,Y,hypothesis)
    return g_X_bar_/100

def ED_E_out_sin(X,Y,hypothesis):
    g_X_bar_ = g_X_bar_sin(X,Y,hypothesis)
    f_X = np.sin(np.pi*X)
    return np.mean((f_X - g_X_bar_)**2) + np.mean((g_X_bar_ - g_D_sin(X,Y,hypothesis))**2) + 1

X,Y = generate_data_set_sin(2)

print('ED_E_out for constant hypothesis:',ED_E_out_sin(X,Y,1))
print('ED_E_out for linear hypothesis:',ED_E_out_sin(X,Y,2))
print("So the best hypothesis is the constant hypothesis")