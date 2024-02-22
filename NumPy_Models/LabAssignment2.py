import numpy as np # for numerical computation
import pandas as pd # for data manipulation
from sklearn.datasets import fetch_california_housing , load_iris, load_digits # for loading the dataset
from sklearn.preprocessing import Normalizer,OneHotEncoder # for normalizing the data and one hot encoding
from sklearn.model_selection import train_test_split # for splitting the data
from tqdm import trange # for progress bar
import matplotlib.pyplot as plt # for plotting

class MultiplicationLayer : # Layer 1
    """
    Inputs : X ∈ R^(1xd) , W ∈ R^(dxK) # X is the input data, W is the weight matrix
    """
    def __init__(self, X, W) : # This is the constructor of the class
        self.X = X 
        self.W = W 

    def __str__(self,): # This is the string representation of the class
        return " An instance of Muliplication Layer."

    def forward(self):  # This is the forward pass of the layer
        self.Z = np.dot(self.X, self.W) # Z = XW

    def backward(self): # This is the backward pass of the layer
        self.dZ_dW = (self.X).T  # dZ/dW 
        self.dZ_daZ_prev = self.W  # dZ/dX 

class BiasAdditionLayer : # Layer 2
    """
    Inputs : Z ∈ R^(1xK), B ∈ R^(1xK) # Z is the input data, B is the bias matrix
    """
    def __init__(self, Z : np.ndarray , bias : np.ndarray ): # This is the constructor of the class
        self.B = bias
        self.Z = Z
    
    def __str__(self,): # This is the string representation of the class
        return "An instance of Bias Addition Layer."
    
    def forward(self,): # This is the forward pass of the layer
        self.Z = self.Z + self.B #Z = Z + B
    
    def backward(self,): # This is the backward pass of the layer
        self.dZ_dB = np.identity( self.B.shape[1] ) #dZ/dB

class MeanSquaredLossLayer : # Layer 3
    """
    Inputs : Y ∈ R^(1xK) , Y_hat ∈ R^(1xK) # Y is the true output, Y_hat is the predicted output
    # aZ denotes output of previous activation layer 
    """
    def __init__(self, Y : np.ndarray , Y_hat : np.ndarray): # This is the constructor of the class
        self.Y = Y 
        self.aZ = Y_hat 
    
    def __str__(self,): # This is the string representation of the class
        return "An instance of Mean Squared Loss Layer"
    
    def forward(self, ): # This is the forward pass of the layer
        self.L = np.mean( ( self.aZ - self.Y)**2 ) #L = (1/n) * || Y_hat - Y||**2 
        
    def backward(self,): # This is the backward pass of the layer
        self.dL_daZ = (2/len(self.Y))*(self.aZ - self.Y).T   #dL/dY_hat = (2/n)*(Y_hat - Y).T      

class SoftMaxActivation : # Layer 4
    """
    Input : a numpy array Z ∈ R^(1XK)  # Z is the input data
    """
    def __init__(self, Z): # This is the constructor of the class
        self.Z = Z 
        
    def __str__(self,): # This is the string representation of the class
        return "An instance of Softmax Activation Layer"
        
    def forward(self,): # This is the forward pass of the layer
        self.aZ = self.softmax(self.Z) #aZ = softmax(Z).T
    
    def backward(self,): # This is the backward pass of the layer
        self.daZ_dZ = np.diag( self.aZ.reshape(-1) ) - (self.aZ.T)@( (self.aZ))  #daZ/dZ  = diag(aZ) - sZ*transpose(aZ)
        # Shape = (K,K) where K = len( sZ )
    
    @staticmethod # We are making this method static as it does not depend on the object state and only on the input Z and why are we doing this? Because we can call this method without creating an instance of the class
    def softmax(Z : np.ndarray): # This is the softmax function
        max_Z = np.max( Z, axis=1 ,keepdims=True ) #max_Z = max(Z)
        return (np.exp(Z - max_Z ))/np.sum( np.exp(Z - max_Z), axis=1 , keepdims=True) #softmax(Z) = exp(Z - max_Z)/sum(exp(Z - max_Z))
    
class SigmoidActivation : # Layer 5
    """
    Input : a numpy array Z ∈ R^(Kx1) # Z is the input data
    """
    
    def __init__(self,Z ): # This is the constructor of the class
        self.Z = Z 
    
    def __str__(self,): # This is the string representation of the class
        return "An instance of Sigmoid Activation Layer"
    
    def forward(self,): # This is the forward pass of the layer
        self.aZ = self.sigmoid( self.Z )  # aZ = sigmoid( Z )
    
    def backward(self,): # This is the backward pass of the layer
        diag_entries = np.multiply(self.aZ, 1-self.aZ).reshape(-1) #aZ_i*(1-aZ_i)
        self.daZ_dZ = np.diag(diag_entries) #daZ/dZ = diag(aZ_i*(1-aZ_i))
    
    @staticmethod # We are making this method static as it does not depend on the object state and only on the input Z and why are we doing this? Because we can call this method without creating an instance of the class
    def sigmoid( Z : np.ndarray ) : # This is the sigmoid function
        return  1./(1 + np.exp(-Z) ) #sigmoid(Z) = 1/(1 + exp(-Z))
    
class CrossEntropyLossLayer :  # Layer 6
    """
    Inputs : Y ∈ R^(1xK) , Y_pred ∈ R^(1xK) # Y is the true output, Y_pred is the predicted output
    """    
    def __init__(self, Y , Y_pred): # This is the constructor of the class
        self.Y = Y
        self.aZ = Y_pred
        self.epsilon = 1e-40  
        
    
    def __str__(self, ): # This is the string representation of the class
        return "An instance of Cross Entropy Loss Layer"
    
    def forward(self, ): # This is the forward pass of the layer
        self.L = - np.sum( self.Y * np.log(self.aZ+self.epsilon) ) #L = -1 * dot product of Y & log(Y_pred)
        
    def backward(self, ): # This is the backward pass of the layer
        self.dL_daZ = -1*(self.Y/(self.aZ + self.epsilon)).T # dL/dY_pred ∈ R^(Kx1)

class LinearActivation : # Layer 7
    """
    Input : Z ∈ R^(1xn) # Z is the input data
    """
    def __init__(self, Z): # This is the constructor of the class
        self.Z = Z 
        
    def __str__(self,): # This is the string representation of the class
        return "An instance of Linear Activation."
    
    def forward(self, ): # This is the forward pass of the layer
        self.aZ = self.Z  # aZ = Z
    
    def backward(self,): # This is the backward pass of the layer
        self.daZ_dZ = np.identity( self.Z.shape[1] ) #daZ/dZ = I

class tanhActivation: # Layer 8
    """
    Input : a numpy array Z ∈ R^(1xK)
    """

    def __init__(self, Z): # This is the constructor of the class
        self.Z = Z

    def __str__(self,): # This is the string representation of the class
        return "An instance of tanhActivation class."

    def forward(self,): # This is the forward pass of the layer
        self.aZ = np.tanh(self.Z) #aZ = tanh(Z)

    def backward(self,): # This is the backward pass of the layer
        self.daZ_dZ = np.diag(1 - self.aZ.reshape(-1)**2) #daZ/dZ = diag(1 - aZ**2) ∈ R^(KxK)

class ReLUActivation : # Layer 9
    """
    Input : a numpy array Z ∈ R^(1xK)
    """
    def __init__(self, Z): # This is the constructor of the class
        self.Z = Z 
        self.Leak = 0.01
    
    def __str__(self,): # This is the string representation of the class
        return "An instance of ReLU activation"
    
    def forward(self,): # This is the forward pass of the layer
        self.aZ = np.maximum(self.Z,0) #aZ = max(Z,0)
    
    def backward(self,): # This is the backward pass of the layer
        self.daZ_dZ = np.diag( [1. if x>=0 else self.Leak for x in self.aZ.reshape(-1)]) #daZ/dZ = diag( 1 if aZ_i>0 else 0 )

def load_data(dataset_name='california', 
             normalize_X=False, 
             normalize_y=False,
             one_hot_encode_y = False, 
             test_size=0.2): # This function is used to load the dataset
    if dataset_name == 'california' : 
        data = fetch_california_housing() # Load the california housing dataset
    elif dataset_name == 'iris' : 
        data = load_iris() # Load the iris dataset
    elif dataset_name == 'mnist':
        data = load_digits() # Load the mnist dataset
        data['data'] = 1*(data['data']>=8) # Binarize the mnist dataset

    X = data['data'] # X is the input data which we are clipping from the data's dictionary column name 'data'
    y = data['target'].reshape(-1,1) # y is the output data which we are clipping from the data's dictionary column name 'target'
    
    if normalize_X == True : # normalising the input data to make the training process faster
        normalizer = Normalizer() # Create an instance of the normalizer
        X  = normalizer.fit_transform(X) # Normalize the input data
    
    if normalize_y == True : # normalising the output data to make the training process faster
        normalizer = Normalizer() # Create an instance of the normalizer
        y = normalizer.fit_transform(y) # Normalize the output data
    
# normalising makes the training process faster and more accurate because the range of the input and output data is reduced to a smaller range

    if one_hot_encode_y == True: # one hot encode the output data to make it suitable for classification problems
        encoder = OneHotEncoder()
        y = encoder.fit_transform(y).toarray() # One hot encode the output data
        # y = np.eye(3)[y.reshape(-1)]

# one hot encoding is used to convert the output data into a binary matrix because the output data is in the form of a vector and we need to convert it into a binary matrix to make it suitable for classification problems

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size) # Split the data into training and testing data according to the test size
    return X_train, y_train, X_test, y_test # Return the training and testing data

class Layer: # this is the most important class, it is the parent class of all the layers and it is used to create the neural network model given the input and output dimensions and the activation function
                    # It is used to create the neural network model by stacking the layers on top of each other and connecting them to form a neural network model 
    """
    Input - activation : Activation Layer Name ,n_inp : dimension of input ,  n_out :  Number of output neurons 
    """

    def __init__(self, n_inp, n_out, activation_name="linear", seed=42): # This is the constructor of the class which initialises n_inp, n_out, activation_name and seed

        np.random.seed(seed)  # for reproducability of code

        self.n_inp = n_inp # dimension of input
        self.n_out = n_out # Number of output neurons

        # here X and Z denote input and output of the given layer respectively 

        # random initialization of input X  and output Z
        self.X = np.random.random((1, n_inp))   # input
        self.Z = np.random.random((1, n_out))  # output

        # here W and B are initialized with some scaling to avoid over-flow for relu and tanh activation functions for regression problems and for sigmoid and softmax activation functions for classification problems

        # Initialize W & B with some scaling to avoid over-flow
        self.W = np.random.random((n_inp, n_out)) * \
            np.sqrt(2 / (n_inp + n_out)) # weight matrix; scaling is done by np.sqrt(2 / (n_inp + n_out)) which is called as He initialization
        self.B = np.random.random((1, n_out))*np.sqrt(2 / (1 + n_out)) # bias matrix; scaling is done by np.sqrt(2 / (1 + n_out)) which is called as He initialization
        # He initialization is used to initialize the weights of the neural network by scaling the weights according to the number of input and output neurons using np.sqrt(2 / (n_inp + n_out))
        # define multiplication layer, bias addition layer , and activation layer

        self.multiply_layer = MultiplicationLayer(self.X, self.W) # create an instance of the MultiplicationLayer class
        self.bias_add_layer = BiasAdditionLayer(self.B, self.B) # create an instance of the BiasAdditionLayer class

        if activation_name == 'linear':
            self.activation_layer = LinearActivation(self.Z) # create an instance of the LinearActivation class
        elif activation_name == 'sigmoid':
            self.activation_layer = SigmoidActivation(self.Z) # create an instance of the SigmoidActivation class
        elif activation_name == 'softmax':
            self.activation_layer = SoftMaxActivation(self.Z) # create an instance of the SoftMaxActivation class
        elif activation_name == 'tanh':
            self.activation_layer = tanhActivation(self.Z) # create an instance of the tanhActivation class
        elif activation_name == 'relu':
            self.activation_layer = ReLUActivation(self.Z) #create an instance of the ReLUActivation class

        """
        The forward pass works as follows:
        The input X is passed to the multiplication layer which multiplies the input X with the weight matrix
        W to get the output Z and then the output Z is passed to the bias addition layer which adds the bias matrix B to the output Z
        to get the next output Z and then this output Z is passed to the activation layer which applies the activation function to the output Z
        to get the next output Z and then this output Z is the final output of the given layer 
        """
    def forward(self,): # forward pass of the layer
        self.multiply_layer.X = self.X # input to the multiplication layer
        self.multiply_layer.forward() # forward pass of the multiplication layer

        self.bias_add_layer.Z = self.multiply_layer.Z # input to the bias addition layer
        self.bias_add_layer.forward() # forward pass of the bias addition layer

        self.activation_layer.Z = self.bias_add_layer.Z # input to the activation layer
        self.activation_layer.forward() # forward pass of the activation layer

        self.Z = self.activation_layer.aZ  # output of given layer

        """
        The backward pass works as follows:
        The output Z is passed to the activation layer which applies the activation function to the output Z
        to get the derivative of the output Z with respect to the input and then this derivative is passed to the multiplication layer
        which gets the derivative of the previous output Z with respect to the weight matrix W and then this derivative is passed to the bias addition layer
        which gets the derivative of the previous output Z with respect to the bias matrix B and then this derivative is the final derivative of the output Z
        """

    def backward(self,): # backward pass of the layer
        self.activation_layer.backward() # backward pass of the activation layer
        self.bias_add_layer.backward() # backward pass of the bias addition layer
        self.multiply_layer.backward() # backward pass of the multiplication layer

class NeuralNetwork(Layer): # This class is used to create the neural network model by stacking the layers on top of each other and connecting them to form a neural network model
    """
    Input  - layers : list of layer objects , loss_name : Name of loss layer
    """

    # [ "mean_squared", "cross_entropy"]
    def __init__(self, layers, loss_name="mean_squared", learning_rate=0.01, seed=42): # This is the constructor of the class which initialises layers, loss_name, learning_rate and seed
        np.random.seed(seed)

        self.layers = layers # list of layer objects
        self.n_layers = len(layers)  # number of layers
        self.learning_rate = learning_rate # learning rate

        self.inp_shape = self.layers[0].X.shape # input shape
        self.out_shape = self.layers[-1].Z.shape # output shape

        # random initialization of input X  and output Z
        self.X = np.random.random(self.inp_shape)   # input of neural network
        self.Y = np.random.random(self.out_shape)  # output of neural network

        # define loss layer
        if loss_name == "mean_squared":
            self.loss_layer = MeanSquaredLossLayer(self.Y, self.Y) # create an instance of the MeanSquaredLossLayer class
        if loss_name == "cross_entropy":
            self.loss_layer = CrossEntropyLossLayer(self.Y, self.Y) # create an instance of the CrossEntropyLossLayer class

    """
    The forward pass works as follows:
    The input X is passed to the first layer which applies the forward pass to the input X to get the next output Z
    and then this output Z is passed to the next layer which applies the forward pass to this output Z to get the next output Z
    """

    def forward(self,): # forward pass of the neural network
        self.layers[0].X = self.X # input to the first layer
        self.loss_layer.Y = self.Y # true output

        self.layers[0].forward() # forward pass of the first layer
        for i in range(1, self.n_layers): # forward pass of the remaining layers
            self.layers[i].X = self.layers[i-1].Z # input to the next layer
            self.layers[i].forward() # forward pass of the next layer

        self.loss_layer.aZ = self.layers[-1].Z # predicted output
        self.loss_layer.forward() # forward pass of the loss layer

    """
    The backward pass works as follows:
    The predicted output Z is passed to the loss layer which applies the backward pass to the predicted output Z to get the derivative
    of the predicted output Z with respect to the true output and then this derivative is passed to the last layer which applies the
    backward pass to the derivative of the predicted output Z
    """

    def backward(self,): # backward pass of the neural network

        self.loss_layer.Z = self.Y # predicted output
        self.loss_layer.backward() # backward pass of the loss layer
        self.grad_nn = self.loss_layer.dL_daZ # derivative of the predicted output Z with respect to the true output
        for i in range(self.n_layers-1, -1, -1): # backward pass of the layers
            self.layers[i].backward() # backward pass of the layer

            dL_dZ = np.dot(
                self.layers[i].activation_layer.daZ_dZ, self.grad_nn) # dL/dZ
            dL_dW = np.dot(self.layers[i].multiply_layer.dZ_dW, dL_dZ.T) # dL/dW
            dL_dB = np.dot(self.layers[i].bias_add_layer.dZ_dB, dL_dZ).T # dL/dB

            # Update W & B
            self.layers[i].W -= self.learning_rate*dL_dW # update weight matrix
            self.layers[i].B -= self.learning_rate*dL_dB # update bias matrix

            # Update outer_grad
            self.grad_nn = np.dot(
                self.layers[i].multiply_layer.dZ_daZ_prev, dL_dZ) # dL/dZ_prev

            del dL_dZ, dL_dW, dL_dB # delete the variables to free up memory

def createLayers(inp_shape, layers_sizes, layers_activations): # This function is used to add the layers to the neural network model given the input and output dimensions and the activation function
    # This function works as follows: It creates an instance of the Layer class for each layer and appends the layer to the list of layers to create the neural network model 
    layers = [] # list of layers
    n_layers = len(layers_sizes) # number of layers
    layer_0 = Layer(inp_shape, layers_sizes[0], layers_activations[0]) # create an instance of the Layer class
    layers.append(layer_0) # append the layer to the list of layers
    inp_shape_next = layers_sizes[0] # input shape of the next layer
    for i in range(1, n_layers): # create the remaining layers
        layer_i = Layer(inp_shape_next, layers_sizes[i], layers_activations[i]) # create an instance of the Layer class 
        layers.append(layer_i) # append the layer to the list of layers
        inp_shape_next = layers_sizes[i] # input shape of the next layer

    out_shape = inp_shape_next # output shape
    return inp_shape, out_shape, layers # return the input shape, output shape and the list of layers
# Note that this function and the class Neural Network do the same thing but this function is used to create the layers of the neural network model and the class Neural Network is used to create the neural network model
# That is, first we create the layers of the neural network model one by one using this function and then we create the neural network model using the class Neural Network

# stochastic gradient descent is used to train the model and it is a type of gradient descent in which instead of using the entire
# dataset to compute the gradient of the cost function in each iteration, it uses only a randomly chosen sample subset of the data for each iteration.
def SGD_NeuralNetwork(X_train,
                      y_train,
                      X_test,
                      y_test,
                      nn,
                      inp_shape=1, 
                      out_shape=1,
                      n_iterations=1000,
                      task="regression"
                      ): # This function is used to train the neural network model using stochastic gradient descent
    iterations = trange(n_iterations, desc="Training ...", ncols=100) # progress bar

    for iteration, _ in enumerate(iterations): # train the model for each iteration
        randomIndx = np.random.randint(len(X_train)) # randomly choose a sample subset of the data
        X_sample = X_train[randomIndx, :].reshape(1, inp_shape) # input data
        Y_sample = y_train[randomIndx, :].reshape(1, out_shape) # output data

        nn.X = X_sample # initialize the input data to the sample subset of the data
        nn.Y = Y_sample # initialize the output data to the sample subset of the data

        nn.forward()  # Forward Pass
        nn.backward()  # Backward Pass

    # Now we'll run only forward pass for train and test data and check accuracy/error because we have already updated the weights and biases in the backward pass

    if task == "regression": # check the error for regression problems
        
        nn.X = X_train 
        nn.Y = y_train
        nn.forward()
        train_error = nn.loss_layer.L # error for training data
        
        nn.X = X_test
        nn.Y = y_test
        nn.forward()
        test_error = nn.loss_layer.L # error for testing data

        if isinstance(nn.loss_layer, MeanSquaredLossLayer): # check the error for mean squared loss layer
            print("Mean Squared Loss Error (Train Data)  : %0.5f" % train_error)
            print("Mean Squared Loss Error (Test Data)  : %0.5f" % test_error)

    if task == "classification": # check the accuracy for classification problems
        
        nn.X = X_train
        nn.Y = y_train
        nn.forward()
        y_true = np.argmax(y_train, axis=1) 
        y_pred = np.argmax(nn.loss_layer.aZ, axis=1) 
        acc = 1*(y_true == y_pred) # accuracy for training data
        print("Classification Accuracy (Training Data ): {0}/{1} = {2} %".format(
            sum(acc), len(acc), sum(acc)*100/len(acc))) # {0}/{1} = {2} is used to format the output

        nn.X = X_test
        nn.Y = y_test
        nn.forward()
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(nn.loss_layer.aZ, axis=1)
        acc = 1*(y_true == y_pred) # accuracy for testing data
        print("Classification Accuracy (Testing Data ): {0}/{1} = {2} %".format(sum(acc), len(acc), sum(acc)*100/len(acc))) # {0}/{1} = {2} is used to format the output

X_train, y_train, X_test, y_test = load_data('california', normalize_X=True, normalize_y=False, test_size=0.2) # load the data

inp_shape = X_train.shape[1] # input shape
layers_sizes = [1] # number of neurons in each layer
layers_activations = ['linear'] # activation function for each layer

inp_shape, out_shape, layers = createLayers(inp_shape, layers_sizes, layers_activations) # create the layers of the neural network model
loss_nn = 'mean_squared' # loss function

nn = NeuralNetwork(layers, loss_nn, learning_rate=0.1) # create the neural network model

SGD_NeuralNetwork(X_train,y_train,X_test,y_test,nn,inp_shape, out_shape,n_iterations=10000,task="regression") # train the neural network model using stochastic gradient descent

inp_shape = X_train.shape[1] # input shape
layers_sizes = [13,1] # number of neurons in each layer
layers_activations = ['sigmoid','linear'] # activation function for each layer

inp_shape, out_shape, layers = createLayers(inp_shape, layers_sizes, layers_activations) # create the layers of the neural network model
loss_nn = 'mean_squared' # loss function

nn = NeuralNetwork(layers, loss_nn, learning_rate=0.01) # create the neural network model

SGD_NeuralNetwork(X_train,y_train,X_test,y_test,nn,inp_shape, out_shape,n_iterations=1000,task="regression") # train the neural network model using stochastic gradient descent

inp_shape = X_train.shape[1] # input shape
layers_sizes = [13,13,1] # number of neurons in each layer
layers_activations = ['sigmoid','sigmoid','linear'] # activation function for each layer

inp_shape, out_shape, layers = createLayers(inp_shape, layers_sizes, layers_activations) # create the layers of the neural network model
loss_nn = 'mean_squared' # loss function

nn = NeuralNetwork(layers, loss_nn, learning_rate=0.001) # create the neural network model

SGD_NeuralNetwork(X_train,y_train,X_test,y_test,nn,inp_shape, out_shape,n_iterations=1000,task="regression") # train the neural network model using stochastic gradient descent

X_train, y_train, X_test, y_test = load_data('mnist', one_hot_encode_y=True, test_size=0.3) # load the data

inp_shape = X_train.shape[1] # input shape
layers_sizes = [89,10] # number of neurons in each layer
layers_activations = ['tanh','sigmoid'] # activation function for each layer

inp_shape, out_shape, layers = createLayers(inp_shape, layers_sizes, layers_activations) # create the layers of the neural network model
loss_nn = 'mean_squared' # loss function

nn = NeuralNetwork(layers, loss_nn, learning_rate=0.1) # create the neural network model

SGD_NeuralNetwork(X_train,y_train,X_test,y_test,nn,inp_shape, out_shape,n_iterations=10000,task="classification") # train the neural network model using stochastic gradient descent

inp_shape = X_train.shape[1] # input shape
layers_sizes = [89,10] # number of neurons in each layer
layers_activations = ['tanh','softmax'] # activation function for each layer

inp_shape, out_shape, layers = createLayers(inp_shape, layers_sizes, layers_activations) # create the layers of the neural network model
loss_nn = 'cross_entropy' # loss function

nn = NeuralNetwork(layers, loss_nn, learning_rate=0.01) # create the neural network model

SGD_NeuralNetwork(X_train,y_train,X_test,y_test,nn,inp_shape, out_shape,n_iterations=10000,task="classification") # train the neural network model using stochastic gradient descent

# Assuming we are given single channel input and initial filter to be a 3x3 matrix:

def convolutional_layer(zero_pad_input, l_filter):
    inp = zero_pad_input  # input matrix
    l = len(inp)  # length of input matrix
    m = len(l_filter)  # length of filter
    c = len(zero_pad_input)  # size of zero-padded matrix
    s = (c - m) + 1  # to be used for loop for filtering
    out = np.zeros((l, l))  # output after convolution

    # filtering
    for i in range(s): 
        for j in range(s):
            temp = np.zeros((m, m)) # temporary matrix to store the filtered matrix
            row, col = np.indices((m, m)) # indices of the filter matrix
            temp = np.multiply(zero_pad_input[row+i, col+j], l_filter) # element-wise multiplication of input matrix and filter matrix

            out[i][j] = np.sum(temp) # sum of the elements of the temporary matrix
    return out # output after convolution

# filtering is done to extract features from the input matrix and it is done by element-wise multiplication of the input matrix and the filter matrix and then summing the elements of the temporary matrix to get the output after convolution
# filtering helps in reading features from the input matrix and it is done by sliding the filter matrix over the input matrix till the end of the input matrix
def Forward_pass(inp, l_filter): # Forward pass implementation
    l = len(inp) # length of input matrix
    zero_pad_input = np.zeros((l+2, l+2)) # zero-padded input matrix
    zero_pad_input[1:l+1, 1:l+1] = inp # here we are setting the input matrix in the center of the zero-padded input matrix

    f_out = convolutional_layer(zero_pad_input, l_filter) # output after convolution
    return f_out 

def rotateMatrix(mat): # Rotate the matrix by 180 degree
    N = len(mat) # length of the matrix
    rot_mat = np.zeros((N, N)) # rotated matrix
    k = N - 1 # index of the last element
    t1 = 0 # index of the first element
    while (k >= 0 and t1 < 3):
        j = N - 1 
        t2 = 0
        while (j >= 0 and t2 < N):
            rot_mat[t1][t2] = mat[k][j]
            j = j - 1
            t2 = t2 + 1
        k = k - 1
        t1 = t1 + 1

    return rot_mat

# We are rotating the filter matrix by 180 degree to apply convolution to the input matrix to get the gradient of the loss w.r.t input matrix and the gradient of the loss w.r.t filter matrix in the backward pass 

def Backward_pass(inp, output, l_filter): # Backward pass implementation
    l = len(inp) # length of input matrix
    zero_pad_input = np.zeros((l+2, l+2)) # zero-padded input matrix
    zero_pad_input[1:l+1, 1:l+1] = inp # here we are setting the input matrix in the center of the zero-padded input matrix

    grad_filter = convolutional_layer(zero_pad_input, output) # gradient of loss w.r.t filter matrix
    # we can use gradient of filter coefficient matrix to update the filter matrix:
    # filter = filter - learning_rate * gradient of filter coefficient matrix

    # for gradient of loss w.r.t input, we need to rotate the filter by 180° and apply convolution.

    rotated_filter = rotateMatrix(l_filter) # rotate the filter matrix by 180 degree
    zero_pad_output = np.zeros((l+2, l+2)) # zero-padded output matrix
    zero_pad_output[1:l+1, 1:l+1] = output # here we are setting the output matrix in the center of the zero-padded output matrix
    grad_X = convolutional_layer(zero_pad_output, rotated_filter) # gradient of loss w.r.t input matrix

    return grad_filter, grad_X

def flatten(inp_mat): # Flatten the matrix
    flatten_vector = [] # flattened vector

    for i in range(len(inp_mat)):  # number of rows
        for j in range(len(inp_mat[0])):  # number of columns
            flatten_vector.append(inp_mat[i][j]) # append the elements of the matrix to the flattened vector

    flatten_vector = np.array(flatten_vector) # convert the flattened vector to a numpy array
    return flatten_vector 

# flattening is done to convert the matrix into a vector and it is done by appending the elements of the matrix to the flattened vector

class ConvolutionalLayer: # Implementation of Convolutional Layer consist of Convolution  followed by flattening  and Activation operation
    def __init__(self,
                 # inp_shape = (input_channels, input_height, input_width )
                 inp_shape,
                 activation='tanh',
                 # filter_shape = (filter_height, filter_width)
                 filter_shape=(1, 1),
                 lr=0.01,
                 Co=1,
                 seed=42): # This is the constructor of the class which initialises inp_shape, activation, filter_shape, lr, Co and seed where lr is the learning rate and Co is the number of output channels and Ci is the number of input channels

        inp = np.random.rand(*inp_shape) # random input
        np.random.seed(seed) # for reproducability of code
        # Check if filter is valid or NOT by comparing input and filter shape
        assert (inp_shape[1] >= filter_shape[0] and inp_shape[2] >= filter_shape[1]), \
            "Error : Input {} incompatible with filter {}".format(
                inp.shape, filter_shape) # check if the input is compatible with the filter

        self.inp = np.random.rand(*inp_shape) # random input
        self.inp_shape = inp_shape # input shape
        # number of channels in input here denoted as inp

        self.Ci = self.inp.shape[0] # number of input channels
        self.Co = Co # number of output channels
        self.filters_shape = (self.Co, self.Ci,  *filter_shape) # filter shape
        self.out_shape = (self.Co, self.inp.shape[1] - filter_shape[0] + 1, self.inp.shape[2] - filter_shape[1] + 1) # output shape
        self.flatten_shape = np.prod(self.out_shape) # flattened shape
        self.lr = lr # learning rate

        self.filters = np.random.rand(*self.filters_shape) # random filters
        self.biases = np.random.rand(*self.out_shape) # random biases
        self.out = np.random.rand(*self.out_shape) # random output
        self.flatten_out = np.random.rand(1, self.flatten_shape) # random flattened output

        if activation == 'tanh':
            self.activation_layer = tanhActivation(self.out) # create an instance of the tanhActivation class

    """
    The forward pass works as follows:
    The input X is passed to the convolutional layer which applies the forward pass to the input X to get the next output Z
    and then this output Z is passed to the activation layer which applies the activation function to this output Z to get the next output Z
    and then this output Z is the final output of the convolutional layer
    """

    def forward(self, ): # forward pass of the convolutional layer
        self.out = np.copy(self.biases) # output after convolution
        for i in range(self.Co):  # for each output channel
            for j in range(self.Ci): # for each input channel
                self.out[i] += self.convolve(self.inp[j], self.filters[i, j])  # convolution operation

        self.flatten() # flatten the output
        self.activation_layer.Z = self.flatten_out # input to the activation layer
        self.activation_layer.forward() # forward pass of the activation layer

    """
    The backward pass works as follows:
    The output Z is passed to the activation layer which applies the activation function to the output Z
    to get the derivative of the output Z with respect to the input and then this derivative is passed to the convolutional layer
    which gets the derivative of the previous output Z with respect to the input and the filter matrix and then this derivative is the final derivative of the output Z
    """

    def backward(self, grad_nn): # backward pass of the convolutional layer

        self.activation_layer.backward() # backward pass of the activation layer
        loss_gradient = np.dot(self.activation_layer.daZ_dZ, grad_nn) # loss gradient
        # reshape to (Co, H_out, W_out)
        loss_gradient = np.reshape(loss_gradient, self.out_shape) # reshape the loss gradient

        # dL/dKij for each filter  Kij    1<=i<=Ci , 1<=j<=Co
        self.filters_gradient = np.zeros(self.filters_shape) # dL/dKij
        self.input_gradient = np.zeros(self.inp_shape)  # dL/dXj
        self.biases_gradient = loss_gradient  # dL/dBi  = dL/dYi
        padded_loss_gradient = np.pad(loss_gradient, ((0, 0), (self.filters_shape[2]-1, self.filters_shape[2]-1), (self.filters_shape[3]-1, self.filters_shape[3]-1))) # padded loss gradient

        for i in range(self.Co): # for each output channel
            for j in range(self.Ci): # for each input channel
                self.filters_gradient[i, j] = self.convolve(self.inp[j], loss_gradient[i])  # dL/dKij = convolution( Xj, dL/dYi)
                rot180_Kij = np.rot90(np.rot90(self.filters[i, j], axes=(0, 1)), axes=(0, 1))  # rotate the filter matrix by 180 degree
                self.input_gradient[j] += self.convolve(padded_loss_gradient[i], rot180_Kij) # dL/dXj = convolution( dL/dYi, rot180(Kij))

        self.filters -= self.lr*self.filters_gradient # update filters
        self.biases -= self.lr*self.biases_gradient # update biases

    # flattening output to 1 Dimension so it can be fed int neural network

    def flatten(self, ):
        self.flatten_out = self.out.reshape(1, -1) # flatten the output

    # convolutional operation with stride=1, where stride is the number of pixels by which we slide the filter matrix over the input matrix
    def convolve(self, x, y): # convolution operation
        x_conv_y = np.zeros((x.shape[0] - y.shape[0] + 1, x.shape[1] - y.shape[1] + 1)) # output after convolution
        for i in range(x.shape[0]-y.shape[0] + 1): # for each row
            for j in range(x.shape[1] - y.shape[1] + 1): # for each column
                tmp = x[i:i+y.shape[0], j:j+y.shape[1]] # temporary matrix
                tmp = np.multiply(tmp, y) # element-wise multiplication of the input matrix and the filter matrix
                x_conv_y[i, j] = np.sum(tmp) # sum of the elements of the temporary matrix
        return x_conv_y

class CNN : # Implementation of Convolutional Neural Network
    # In this class we are basically generating the input and output for the Convolutional Layer and Neural Network
    
    def __init__(self, 
                convolutional_layer,                 
                nn,                                    
                seed = 42): 

        self.nn = nn # feed forward neural network
        self.convolutional_layer = convolutional_layer  # convolutional layer
        self.X = np.random.rand(*self.convolutional_layer.inp_shape) # random input
        self.Y = np.random.rand(*self.nn.out_shape) # random output
    
    """
    The forward pass works as follows:
    The input X is passed to the convolutional layer which applies the forward pass to the input X to get the next output Z
    and then this output Z is passed to the neural network which applies the forward pass to this output Z to get the next output Z
    """

    def forward(self,): # forward pass of the convolutional neural network
        self.convolutional_layer.inp = self.X # input to the convolutional layer
        self.convolutional_layer.forward() # forward pass of the convolutional layer

        self.nn.X = self.convolutional_layer.activation_layer.aZ # input to the neural network
        self.nn.Y = self.Y  # true output
        self.nn.forward()  # Forward Pass
    
    """
    The backward pass works as follows:
    The predicted output Z is passed to the neural network which applies the backward pass to the predicted output Z to get the derivative
    of the predicted output Z with respect to the true output and then this derivative is passed to the convolutional layer which applies the
    backward pass to the derivative of the predicted output Z
    """

    def backward(self,): # backward pass of the convolutional neural network
        self.nn.backward() # Backward Pass
        self.convolutional_layer.backward( self.nn.grad_nn )  # Backward Pass

def SGD_CNN(X_train,
            y_train,
            X_test,
            y_test,
            cnn,
            inp_shape,
            out_shape,
            n_iterations=1000,
            task="classification"): # This function is used to train the convolutional neural network model using stochastic gradient descent

    iterations = trange(n_iterations, desc="Training ...", ncols=100) # progress bar

    for iteration, _ in enumerate(iterations): # train the model for each iteration
        randomIndx = np.random.randint(len(X_train)) # randomly choose a sample subset of the data
        X_sample = X_train[randomIndx, :].reshape(inp_shape) # input data
        Y_sample = y_train[randomIndx, :].reshape(out_shape) # output data

        cnn.X = X_sample # initialize the input data to the sample subset of the data
        cnn.Y = Y_sample # initialize the output data to the sample subset of the data

        cnn.forward()  # Forward Pass
        cnn.backward()  # Backward Pass

    # We'll run only forward pass for train and test data and check accuracy/error because we have already updated the weights and biases in the backward pass

    if task == "classification": # check the accuracy for classification problems
        X_train = X_train.reshape(-1, 8, 8) # reshape the input data
        y_true = np.argmax(y_train, axis=1) # true output
        acc = 0 # accuracy
        for i in range(len(X_train)): # for each sample in the training data
            cnn.X = X_train[i][np.newaxis, :, :] # input data
            cnn.Y = y_train[i] # true output
            cnn.forward() # forward pass of the convolutional neural network
            y_pred_i = np.argmax(cnn.nn.loss_layer.aZ, axis=1) # predicted output
            if (y_pred_i == y_true[i]): # check if the predicted output is equal to the true output
                acc += 1 # increment the accuracy
        
        print("Classification Accuracy (Training Data ):" + str(acc) + "/" + str(len(y_true)) + " = " + str(acc*100/len(y_true)) + " %" ) #str is used to convert the output to a string

        X_test = X_test.reshape(-1, 8, 8) # reshape the input data
        y_true = np.argmax(y_test, axis=1) # true output
        acc = 0 # accuracy
        for i in range(len(X_test)): # for each sample in the testing data
            cnn.X = X_test[i][np.newaxis, :, :] # input data
            cnn.Y = y_test[i] # true output
            cnn.forward() # forward pass of the convolutional neural network
            y_pred_i = np.argmax(cnn.nn.loss_layer.aZ, axis=1) # predicted output
            if (y_pred_i == y_true[i]): # check if the predicted output is equal to the true output
                acc += 1 # increment the accuracy
        
        print("Classification Accuracy (Testing Data ):" + str(acc) + "/" + str(len(y_true)) + " = " + str(acc*100/len(y_true)) + " %" ) #str is used to convert the output to a string

X_train, y_train, X_test, y_test = load_data('mnist', one_hot_encode_y=True) # load the data

conv_inp_shape = (1,8,8)   # sklearn digit dataset has images of shape 1 x 8 x 8
Co = 16  # 16 channel output 
conv_filter_shape = (3,3) # 3 x 3 filter 
conv_activation = 'tanh' # activation function for the convolutional layer
convolutional_layer = ConvolutionalLayer(conv_inp_shape, 
                                        filter_shape = conv_filter_shape, 
                                        Co = Co, 
                                        activation = conv_activation,
                                        lr = 0.01) # create an instance of the ConvolutionalLayer class
nn_inp_shape = convolutional_layer.flatten_shape # input shape
layers_sizes = [10] # number of neurons in the layer
layers_activations = ['softmax'] # activation function for the layer

nn_inp_shape, nn_out_shape, layers = createLayers(nn_inp_shape, layers_sizes, layers_activations) # create the layers of the neural network model
loss_nn = 'cross_entropy' # loss function

nn = NeuralNetwork(layers, loss_nn, learning_rate=0.01) # create the neural network model

cnn = CNN( convolutional_layer, nn) # create the convolutional neural network model
out_shape =  (1, layers_sizes[-1])  # one_hot encoded ouptut 

SGD_CNN(X_train,y_train,X_test,y_test, cnn,conv_inp_shape, out_shape,n_iterations=5000) # train the convolutional neural network model using stochastic gradient descent