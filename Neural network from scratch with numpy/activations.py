import numpy as np

class Activation(object):

    """
    Interface for activation functions (non-linearities).
    This is an abstract base class for the other specific activation functions.
    
    Input is state array of neural network and calls forwards function of non-linear activation function.
    
    Params:
        state: input to activation layer, size [batch_size, hidden_layer_size]
        
    Returns:
        forward: f(x) based on specific non-linearity
        
    Derivative: calculates derivative of output from forward function.
    
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        raise NotImplementedError("Forward function not implemented")

    def derivative(self,x):
        raise NotImplementedError("Derivative function not implemented")



class Identity(Activation):

    """
    Identity activation function. Inherits Activation class.
    
    Forwards: f(x)=x
    Derivative; f'(x)=1
    
    Input is state array of size [batch_size, hidden_layer_size]    
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward function of Identity
        
        Params:
            input 'x': array of size [batch_size,hidden_layer_size]
        
        Retruns: 
            f(x)=x, array of size [batch_size,hidden_layer_size]
        
        """
        self.state = x
        return x

    def derivative(self):
        """
        Derivative of Identity function
        
        Params:
            input 'x' : array of size [batch_size,hidden_layer_size]
        
        Retruns: 
            f'(x)=1, array of size [batch_size,hidden_layer_size]
        """
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linear activation function.
    
    Forwards: f(x)= 1/(1+exp(-x))
    
    Derivative; f'(x)= f(x)*(1-f(x))

    Input is state array of size [batch_size, hidden_layer_size] 
    """

    def __init__(self):
        super(Sigmoid, self).__init__()


    def forward(self, x): 
        """
        Forward function of Sigmoid non-linear activation
        
        Params:
            input 'x': array of size [batch_size,hidden_layer_size]
        
        Retruns: 
            f(x)= 1/(1+exp(-x)) , array of size [batch_size,hidden_layer_size]
        
        """
        
        self.state=np.divide(1,(1+np.exp(-x)))
        # print(self.state.shape)
        return self.state

    def derivative(self):
        
        """
        Calculates Derivative of Sigmoid non-linear activation
        
        Params:
            input 'x' : state array from forward function of sigmoid
                        size:[batch_size,hidden_layer_size]
        
        Retruns: 
            f'(x)= f(x)*(1-f(x)): f(x) is the state from forward array.
                                  size:[batch_size,hidden_layer_size].
            
        """
        
        return (self.state*(1-self.state))


class Tanh(Activation):

    """
    Tanh non-linear activation function.
    
    Forwards: f(x)= (exp(x)-exp(-x)) / (exp(x)+exp(-x))
    
    Derivative: f'(x)= 1-(f(x))^2

    Input is state array of size [batch_size, hidden_layer_size] 
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        """
        Forward function of Tanh non-linear activation
        
        Params:
            input 'x': array of size [batch_size,hidden_layer_size]
        
        Retruns: 
            f(x)= (exp(x)-exp(-x)) / (exp(x)+exp(-x)) , array of size [batch_size,hidden_layer_size]
        
        """
        self.state = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.state
        
    def derivative(self):
        """
        Calculates Derivative of Tanh non-linear activation
        
        Params:
            input 'x' : state array from forward function of Tanh
                        size:[batch_size,hidden_layer_size]
        
        Retruns: 
            f'(x)= 1-(f(x))^2: f(x) is the state from forward array.
                                  size:[batch_size,hidden_layer_size].
            
        """
        return 1-self.state**2                          


