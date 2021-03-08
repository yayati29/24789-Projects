import numpy as np


class MLP(object):

    """
    A multilayer perceptron
    
    Params:
        input_size: int, size of input array
        
        output_size: int, size of output array (size number of labels)
        
        hidden=list, list of hidden layer dimensions 
                eg.[128,128,64]-> to create a 3 layer network with hidden layers of size 128,128 and 64 respectively
        
        activations: list, list of activations layers, length of list should be len(hidden)+1
        
        criterion: Loss function
        
        lr:float, learning rate
        
    """


    def __init__(self, input_size, output_size, hiddens, activations, criterion, lr):

        
        self.train_mode = True
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        

        #class attributes
        self.nn_dim = [input_size] + hiddens + [output_size]
        
        # list containing Weight matrices of each layer, random initialization
        self.W = [self.__random_normal_weight_init(self.nn_dim[i], self.nn_dim[i+1]) for i in range(self.nlayers)]
        
        # list containing derivative of Weight matrices of each layer,should be a np.array
        self.dW = [np.zeros_like(weight) for weight in self.W]
        
        # list containing bias vector of each layer, 0 initialization
        self.b = [self.__zeros_bias_init(self.nn_dim[i+1]) for i in range(self.nlayers)]
        
        # list containing derivative of bias vector of each layer
        self.db = [np.zeros_like(bias) for bias in self.b]

    def __random_normal_weight_init(self,d0, d1):
        """random initialization of weigths"""
        return np.random.randn(d0,d1)
    
    def __zeros_bias_init(self,d):
        """For 0 initialization of bias"""
        return np.zeros((d))
    
    
    
    def forward(self, x):
        """
        forward layer
        params:
            x: state array input for the hidden layer
        return: updated state array after nonlinear activation
        """
        lout=[]
        a=x
        for i in range(self.nlayers):          
            lout.append(a)
            # non linear activation 
            a = self.activations[i](a @ self.W[i]+ self.b[i])
        self.lin_out=lout
        self.out = a
        
        
    
    def zero_grads(self):
        """
        set dW(weight derivative) and db(bias derivative) to be zero
        """
        self.dW=np.multiply(0,self.W)
        self.db=np.multiply(0,self.b)
        
    
    def step(self):     
        """update the W and b on each layer"""
        self.W = (self.W-np.multiply(self.lr,self.dW))
        self.b = (self.b-np.multiply(self.lr,self.db))

    def backward(self, labels):
        """Backward propogation"""
        if self.train_mode:
            # calculate dW and db only under training mode
            self.criterion(self.out,labels)
            act_out=self.criterion.derivative()

            for i in range(self.nlayers-1,-1,-1):
                act_out=np.multiply(act_out, self.activations[i].derivative())
                self.dW[i]=np.dot(np.transpose(self.lin_out.pop()),act_out)/np.shape(act_out)[0]
                self.db[i]=np.sum(act_out,axis=0)/np.shape(act_out)[0]
                act_out=np.dot(act_out,np.transpose(self.W[i]))
        return

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        """training mod"""
        self.train_mode = True

    def eval(self):
        """ evaluation mode"""
        self.train_mode = False

    def get_loss(self, labels):
        """return the current loss value given labels"""
        return np.sum(self.criterion(self.out, labels), axis=0)

    def get_error(self, labels):
        """return the number of incorrect preidctions gievn labels"""
        return np.sum((np.argmax(labels, axis=-1)!= np.argmax(self.out, axis=-1)))

    def save_model(self, path):
        """save the parameters of MLP (do not change)"""
        np.savez(path, self.W, self.b)
