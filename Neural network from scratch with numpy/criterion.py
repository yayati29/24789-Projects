import numpy as np

class Criterion(object):

    """
    Interface for loss functions.
    This is an abstract base class for the other specific activation functions.
    
    Calculate loss based on loss function between output from neural network(logits) and ground truth(labels).
    
    Params:
        logits: output from neural network
        labels: ground truth labels
    Returns:
        Loss
    
    """
    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplementedError("Forward function not implemented")

    def derivative(self):
        raise NotImplementedError("Derivative function not implemented")


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss function for multi-label classification
    This class inherits the criterion class.
    
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()


    def forward(self, logits, labels):
        """
        Calculate cross entroy loss
        
        Params:
            logits: output from neural network
            labels: ground truth
        
        """
                
        self.logits=logits
        self.labels=labels
        
        batch_size=self.logits.shape[0]
        
        exp=np.exp(self.logits)
        
        exp_sum=np.sum(exp, axis=1)
        
        self.soft_max= exp /exp_sum[:,None]
        
        
        loss_list=[]
        for i in range(batch_size):
            loss_list.append(self.soft_max[i,np.argmax(self.labels[i])])
            
        
        log_loss = -np.log(loss_list)
        
        return log_loss


    def derivative(self):
        """
        Calculates derivative of cross entroy loss for backpropogation
        
        """

        return self.soft_max-self.labels