import numpy as np
import os
import matplotlib.pyplot as plt

from activations import Sigmoid,Identity,Tanh
from criterion import SoftmaxCrossEntropy
from dataset import Dataset
from perceptron import MLP


class train_test(object):
    """This class trains the network
    
    Params:
        mlp: initialized neural network
        
        dset:list, train, validation, test dataset with list labels
            format [[train_imgs, train_labels],
                [val_imgs, val_labels],
                [test_imgs, test_labels]
                ]
            
        num_epoch:int, number of epochs to train te network for
        batch_size: int, mini batch size
    """
    
    def __init__(self,mlp,dset,num_epochs,batch_size):
        
        #initialize variables
        
        self.mlp=mlp
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        
        self.train,self.val,self.test=dset
        
        self.train_x, self.train_labels=self.train
        
        self.val_x, self.val_labels=self.val
        
        self.test_x, self.test_labels=self.test
        
        self.training_losses = []
        self.training_errors = []
        self.validation_losses = []
        self.validation_errors = []
    
        
    def __fit(self,x,label,loss,error,Flag):
        """Fit the input x mini batch data to labels
        
        Params:
            x: input
            label: ground truth
            loss:int, initilize with 0
            error:int, inintilize with 0
            flag: str, 'train' while training, 'eval' for testing and validation
        
        """
        for b in range(0, len(x), self.batch_size):
            
            if Flag== 'train':
                self.mlp.train()
            elif Flag=='eval':
                self.mlp.eval()
            else:
                self.mlp.eval()
            
            #forward propogation for a given batch 
            self.mlp(x[b:b+self.batch_size])
            #backpropogation for a given batch 
            self.mlp.backward(label[b:b+self.batch_size])
            self.mlp.step()
            
            #running loss and error counter
            loss += self.mlp.get_loss(label[b:b+self.batch_size])
            error += self.mlp.get_error(label[b:b+self.batch_size])
        return loss,error
    
    
    def train_network(self):
        
        """trains the network"""
    
        for epoch in range(self.num_epochs):
            print('\nEpoch: ',epoch)
            
            train_loss = 0
            train_error = 0
            val_loss = 0
            val_error = 0
            num_train = len(self.train_x)
            num_val = len(self.val_x)
            
            #call fit function to train the mini batches over fixed number of epochs            
            train_loss,train_error=self.__fit(self.train_x, self.train_labels, train_loss, train_error, Flag='train')
            
            self.training_losses += [train_loss/num_train]
            self.training_errors += [train_error/num_train]
            print("training loss: ", train_loss/num_train)
            print("training error: ", train_error/num_train)
            
             #call fit function on validation data
            val_loss,val_error=self.__fit(self.val_x, self.val_labels, val_loss, val_error, Flag='eval')
            self.validation_losses += [val_loss/num_val]
            self.validation_errors += [val_error/num_val]
            print("validation loss: ", val_loss/num_val)
            print("validation error: ", val_error/num_val)
            
            
    def test_network(self):
        """
        Test network efficiency for unseen data
        """
        
        print('\nTest data set')
        test_loss = 0
        test_error = 0
        num_test = len(self.test_x)
        
        #call fit function to run trained weights over test dataset        
        test_loss,test_error=self.__fit(self.test_x, self.test_labels,test_loss, test_error, Flag='eval')
        
        test_loss /= num_test
        test_error /= num_test
        print("Test loss: ", test_loss)
        print("Test error: ", test_error)
        
            
    def plots(self):
        """
        Creates loss and error plots during training for infering convergence.
        """
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        #plot 1: Plots training loss and validation loss during training
        ax1.plot(self.training_losses, color='blue', label="training")
        ax1.plot(self.validation_losses, color='red', label='validation')
        ax1.set_title('Loss during training')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.legend()
        
        #plot 2: Plots training error and validation error during training
        ax2.plot(self.training_errors, color='blue', label="training")
        ax2.plot(self.validation_errors, color='red', label="validation")
        ax2.set_title('Error during training')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('error')
        ax2.legend()
        plt.savefig('train_validation loss and error')
        plt.show()
        
        
    def save(self,path):
        """
        Save the trained model paramets
        params:
            path: str, path of file to be saved
        Returns:npz file containing weights and bias
            
        """
        
        self.mlp.save_model(path)
        

def main():
    
    print('Loading data')
    data=Dataset("mnist_train.csv","mnist_test.csv",0.10,10)
    dset=data.create_dataset()
    print('Data loaded')
    
    image_size = 28 # width and length of mnist image
    num_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9, since mnist has 10 classes.
    image_pixels = image_size * image_size
    
    # create hidden layes list, the list length should be equal to number of layers and the numbers should correspond to number of hidden neurons in each layer.
    hiddens = [128,128,64]# this gives 3 hidden layes of size 128,128 and 64 respectively
    
    #since cross entropy is used the last activation layer should be identity.
    # size of activation list should be equal to len(hidden)+1 with last layer as identity for cross entrpoy
    activations = [Sigmoid(), Tanh(), Sigmoid(), Identity()]
    lr = 0.1
    num_epochs = 100
    batch_size = 784
    

    # build your MLP model
    mlp = MLP(
        input_size=image_pixels, 
        output_size=num_labels, 
        hiddens=hiddens, 
        activations=activations, 
        criterion=SoftmaxCrossEntropy(), 
        lr=lr
    )

    # train the neural network
    t=train_test(mlp, dset, num_epochs, batch_size)
    t.train_network()
    
    #create plots
    t.plots()
    
    #test network
    t.test_network()
    
    #save trained weights.
    t.save('weights.npz')
    
if __name__ == "__main__":
    main()