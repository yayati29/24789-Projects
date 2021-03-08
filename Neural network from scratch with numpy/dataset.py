import numpy as np

class Dataset(object):
    """This class reads train and test csv files and splits train dataset into train and validation based on validation_split ratio.
    It also creates one-hot-encodings for labels
    
    Params:
        inputs:
        train_data_set: string, train file path.
        test_data_set: string, test file path.
        val_split: float, ratio of chunk of train used for validation.
        number_labels: int, number of total labels for classification.
    
    Call create_dataset to generate train, test, validation dataset from given file
    
    
    """
    
    
    def __init__(self,train_data_set,test_data_set,val_split,number_labels):
        
        self.val_split=val_split
        self.number_labels=number_labels
        
        self.train_data=np.loadtxt(train_data_set, delimiter=",")
        self.test_data = np.loadtxt(test_data_set, delimiter=",")
        
        self.__train_val_split()
        
    
    def __train_val_split(self):
        """
        split validation from train
        """
        
        #force the ratio to be int
        val_index_split=int(self.val_split*self.train_data.shape[0])
        
        #split validation from train
        self.val_data=np.asfarray(self.train_data[:val_index_split])
        
        #updated train_data
        self.train_data=np.asfarray(self.train_data[val_index_split:])
        
        
        
    def get_one_hot(self,in_array, one_hot_dim):
        
        """create on hot encoding of the input vectors
        Params:
            input:
            in_array: array of labels
        
        Returns:
            out_array: one hot encoded array from input labels
            
        """
        
        dim = in_array.shape[0]
        out_array = np.zeros((dim, one_hot_dim))
        for i in range(dim):
            idx = int(in_array[i])
            out_array[i, idx] = 1
            
        return out_array
        
    def create_dataset(self):
        """
        Create train,validation, test data.
        The input data is rescaled from 0-1 since the initial input is image array in range 0-255
        
        Params:
            train_data and validation data after splitting
            test data from file
        
        Returns:
            dataset=list, 
            
            [[train_imgs, train_labels],
                [val_imgs, val_labels],
                [test_imgs, test_labels]
                ]
        
        """
        
        # rescale image from 0-255 to 0-1
        fac = 1.0 / 255
        train_imgs=np.asfarray(self.train_data[:, 1:]) * fac
        val_imgs=np.asfarray(self.val_data[:, 1:]) * fac
        test_imgs = np.asfarray(self.test_data[:, 1:]) * fac
        
        #isolate labels from data
        train_labels = np.asfarray(self.train_data[:, :1])
        val_labels=np.asfarray(self.val_data[:, :1])
        test_labels = np.asfarray(self.test_data[:, :1])
        
        #create one hot encoding for labels
        train_labels=self.get_one_hot(train_labels, self.number_labels)
        val_labels=self.get_one_hot(val_labels, self.number_labels)
        test_labels=self.get_one_hot(test_labels, self.number_labels)
        
        return [[train_imgs, train_labels],
                [val_imgs, val_labels],
                [test_imgs, test_labels]
                ]