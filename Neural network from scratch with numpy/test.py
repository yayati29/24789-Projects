import numpy as np
from zipfile import ZipFile
zf = ZipFile('mnist.zip', 'r')
train=np.loadtxt(zf.open('mnist_train.csv'),delimiter=',')
print(train.shape)
