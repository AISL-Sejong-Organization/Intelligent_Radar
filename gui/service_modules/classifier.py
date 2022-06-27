import numpy as np
import copy

class preprocessing():
    def __init__(self):
        pass
    
    def abs(self):
        self.double = self.predata ** 2
        sum = np.sum(self.double, axis = 1)
        root = np.sqrt(sum)
        return np.expand_dims(root, axis=0).T
    
    def rms(self):
        mean = np.mean(self.double, axis = 1)
        root = np.sqrt(mean)
        return np.expand_dims(root, axis=0).T
    
    def max(self):
        maximum = np.max(self.predata, axis = 1)
        return np.expand_dims(maximum, axis = 0).T
    
    def min(self):
        minimum = np.min(self.predata, axis = 1)
        return np.expand_dims(minimum, axis=0).T
    
    def avg(self):
        mean = np.mean(self.predata, axis = 1)
        return np.expand_dims(mean, axis=0).T

    def processedData(self, predata):
        self.predata = copy.deepcopy(predata)
        data = np.append(self.predata, self.max(), axis = 1)
        data = np.append(data, self.min(), axis = 1)
        data = np.append(data, self.avg(), axis = 1)
        data = np.append(data, self.abs(), axis = 1)
        data = np.append(data, self.rms(), axis = 1)
        return data