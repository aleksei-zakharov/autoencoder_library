import numpy as np


class Normalizer():
    def __init__(self):
        self.min = None
        self.max = None


    def normalize(self, data):
        data_ = data
        
        # Logarithm
        # data_ = np.log(data_)

        # Minmax scaler
        self.min = data_.min()
        self.max = data_.max()
        data_ = (data_ - self.min) 
        data_ = data_ / (self.max - self.min)
        
        return data_


    def denormalize(self, data):
        data_ = data

        # Rescale MINMAX
        data_ = data_ * (self.max - self.min)
        data_ = data_ + self.min
        
        # The action opposite to the logarithm in normalize function above
        # data_ = np.exp(data_)
        
        return data_