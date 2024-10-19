from numpy import log, exp


class Normalizer():
    """
    2 aims of Normalizer class:

        1) To apply logarithm of data to make data more dispersed and not concentrated 
        in low values, which improves data distinguishability.

        2) To apply min-max scaler to increase the speed of convergence of neural network
    
    Methods:

        normalize: returns normalized data: take logarithm of data and then apply 
        min-max normalization to get data in [0,1] interval

        denormalize: returns initial values from normalized data: apply the inverse 
        of min-max normalization and take exponent    
    """

    def __init__(self):
        self.min = None
        self.max = None


    def normalize(self, data):
        data_ = data
        
        # Logarithm
        data_ = log(data_)

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
        data_ = exp(data_)
        
        return data_