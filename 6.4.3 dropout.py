class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    
    def backward(self,dout):
        return dout * self.mask
from scratch.common.multi_layer_net_extend import MultiLayerNetExtend
from scratch.common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)