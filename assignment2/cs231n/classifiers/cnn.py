from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # x = np.random.randn(4, 3, 5, 5)
        # w = np.random.randn(2, 3, 3, 3)
        # b = np.random.randn(2,)
        # dout = np.random.randn(4, 2, 5, 5)
        # conv_param = {'stride': 1, 'pad': 1}

        # Conv params
        F = num_filters # 32
        self.F = F # Used at loss()
        K = filter_size # 7
        C, H, W = input_dim
        pad = (K - 1) // 2
        # print("pad", pad) # OK

        self.params['W1'] = np.random.normal(0, weight_scale, (F, C, K, K))
        self.params['b1'] = np.zeros(F)

        # Affine
        HD = hidden_dim # 100
        H, W = H//2, W//2 # Divide 2 for max_pool
        self.params['W2'] = np.random.normal(0, weight_scale, (F * H * W , HD))
        self.params['b2'] = np.zeros(HD)
        C = num_classes # 10
        self.params['W3'] = np.random.normal(0, weight_scale, (HD , C))
        self.params['b3'] = np.zeros(C)

        print("W1, b1", self.params['W1'].shape, self.params['b1'].shape)
        print("W2, b2", self.params['W2'].shape, self.params['b2'].shape)
        print("W3, b3", self.params['W3'].shape, self.params['b3'].shape)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        '''conv - relu - 2x2 max pool - affine - relu - affine - softmax'''

        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        # cache1 = (conv_cache, relu_cache, pool_cache)
        # print("Forward1", out1.shape) # (50, 32, 16, 16)

        out2, cache2 = affine_relu_forward(out1, W2, b2) # Problem W2, b2
        # cache2 = (fc_cache, relu_cache)
        # print("Forward2", out2.shape) # (2, 7)
        
        out3, cache3 = affine_forward(out2, W3, b3)
        # cache3 = x, w, b

        scores = out3

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dx4 = softmax_loss(out3, y)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

        dx3, dw3, db3 = affine_backward(dx4, cache3)
        dw3 += self.reg * self.params['W3']
        grads['W3'] = dw3
        grads['b3'] = db3

        # print("dx3 and out2 shape", dx3.shape, out2.shape)
        # print("dw3, W3 shape", dw3.shape, W3.shape)

        dx2, dw2, db2 = affine_relu_backward(dx3, cache2)
        dw2 += self.reg * self.params['W2']
        grads['W2'] = dw2
        grads['b2'] = db2

        # print("dx2 out1 shape", dx2.shape, out1.shape)
        # print("dw2, W2 shape", dw2.shape, W2.shape)

        N, C, W, H = X.shape
        F = self.F
        # print("input shape", X.shape)
        dx2 = dx2.reshape((N, F, W//2, H//2)) # Problem at C
        dx1, dw1, db1 = conv_relu_pool_backward(dx2, cache1)
        # print("dw1.shape", dw1.shape)
        dw1 += self.reg * self.params['W1']
        grads['W1'] = dw1
        grads['b1'] = db1
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
