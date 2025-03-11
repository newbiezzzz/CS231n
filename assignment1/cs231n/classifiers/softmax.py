from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import math


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # x = 500 * 3072, W = 3072 * 10
    scores = X.dot(W)
    scores -= np.max(scores)

    num_train = X.shape[0]
    dim_train = X.shape[1]
    num_class = W.shape[1]

    for i in xrange(num_train):
      denominator = 0.0
      for j in xrange(num_class):
        denominator += math.exp(scores[i,j])
      
      loss += -1 * math.log(math.exp(scores[i,y[i]]) / denominator)
      # loss += -1 * scores[i,y[i]] + math.log(denominator)
      # Derivative of wyi*x + log(e^wj1*x + e^wj2*x + ..)
      # dwyi = x
      # dwj = xe^wjx/(e^wjx + e^wj2a + ...)
      for j in xrange(num_class):
        dW[:, j] += 1/denominator * math.exp(scores[i,j]) * X[i]
        if j == y[i]:
          dW[:, j] -= X[i]
    
    loss /= num_train
    dW /= num_train

    # Reg
    loss += reg * np.sum(W * W)
    dW += reg * W * 2
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    scores -= np.max(scores)
    exp_scores = np.exp(scores)

    num_train = X.shape[0]
    # dim_train = X.shape[1]
    num_class = W.shape[1]

    denominator = np.sum(exp_scores, axis = 1)
    # denominator = exp_scores.sum(axis = 1)
    
    numerator = exp_scores[range(num_train), y]

    # numerator = np.zeros(num_train)
    #for count, idx in enumerate(y):
    #  numerator[count] = exp_scores[count,idx]

    ##############################################
    

    loss = np.sum(np.log(numerator / denominator) * -1)

    # Derivative of wyi*x + log(e^wj1*x + e^wj2*x + ..)
    # dwyi -= x
    # dwj = xe^wj1x/(e^wj1x + e^wj2x + ...)
    # dW = 3072 * 10
    # X.T = 3072 * 500 scores = 500 * 10
    correct_scores = np.zeros_like(scores)
    correct_scores[range(num_train), y] = -1
    dWyi = X.T.dot(correct_scores)

    other_scores = np.zeros_like(scores)
    rdenoms = np.reshape(denominator, (-1,1))
    scores = scores 
    dWj = (X.T).dot(exp_scores/rdenoms)
    dW = dWyi + dWj
    
    loss /= num_train
    dW /= num_train

    dW += reg * W 
    loss += reg * np.sum(W * W) 
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
