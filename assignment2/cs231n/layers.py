from builtins import range
from types import WrapperDescriptorType
from typing import NamedTuple
import numpy as np
from numpy.ma.core import expand_dims


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # print("x_flat shape:", x.shape)
    # print("w shape:", w.shape)
    # print("b shape:", b.shape)
    x = x.reshape(x.shape[0], -1) #Flatten
    out = np.dot(x, w) + b

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # dout = (2, 4), dx = (2,3), dw = (3,4), db = (4)
    dx = np.dot(dout, w.T)
    dw = np.dot(x.T, dout)
    db = np.sum(dout, axis=0)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.where(x > 0, dout, 0)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Loss
    shift_x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shift_x)
    sum_exp = np.sum(exp_x, axis=1, keepdims=True)
    loss_ij = -shift_x + np.log(sum_exp) # log_probs
    N = x.shape[0]
    loss = np.sum(loss_ij[np.arange(N), y]) / N

    # dx
    probs = exp_x / sum_exp
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx = dx / N

    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute batch mean and variance
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)

        # Normalize the input
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)

        # Scale and shift
        out = gamma * x_hat + beta

        # Update running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        # Store values needed for backward pass
        cache = {}
        cache["x"] = x
        cache["x_hat"] = x_hat
        cache["mean"] = sample_mean
        cache["var"] = sample_var
        cache["gamma"] = gamma
        cache["beta"] = beta
        cache["eps"] = eps

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        # Scale and shift
        out = gamma * x_hat + beta 
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x = cache["x"] 
    x_hat = cache["x_hat"]
    mean = cache["mean"]
    var = cache["var"]
    gamma = cache["gamma"] 
    beta = cache["beta"]
    eps = cache["eps"]
    N, D = x.shape

    dx_hat = dout * gamma
    # print(dx_hat.shape, x_hat.shape)
    dvar = np.sum(dx_hat * (x - mean) * -0.5 * (var + eps)**(-3/2), axis=0)
    # print(dvar.shape, var.shape)
    dmean = np.sum(dx_hat * -1/np.sqrt(var + eps), axis = 0) + dvar * np.sum(-2*(x-mean) / N, axis=0)
    # print(dmean.shape, mean.shape)
    dx = (dx_hat * 1/np.sqrt(var + eps)) + (dvar*2*(x - mean)/N) +  (dmean/N)

    dgamma = np.sum(dout * x_hat, axis = 0)
    dbeta = np.sum(dout, axis = 0)



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x = cache["x"] 
    x_hat = cache["x_hat"]
    mean = cache["mean"]
    var = cache["var"]
    gamma = cache["gamma"] 
    beta = cache["beta"]
    eps = cache["eps"]
    sigma = np.sqrt(var + eps)
    N, D = x.shape

    dx_hat = dout * gamma
    dsigma = np.sum(dx_hat * (x - mean)/sigma**(2), axis=0)
    dvar = dsigma*(-1/(2*sigma))
    dmean = np.sum(-2 * (x - mean), axis=0) / N + np.sum(dx_hat * -1/ sigma, axis=0)
    dx = (dvar * 2 * (x - mean) / N) + (dmean / N) + dx_hat / sigma

    dgamma = np.sum(dout * x_hat, axis = 0)
    dbeta = np.sum(dout, axis = 0)

    # Step-by-step gradient computation
    # dx_hat = dout * gamma  # Gradient of x_hat
    # dvar = np.sum(dx_hat * (x - mean) * -0.5 * (sigma**-3), axis=0)  # Gradient of variance
    # dmean = (
    #     np.sum(dx_hat * -1 / sigma, axis=0)  # Direct gradient of mean
    #     + dvar * np.sum(-2 * (x - mean), axis=0) / N  # Indirect effect of variance on mean
    # )
    # dx = (
    #     dx_hat / sigma  # Gradient of x_hat through normalization
    #     + dvar * 2 * (x - mean) / N  # Gradient of variance through x
    #     + dmean / N  # Gradient of mean through x
    # )

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Compute batch mean and variance
    sample_mean = np.mean(x, axis=1, keepdims=True)
    sample_var = np.var(x, axis=1, keepdims=True)
    # print(sample_mean.shape, sample_var.shape, x.shape)
    # Normalize the input
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)

    # Scale and shift
    out = gamma * x_hat + beta

    # Store values needed for backward pass
    cache = {}
    cache["x"] = x
    cache["x_hat"] = x_hat
    cache["mean"] = sample_mean
    cache["var"] = sample_var
    cache["gamma"] = gamma
    cache["beta"] = beta
    cache["eps"] = eps


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x = cache["x"] 
    x_hat = cache["x_hat"]
    mean = cache["mean"]
    var = cache["var"]
    gamma = cache["gamma"] 
    beta = cache["beta"]
    eps = cache["eps"]
    sigma = np.sqrt(var + eps)
    N, D = x.shape

    dx_hat = dout * gamma
    # print(dx_hat.shape, x_hat.shape)
    dsigma = np.sum(dx_hat * (x - mean)/sigma**(2), axis=1, keepdims=True)
    # print(dsigma.shape, sigma.shape)
    dvar = dsigma*(-1/(2*sigma))
    # print(dvar.shape, var.shape)
    dmean = np.sum(-2 * (x - mean), axis=1, keepdims=True) / D + np.sum(dx_hat * -1/ sigma, axis=1, keepdims=True)
    # print(dmean.shape, mean.shape)
    # print(dvar.shape, dmean.shape, dx_hat.shape)
    dx = (dvar * 2 * (x - mean) / D) + (dmean / D) + dx_hat / sigma

    dgamma = np.sum(dout * x_hat, axis = 0)
    dbeta = np.sum(dout, axis = 0)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        mask = (np.random.rand(*x.shape) < p) / p 
        out = x * mask # drop
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = x
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dx = mask * dout
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    s = conv_param["stride"]
    p = conv_param["pad"]
    x_pad = np.pad(x, pad_width=((0, 0), (0, 0), (p, p), (p, p)), mode='constant', constant_values=0)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_out = 1 + (H + 2 * p - HH) // s
    W_out = 1 + (W + 2 * p - WW) // s

    im2col = np.zeros((N, C * HH * WW, H_out * W_out)) 
    for i in range(H_out):  # Loop over height
      for j in range(W_out):  # Loop over width
        start_i = i * s 
        start_j = j * s
        patch = x_pad[:, :, start_i:start_i+HH, start_j:start_j+WW].reshape(N, -1)
        im2col[:, :, i * W_out + j] = patch
    
    w_mat = w.reshape(F, -1)
    out = np.zeros((N, F, H_out * W_out))
    for i in range(N):
      out[i] = np.dot(w_mat, im2col[i])

    out = out.reshape(N, F, H_out, W_out) + b.reshape(1, -1, 1, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Cache
    (x, w, b, conv_param) = cache

    # Initialize gradients
    dx = np.zeros(x.shape)
    # dw = np.zeros(w.shape) # (2, 3, 3, 3)
    db = np.ones(b.shape) * np.sum(dout, axis=(0, 2, 3))

    s = conv_param["stride"]
    p = conv_param["pad"]
    x_pad = np.pad(x, pad_width=((0, 0), (0, 0), (p, p), (p, p)), mode='constant', constant_values=0)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_out = 1 + (H + 2 * p - HH) // s
    W_out = 1 + (W + 2 * p - WW) // s

    # Fill in im2col
    im2col = np.zeros((N, C * HH * WW, H_out * W_out)) 
    for i in range(H_out):  # Loop over height
      for j in range(W_out):  # Loop over width
        start_i = i * s 
        start_j = j * s
        patch = x_pad[:, :, start_i:start_i+HH, start_j:start_j+WW].reshape(N, -1)
        im2col[:, :, i * W_out + j] = patch
    
    # w_mat = w.reshape(F, -1)
    dout_reshaped = dout.reshape(N, F, -1) 
    out = np.zeros((N, F, H_out * W_out))
    dw = np.zeros((F, C * HH * WW))

    # Pad dout
    pad = HH - 1 # filter height or width - 1 
    dilation = s - 1 # stride - 1
    H_new = H_out + (H_out - 1) * dilation
    W_new = W_out + (W_out - 1) * dilation 
    dout_dilated = np.zeros((N, F, H_new, W_new))
    dout_dilated[:, :, ::s, ::s] = dout 
    # Pad dout with (kernel_size - 1) on all sides
    dout_padded = np.pad(dout_dilated, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    N_pad, K_pad, H_pad, W_pad = dout_padded.shape
    Hpad_dout = 1 + (H_pad - HH) 
    Wpad_dout = 1 + (W_pad - WW) 
    # Make dout into cols
    dout_cols = np.zeros((N, F, HH * WW, Hpad_dout*Wpad_dout ))  # (4, 2, 9, 49)
    print("Hpad_out",Hpad_dout)
    print("dout_padded",dout_padded.shape) 
    print("w", w.shape)
    print("dout_cols.shape",dout_cols.shape)
    # 49 Loops
    index = 0
    for i in range(Hpad_dout):  # Loop over height
      for j in range(Wpad_dout):  # Loop over width
        start_i = i 
        start_j = j 
        patch = dout_padded[:, :, start_i:start_i+HH, start_j:start_j+WW].reshape(N, F, -1)
        if index == 0:
          print("patch-shape",patch.shape)
          pass # debug
          # print("patch-shape", patch.shape)
        # print("patch-shape", patch.shape) # (4, 2, 9)
        dout_cols[:, :, :, index] = patch
        index = index + 1

    # dout_cols = # (4, 2, 9, 49)
    w_flip = np.flip(w, axis=(2, 3)) # (2, 3, 3, 3)
    # w_flip = w (must be flipped)
    w_flip = w_flip.reshape((F, C, -1)) # (2, 3, 9)
    # w_flip = w_flip.transpose(0, 2, 1)  # (2, 9, 3)
    print("w_flip",w_flip.shape)
    dx_pad = np.zeros(x_pad.shape) # (4, 3, 7, 7) 
    print("dx_pad shape", dx_pad.shape)
    dx_pad = dx_pad.reshape((N, C, -1)) # (4, 3, 49)
    print("dx_pad reshaped", dx_pad.shape)
    for n in range(N):
      for f in range(F):
        # dx = dout * w (NOT SURE HOW TO IMPLEMENT, subbed with np.einsum)
        # (4, 3, 49) = (4, 2, 9, 49) * (2, 3, 9) -> (4, 3, 49)
        # dx_pad[n] += np.dot(dout_cols[n, f], w_flip[f].T)
        pass

    # (4, 2, 9, 49) * (2, 3, 9) -> (4, 3, 49)
    dx_pad = np.einsum('nfhw,fch->ncw', dout_cols, w_flip)
    print("dx_pad.shape",dx_pad.shape)
    dx_pad = dx_pad.reshape((N, C, Hpad_dout, Wpad_dout))
    print("dx_pad.reshape",dx_pad.shape)
    dx = dx_pad[:, :, p:-p, p:-p]
    print("dx.cropped shape",dx.shape)


    print("dw",dw.shape)
    print("im2col", im2col.shape)
    print("out.shape", out.shape)
    # Convolve, dw = dout * x
    for i in range(N):
      # print(dout[i].shape, im2col[i].T.shape)
      dw += np.dot(dout_reshaped[i], im2col[i].T) 
    dw = dw.reshape(F, C, HH, WW)

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    Hprime = 1 + (H - pool_height) // stride
    Wprime = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, Hprime, Wprime))

    index = 0
    for i in range(Hprime):  # Loop over height
      for j in range(Wprime):  # Loop over width
        start_i = i * stride 
        start_j = j * stride
        patch = x[:, :, start_i:start_i+pool_height, start_j:start_j+pool_width]
        patch = np.max(patch, axis=(2,3))
        out[:, :, i, j] = patch
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    Hprime = 1 + (H - pool_height) // stride
    Wprime = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, Hprime, Wprime))
    dx = np.zeros_like(x)

    index = 0
    for i in range(Hprime):  # Loop over height
      for j in range(Wprime):  # Loop over width
        start_i = i * stride 
        start_j = j * stride
        end_i = start_i + pool_height
        end_j = start_j + pool_width
        patch = x[:, :, start_i: end_i , start_j: end_j].reshape((N, C, -1))
        # print("patch-shape", patch.shape) # (3, 2 ,4)
        max_idx_flat = np.argmax(patch, axis=2)
        # print("max_idx-flat-shape",max_idx_flat.shape) # (3, 2)

        # Convert flattened index to 2D (row, col) coordinates
        max_idx_i, max_idx_j = np.unravel_index(max_idx_flat, (pool_height, pool_width))
        # print("max-idx-ij", max_idx_i.shape, max_idx_j.shape)  # (3, 2) (3, 2)
        # We use np.arange(N)[:, None] to broadcast over batch and channel to match (3, 2)
        dx[np.arange(N)[:, None], np.arange(C), start_i + max_idx_i, start_j + max_idx_j] = dout[:, :, i, j]
        # dx[:, :, start_i + max_idx_i, start_j + max_idx_j] = dout[:, :, i, j]




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape # (2, 3, 4, 5)
    x_t = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)
    out_t, cache = batchnorm_forward(x_t, gamma, beta, bn_param)
    out = out_t.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout_t = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)
    dx_t, dgamma, dbeta = batchnorm_backward(dout_t, cache)
    dx = dx_t.reshape(N, H, W, C).transpose(0, 3, 1, 2)



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape # (2, 6, 4, 5)
    S = C // G # C dividing G splited to S
    # Reshape x to (N*G, S, H, W)
    # Tranpose x to (N*G, H, W, S) -> Reshape (N*G*H*W, S)
    # x.reshape(2*2, 3, 4, 5) -> (4, 4, 5, 3) -> (80, 3)
    x_rtr = x.reshape(N*G, S, H, W).transpose(0, 2, 3, 1).reshape(N*G*H*W, S)
    # print("x-rtr.shape",x_rtr.shape) # (80, 3)

    mean = np.mean(x_rtr, axis=1, keepdims=True)
    var = np.var(x_rtr, axis=1, keepdims=True)
    # print(mean.shape, var.shape) # (80, 1) (80, 1) 
    # # Normalize the input
    x_hat = (x_rtr - mean) / np.sqrt(var + eps)
    # print("x_hat.shape", x_hat.shape) # (80, 3)
    x_hat = x_hat.reshape(N*G, H, W, S)
    # print("x_hat.shape", x_hat.shape) # (4, 4, 5, 3)
    x_hat = x_hat.transpose(0, 3, 1, 2)
    # print("x_hat.shape", x_hat.shape) # (4, 3, 4, 5)
    # Reshape back to x.shape
    x_hat = x_hat.reshape(N, S*G, H, W)
    # print("x_hat.shape", x_hat.shape) # (2, 6, 4, 5)

    # # Scale and shift
    out = gamma * x_hat + beta

    # # Store values needed for backward pass
    cache = {}
    cache["x"] = x
    cache["x_hat"] = x_hat
    cache["mean"] = mean
    cache["var"] = var
    cache["gamma"] = gamma
    cache["beta"] = beta
    cache["eps"] = eps
    cache["G"] = G


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x = cache["x"] 
    x_hat = cache["x_hat"]
    mean = cache["mean"]
    var = cache["var"]
    gamma = cache["gamma"] 
    beta = cache["beta"]
    eps = cache["eps"]
    G = cache["G"]

    sigma = np.sqrt(var + eps)
    N, C, H, W = x.shape
    # In layernorm is N, D = x.shape
    # In groupnorm is N*G*H*W, S = x.shape
    S = C // G
    D = S

    # print("dout-shape", dout.shape) # (2, 6, 4, 5)
    # print("xhat-shape",x_hat.shape) # (2, 6, 4, 5)
    dx = np.zeros_like(x)
    dgamma = np.zeros_like(gamma)
    # Just reshape it like above to (4, 3, 4, 5)
    # dout_rtr = dout.reshape(N*G, S, H, W).transpose(0, 2, 3, 1).reshape(N*G*H*W, S)
    # print(dout_rtr.shape)

    dx_hat = dout * gamma
    # print(dx_hat.shape, x_hat.shape)
    # Rseshape dx_hat and x
    dx_hat = dx_hat.reshape(N*G, S, H, W).transpose(0, 2, 3, 1).reshape(N*G*H*W, S)
    x = x.reshape(N*G, S, H, W).transpose(0, 2, 3, 1).reshape(N*G*H*W, S)
    # print(dx_hat.shape)

    dsigma = np.sum(dx_hat * (x - mean)/sigma**(2), axis=1, keepdims=True)
    # print(dsigma.shape, sigma.shape)
    dvar = dsigma*(-1/(2*sigma))
    # print(dvar.shape, var.shape)
    dmean = np.sum(-2 * (x - mean), axis=1, keepdims=True) / D + np.sum(dx_hat * -1/ sigma, axis=1, keepdims=True)
    # print(dmean.shape, mean.shape)
    # print(dvar.shape, dmean.shape, dx_hat.shape)
    dx = (dvar * 2 * (x - mean) / D) + (dmean / D) + dx_hat / sigma

    dx = dx.reshape(N*G, H, W, S).transpose(0, 3, 1, 2).reshape(N, S*G, H, W)
    # print(dx.shape)


    dgamma = np.sum(dout * x_hat, axis = (0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis = (0, 2, 3), keepdims=True)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta



