from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    # X (500, 3073)
    # W (3073, 10)
    # y (500,)
    num_classes = W.shape[1] # 10
    num_train = X.shape[0]  # 500
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        # after sum
        d_sum = np.ones(num_classes) * 1
        # after max
        d_max = np.zeros(num_classes)
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                d_max[j] = 1
        
        d_max = d_max * d_sum
        # after minus +
        d_plus = d_max
        # after minus -
        d_minus = -d_max
        # after broadcast
        d_broad = np.sum(d_minus)
        # after selector
        d_select = np.zeros(num_classes) 
        d_select[y[i]] = d_broad
        # before dot (10,)
        d_before_dot = d_plus + d_select
        # after dot
        d_dot = np.outer(X[i], d_before_dot)
        # add to dW
        dW += d_dot

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    # lost per test
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # after reg
    d_reg = 2*W * reg
    # add to dW
    dW += d_reg


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    S = X @ W
    # S_compare = S.T - S[np.arange(num_train), y] + 1
    S_compare = S - S[np.arange(num_train), y][:,np.newaxis] + 1
    S_compare_ifpos = S_compare > 0
    data_loss = np.sum(S_compare[S_compare_ifpos]) / num_train - 1
    reg_loss = reg * np.sum(W * W)
    loss = data_loss + reg_loss


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #### origin
    # / num_train
    # upper = 1
    d_divide = 1/num_train * 1
    # sum(S_compare) shape (num_train, num_classes)
    # upper = d_divide
    d_sum = np.ones((num_train, num_classes)) * d_divide
    # max(S_compare, 0)
    d_max = S_compare_ifpos * d_sum
    #### end origin
    #### fast
    # d_max = S_compare_ifpos / num_train
    #### end fast
    #### origin
    # minus plus side
    d_plus = 1 * d_max
    # minus minus side
    d_minus = -1 * d_max
    # broadcast (500, 1) to (500, 10)
    d_broad = np.sum(d_minus, axis=1)
    # select (500, 10) yi to (500, 1)
    # only yi take effect
    d_select = np.zeros((num_train, num_classes))
    d_select[np.arange(num_train), y] = d_broad
    # before dot
    d_before_dot = d_select + d_plus
    #### end origin
    #### fast
    # d_max[np.arange(num_train), y] -= np.sum(d_max, axis=1)
    # d_before_dot = d_max
    #### end fast
    
    # dot 
    d_dot = X.T @ d_before_dot


    d_reg = 2*W *reg

    dW = d_dot + d_reg
    


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
