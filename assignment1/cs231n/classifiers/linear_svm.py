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
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

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

    ############
    # readable #
    ############

    # N = X.shape[0]
    # score = X @ W
    # correct_score = score[np.arange(N), y]
    # loss_matrix_before_max = score - correct_score[:,np.newaxis] + 1
    # loss_matrix = np.maximum(0, loss_matrix_before_max)
    # data_loss = 1/N * np.sum( loss_matrix) - 1 
    # reg_loss = reg * np.sum(W*W)
    # loss = data_loss + reg_loss

    #############
    # for speed #
    #############

    N = X.shape[0]
    score = X @ W
    loss_matrix = np.maximum(0,score.T - score[np.arange(N), y] + 1).T
    loss = 1/N * np.sum(loss_matrix) -1 + reg * np.sum(W*W)



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

    ############
    # readable #
    ############

    # # backprop loss
    # dloss = 1
    # # backprop loss = data_loss + reg_loss
    # ddata_loss = (1) * dloss
    # dreg_loss = (1) * dloss
    # # backprop reg_loss = reg * np.sum(W*W) 
    # dW += (2 * reg * W) * dreg_loss
    # # backprop data_loss = 1/N * np.sum(loss_matrix) - 1
    # # loss_matrix N * C, W: D * C
    # dloss_matrix = (1/N) * ddata_loss
    # # backprop loss_matrix = np.maimum(0, loss_matrix_before_max)
    # loss_matrix[loss_matrix>0] = 1
    # dloss_matrix_before_max = (loss_matrix) * dloss_matrix
    # # backprop loss_matrix_before_max = score - correct_score[:,np.newaxis] + 1 
    # dscore = (1) * dloss_matrix_before_max
    # dcorrect_score = (-1) * np.sum(dloss_matrix_before_max, axis=1) 
    # # backprop score = X @ W
    # dW += X.T @ dscore
    # # backprop correct_score = score[np.arange(N), y]
    # dscore2 = np.zeros(score.shape)
    # dscore2[np.arange(N), y] = dcorrect_score
    # # backprop score2 = X @ W
    # dW += X.T @ dscore2

    #############
    # for speed #
    #############

    dW += 2 * reg * W
    loss_matrix[loss_matrix>0] = 1/N
    loss_matrix[np.arange(N), y] -= np.sum(loss_matrix, axis=1)
    dW += X.T @ loss_matrix


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
