
import time

import numpy as np

from util.activation_functions import Activation


class LogisticLayer():
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    delta : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='sigmoid', isClassifierLayer=False):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.activationDerivate = Activation.getDerivative(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.inp = np.ndarray((nIn+1, 1))
        self.inp[0] = 1
        self.outp = np.ndarray((nOut, 1))
        self.delta = np.zeros((nOut, 1))

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nIn + 1, nOut))-0.5
        else:
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

    def forward(self, inp):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        self.inp = inp
        outp = self._fire(inp)
        self.outp = outp
        return outp



    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (back)

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer

        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer
        """

        if (not self.isClassifierLayer):
            temp = self.activationDerivate(self.outp)
            temp = np.insert(temp,0,1)
            array_multi = np.dot(nextWeights,nextDerivatives)
            self.delta = temp * array_multi

        if (self.isClassifierLayer):
            self.delta = self.activationDerivate(self.outp) * (nextDerivatives - self.outp)

        return self.delta


    def updateWeights(self, learningRate):
        """
        Update the weights of the layer
        """
        for neuron in range(0,self.nOut):
            self.weights[:,neuron] -= learningRate * self.delta[neuron]*self.inp

    def _fire(self, inp):
        return self.activation(np.dot(inp,self.weights))