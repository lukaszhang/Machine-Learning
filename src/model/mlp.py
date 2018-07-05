
import sys
import logging

import numpy as np

from util.loss_functions import CrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

from util.loss_functions import *



class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='ce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        #self.cost = cost

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        elif loss == 'ce':
            self.loss = CrossEntropyError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128, 
                           None, inputActivation, False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(128, 10,
                           None, outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)

    # def __getitem__(self, item):
    #     return self.layers[item]

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, input):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        # input layer
        current_layers = self._get_input_layer()
        current_layers.inp = input
        current_layers.outp = LogisticLayer.forward(current_layers,current_layers.inp)
        temp_output = current_layers.outp #output of input layer is output of output layer

        #output layer
        temp_output = np.insert(temp_output,0,1)
        current_layers = self._get_output_layer()
        current_layers.inp = temp_output
        current_layers.outp = LogisticLayer.forward(current_layers,current_layers.inp)
        return current_layers.outp

        
    def _compute_derivative(self, nextDerivatives, nextWeights):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        #for output layer, nextDerivatives is target value

        LogisticLayer.computeDerivative(self.current_layers,nextDerivatives,nextWeights)
    
    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """
        current_layers = self._get_output_layer()
        LogisticLayer.updateWeights(current_layers, learningRate)

        current_layers = self._get_input_layer()
        LogisticLayer.updateWeights(current_layers, learningRate)
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        learned = False
        iteration = 0

        while not learned:
            for epoch in range(self.epochs):
                if verbose:
                    print("Training epoch {0}/{1}.."
                          .format(epoch + 1, self.epochs))

                if verbose:
                    evaluate_matrix = self.evaluate(self.validationSet.input)
                    accuracy = accuracy_score(self.validationSet.label[:,7],
                                              evaluate_matrix[:,7])
                    # Record the performance of each epoch for later usages
                    # e.g. plotting, reporting..
                    self.performances.append(accuracy)
                    print("Accuracy on validation: {0:.2f}%"
                          .format(accuracy * 100))
                    print("-----------------------------")

                for input, label in zip(self.trainingSet.input,
                                        self.trainingSet.label):
                    #feedforward
                    self._feed_forward(input)

                    #compute Derivative
                    current_layers = self._get_output_layer()
                    current_layers.computeDerivative(self.loss.calculateDerivative(
                                                label,current_layers.outp),1.0)
                    Derivatives = current_layers.delta
                    Weights = current_layers.weights

                    current_layers = self._get_input_layer()
                    current_layers.computeDerivative(Derivatives,
                                                        Weights)

                    #weights update
                    self._update_weights(self.learningRate)




    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        #output of input layer
        current_layers = self._get_input_layer()
        temp = LogisticLayer._fire(current_layers, test_instance)
        temp = np.insert(temp,0,1)

        #output of output layer
        current_layers = self._get_output_layer()
        result = LogisticLayer._fire(current_layers, temp)
        return result>0.5

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        length = test.shape[0]
        evaluate_matrix = np.ndarray((length,10))
        for i in range(length):
            evaluate_matrix[i,:] = self.classify(test[i,:])
        return evaluate_matrix

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
