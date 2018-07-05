# -*- coding: utf-8 -*-
import numpy as np

class DataSet(object):
    """
    Representing train, valid or test sets

    Parameters
    ----------
    data : list
    oneHot : bool
        If this flag is set, then all labels which are not `targetDigit` will
        be transformed to False and `targetDigit` bill be transformed to True.
    targetDigit : string
        Label of the dataset, e.g. '7'.

    Attributes
    ----------
    input : list
    label : list
        A labels for the data given in `input`.
    oneHot : bool
    targetDigit : string
    """

    def __init__(self, data, oneHot=True, targetDigit='7'):

        # The label of the digits is always the first fields
        # Doing normalization
        self.input = (1.0 * data[:, 1:])/255
        self.oneHot = oneHot
        self.targetDigit = ['0','1','2','3','4','5','6','7','8','9']
        self.num = data[:, 0]

        # Transform all labels which is not the targetDigit to False,
        # The label of targetDigit will be True,
        if oneHot:
            self.label = data[:, 0]
            self.label = list(map(lambda a: 1 
                            if str(a) == targetDigit else 0, 
                            self.num))

        else:
            self.label= np.ndarray((data.shape[0],10))
            for i in range(10):
                self.label[:,i]=list(map(lambda a: 1
                            if str(a) == self.targetDigit[i] else 0,
                            self.num))


    def __iter__(self):
        return self.input.__iter__()
