import torch
import numpy as np


class SGD(object):

    obj = np.inf

    def __init__(self, params, lr=1.0, stopGrad=1e-6, max_iter=300, **kwargs):
        allowed_kwargs = {'lr', 'objFun', 'gradFun', 'lowerBound',
                          'upperBound', 'oldGrad', 'stopGrad', 'momentum', 'nesterov', 'learnSched', 'lrParam'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument passed to optimizer at: ' + str(k))

        self.__dict__.update(kwargs)
        self.stopGrad = stopGrad
        self.max_iter = max_iter
        self.params = params
        self.nParam = np.size(self.params)
        self.lr = lr
        self.iter = 0
        self.currentIter = self.iter

        # Gradient information
        if not hasattr(self, 'gradFun'):
            raise ValueError('Please provide the gradient function')

        if not hasattr(self, 'objFun'):
            raise ValueError('Please provide the objective function')

        # Set the lower bounds
        if not hasattr(self, 'lowerBound'):
            self.lowerBound = -np.inf * np.ones(self.nParam)
        elif np.size(self.lowerBound) == 1:
            self.lowerBound = self.lowerBound * np.ones(self.nParam)

        # Set the upper bounds
        if not hasattr(self, 'upperBound'):
            self.upperBound = np.inf * np.ones(self.nParam)
        elif np.size(self.upperBound) == 1:
            self.upperBound = self.upperBound * np.ones(self.nParam)

        # Momentum
        if hasattr(self, 'momentum'):
            self.alg = 'Momentum'
            self.momentum = 0.0
        self.paramHist = np.reshape(self.params, (2, 1))

        self.stop = False
        self.updateParam = np.zeros(self.nParam)
        # Nesterov momentum
        if hasattr(self, 'nesterov'):
            if self.nesterov:
                self.alg = 'SGD+Nesterov'
        else:
            self.nesterov = False
        # learning schedule
        if not hasattr(self, 'learnSched'):
            self.learnSched = 'constant'
        elif self.learnSched != 'exponential' and self.learnSched != 'time-based':
            print('no such learning schedule in this module\nSet to constant')
            self.learnSched = 'constant'
        elif not hasattr(self, 'lrParam'):
            self.lrParam = 0.1
        print('Learning schedule: ', self.learnSched)

    def getParams(self):
        return self.params

    def getObj(self):
        """
        To get the current objective (if possible)
        """
        self.evaluateObjFn()
        return self.obj

    def getGrad(self):
        """
        To get the gradients
        """
        return self.grad

    def getParamHist(self):
        """
        To get parameter history
        """
        return self.paramHist

    def evaluateObjFn(self):
        """
        Evalutes the objective function
        objFun should be a function handle with input: param, output: objective
        """
        self.obj = np.append(self.obj, self.objFun(self.params))

    def evaluateGradFn(self):
        """
        Evalutes the gradient function for i-th data point, where i in [0, n]
        gradFun should be a function handle with input: param, output: gradient
        """
        self.grad = self.gradFun(self.param)

    def satisfyBounds(self):
        """
        Satisfies the parameter bounds
        """
        for i in range(self.nParam):
            if self.params[i] > self.upperBound[i]:
                self.params[i] = self.upperBound[i]
            elif self.params[i] < self.lowerBound[i]:
                self.params[i] = self.lowerBound[i]

    def update(self):
        """
        Iteration of SGD
        """
        SGD.learningSchedule(self)
        if self.nesterov:
            grdnt = self.gradFun(self.params - self.momentum * self.updateParam)
            self.updateParam = self.updateParam * self.momentum + self.etaCurrent * grdnt
        else:
            self.updateParam = self.updateParam * self.momentum + self.etaCurrent * self.grad
        self.params = self.params - self.updateParam

        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist, np.reshape(self.params, (2, 1)), axis=1)

    def performIter(self):
        """
        Performs all the iterations of SGD
        """
        SGD.printAlg(self)
        # initialize progress bar
        for i in range(self.iter, self.max_iter, 1):
            if self.stop:
                break
            self.update()
            self.currentIter = i + 1

            # Update the objective and gradient
            if self.max_iter > 1:  # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)
                SGD.evaluateGradFn(self)
                SGD.stopCrit(self)

    def stopCrit(self):
        """
        Stopping criteria
        """
        if self.grad.ndim > 1:
            self.avgGrad = np.mean(self.grad, axis=1)
            if np.linalg.norm(self.avgGrad) < self.stopGrad:
                self.stop = True
        elif np.linalg.norm(self.grad) < self.stopGrad:
            self.stop = True

    def learningSchedule(self):
        """
        creates a learning schedule for SGD
        """
        if self.learnSched == 'constant':
            self.etaCurrent = self.lr  # no change
        elif self.learnSched == 'exponential':
            self.etaCurrent = self.lr * np.exp(-self.lrParam * self.currentIter)
            print(self.etaCurrent)
        elif self.learnSched == 'time-based':
            self.etaCurrent = self.lr / (1.0 + self.lrParam * self.currentIter)

    def printAlg(self):
        """
        prints algorithm
        """
        print('\nAlgorithm: ', self.alg, '\n')

