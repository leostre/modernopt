import numpy as np
import time
from spectrum import Spectrum
from miscellaneous import voigt_, gauss_, lorentz_, voigt, gauss, lorentz
import torch

width_sigma = 2 * np.sqrt(np.log(2))
width_lambda = 2.


def getInitPoint(x, data):
    spc = Spectrum(
        x, data
    )
    spc.get_derivative(2)
    inds, mus = spc.get_extrema(minima=True, tolerance=1e-8)
    A = torch.tensor([data[ind] for ind in inds])
    mus = torch.tensor([mu.item() for mu in mus])
    n = len(inds)
    widths = np.array([(mus[1] - mus[0]) / 2])
    for i in range(1, n - 1):
        dist1 = mus[i + 1] - mus[i]
        dist2 = mus[i] - mus[i - 1]
        widths = np.append(widths, min(dist1, dist2) / 2)
    widths = torch.tensor([w for w in np.append(widths, (mus[n - 1] - mus[n - 2]) / 2)])
    g = torch.distributions.uniform.Uniform(0.5, 1.0).sample([n, ])
    return torch.stack([A, widths, mus, g])


class SGD(object):

    def __init__(self, nParam, eta=0.00005, stopGrad=1e-6, maxIter=1000, **kwargs):
        allowed_kwargs = {'lowerBound', 'upperBound', 'stopGrad', 'learnSched', 'batchSize', 'momentum'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument passed to optimizer at: ' + str(k))

        self.__dict__.update(kwargs)
        self.stopGrad = stopGrad
        self.maxIter = maxIter
        self.nParam = nParam

        self.eta = eta

        self.etaCurrent = self.eta
        self.iter = 0
        self.currentIter = self.iter
        self.obj = np.inf
        self.isStartingPointInit = False
        self.obj = None

        self.isBatch = hasattr(self, 'batchSize')
        if self.isBatch and self.batchSize > 1:
            raise ValueError('Batch size greater than the number of parameters')

        self.size = torch.Size([4, int(self.nParam / 4)])
        self.grad = torch.zeros(self.size)
        self.params = torch.zeros(self.size)
        self.objFun = None
        self.gradFun = None

        # Set the lower bounds
        if not hasattr(self, 'lowerBound'):
            self.lowerBound = -float('inf') * torch.ones(self.size)
        elif np.size(self.lowerBound) == 1:
            self.lowerBound = self.lowerBound * torch.ones(self.size)

        # Set the upper bounds
        if not hasattr(self, 'upperBound'):
            self.upperBound = float('inf') * torch.ones(self.size)
        elif np.size(self.upperBound) == 1:
            self.upperBound = self.upperBound * torch.ones(self.size)

        self.stop = False
        self.updateParam = torch.zeros(self.size)

        if not hasattr(self, 'alg'):
            self.alg = 'SGD+momentum'
            if not hasattr(self, 'momentum'):
                self.alg = 'SGD'
                self.momentum = 0.0
                SGD.printAlg(self)

        # learning schedule
        if not hasattr(self, 'learnSched'):
            self.learnSched = 'constant'
        elif self.learnSched not in ['exponential', 'time-based', 'demon']:
            print('no such learning schedule in this module\nSet to constant')
            self.learnSched = 'constant'
        elif self.learnSched == 'demon':
            self.learnSched = 0.9
        elif not hasattr(self, 'lrParam'):
            self.lrParam = 0.1
        print('Learning schedule: ', self.learnSched)

    def initialize(self, params, objFun, gradFun):
        self.objFun = objFun
        self.gradFun = gradFun
        self.params = params
        self.paramHist = torch.reshape(self.params, (self.nParam, 1))
        self.obj = torch.tensor([objFun(params)])
        self.isStartingPointInit = True

    def getParams(self):
        return self.params

    def getObj(self):
        """
        Returns the current objective
        """
        return self.obj

    def getGrad(self):
        """
        Returns the gradients
        """
        return self.grad

    def getParamHist(self):
        """
        To get parameter history
        """
        return self.paramHist

    def evaluateObjFn(self):
        """
        Evaluates the objective function
        objFun should be a function handle with input: param, output: objective
        """
        self.obj = torch.cat((self.obj, self.objFun(self.params).reshape(1)), 0)

    def evaluateGradFn(self):
        """
        Evaluates the gradient function for i-th data point, where i in [0, n]
        gradFun should be a function handle with input: param, output: gradient
        """
        if self.isBatch:
            self.grad = self.gradFun(self.params, self.__getMask())
        else:
            self.grad = self.gradFun(self.params)

    def __getMask(self):
        indices = np.random.choice([False, True], self.size, p=[1 - self.batchSize, self.batchSize])
        return indices

    def satisfyBounds(self):
        """
        Satisfies the parameter bounds
        """

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.params[i, j] > self.upperBound[i, j]:
                    self.params[i, j] = self.upperBound[i, j]
                elif self.params[i, j] < self.lowerBound[i, j]:
                    self.params[i, j] = self.lowerBound[i, j]

    def update(self):
        """
        Iteration of SGD
        """
        SGD.learningSchedule(self)
        self.updateParam = self.updateParam * self.momentum + self.etaCurrent * self.grad
        self.params = self.params - self.updateParam

        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist, np.reshape(self.params, (self.nParam, 1)), axis=1)

    def run(self):
        """
        Performs all the iterations of SGD
        """
        if not self.isStartingPointInit:
            return

        t = time.time()
        for i in range(self.iter, self.maxIter, 1):
            if self.stop:
                break
            self.update()
            self.currentIter = i + 1

            # Update the objective and gradient
            if self.maxIter > 1:  # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)
                SGD.evaluateGradFn(self)
                SGD.stopCrit(self)

        print('Complete: Time Elapsed = ' + str(
            np.around(time.time() - t, decimals=4)) + 's' + ', Objective = ' +
              str(np.around(self.obj[self.currentIter - 1], decimals=6)))
        print('Number of iterations: ' + str(self.currentIter))
        print('Params: Amps, w, m, g ' + str(self.getParams()))

    def stopCrit(self):
        """
        Stopping criteria
        """
        if self.grad.ndim > 1:
            avgGrad = self.grad.mean(0)
            if np.linalg.norm(avgGrad) < self.stopGrad:
                self.stop = True
        elif np.linalg.norm(self.grad) < self.stopGrad:
            self.stop = True

    def learningSchedule(self):
        """
        creates a learning schedule for SGD
        """
        if self.learnSched == 'constant':
            self.etaCurrent = self.eta  # no change
        elif self.learnSched == 'exponential':
            self.etaCurrent = self.eta * np.exp(-self.lrParam * self.currentIter)
        elif self.learnSched == 'time-based':
            self.etaCurrent = self.eta / (1.0 + self.lrParam * self.currentIter)
        elif self.learnSched == 'demon':
            coef = (1 - self.currentIter / self.maxIter)
            self.momentum = self.lrParam * coef / \
                            (1 - self.lrParam + self.lrParam * coef)

    def printAlg(self):
        print('\nAlgorithm: ', self.alg, '\n')


class Adam(SGD):
    """
    ==============================================================================
    |                        Adam Stochastic Gradient Descent                    |
    ==============================================================================
    Initialization:
        adm = Adam(m, v, beta1, beta2, obj, grad, eta, param,
                   iter, maxIter, objFun, gradFun, lowerBound, upperBound)

    ==============================================================================
    Attributes:
        grad:           Gradient information (array of dimension nParam-by-1)
        eta:            learning rate
        param:          the parameter vector (array of dimension nParam-by-1)
        nParam:         number of parameters
        beta1, beta2:   exponential decay rates in [0,1)
                        (default beta1 = 0.9, beta2 = 0.999)
        m:              First moment (array of dimension nParam-by-1)
        v:              Second raw moment (array of dimension nParam-by-1)
        epsilon:        square-root of machine-precision
                        (required to avoid division by zero)
        iter:           iteration number
        maxIter:        maximum iteration number (optional input, default = 1)
        objFun:         function handle to evaluate the objective
                        (not required for maxIter = 1 )
        gradFun:        function handle to evaluate the gradient
                        (not required for maxIter = 1 )
        lowerBound:     lower bound for the parameters (optional input)
        upperBound:     upper bound for the parameters (optional input)
        stopGrad:       stopping criterion based on 2-norm of gradient vector
                        (default 10^-6)
    ==============================================================================
    """

    def __init__(self, nParam, m=0.0, v=0.0, beta1=0.9, beta2=0.99, **kwargs):

        self.alg = 'Adam'
        SGD.printAlg(self)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = np.finfo(float).eps  # The machine precision
        SGD.__init__(self, nParam, **kwargs)
        # Initialize first moment
        self.m = torch.zeros(self.size)
        # Initialize second raw moment
        self.v = torch.zeros(self.size)

    def update(self):
        """ Perform one iteration of Adam
        """
        SGD.learningSchedule(self)
        # Moment updates
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * self.grad  # Update biased first moment estimate
        self.mHat = self.m / (
                1.0 - self.beta1 ** (self.currentIter + 1))  # Compute bias-corrected first moment estimate

        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * torch.square(self.grad)
        self.vHat = self.v / (
                1.0 - self.beta2 ** (self.currentIter + 1))  # Compute bias-corrected second moment estimate
        # Parameter updates
        self.params = self.params - self.etaCurrent * self.mHat / (torch.sqrt(self.vHat) + self.epsilon)
        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist, np.reshape(self.params, (self.nParam, 1)), axis=1)

    def run(self):
        """
        Performs all the iterations of Adam
        """
        t = time.time()
        for i in range(self.iter, self.maxIter, 1):
            if self.stop:
                break
            self.update()
            self.currentIter = i + 1

            # Update the objective and gradient
            if self.maxIter > 1:  # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)
                SGD.evaluateGradFn(self)
                SGD.stopCrit(self)

        print('Complete: Time Elapsed = ' + str(
            np.around(time.time() - t, decimals=2)) + 's' + ', Objective = ' +
              str(np.around(self.obj[self.currentIter - 1], decimals=6)))
        print('Number of iterations: ' + str(self.currentIter))
        print('Params: Amps, w, m, g ' + str(self.getParams()))

    def getMoments(self):
        return self.m, self.v


class QHAdam(Adam):
    """
    ==============================================================================
    |                        Quasihyperbolic Adam SGD                          |
    ==============================================================================
    Initialization:
        adm = QHAdam(m, v, beta1, beta2, nu1, nu2, obj, eta, param,
                   iter, maxIter, objFun, gradFun, lowerBound, upperBound)

    ==============================================================================
    Attributes:
        eta:            learning rate
        param:          the parameter vector (array of dimension nParam-by-1)
        nParam:         number of parameters
        beta1, beta2:   exponential decay rates in [0,1)
                        (default beta1 = 0.9, beta2 = 0.999)
        nu1, nu2 :      weights
        m:              First moment (array of dimension nParam-by-1)
        v:              Second raw moment (array of dimension nParam-by-1)
        epsilon:        square-root of machine-precision
                        (required to avoid division by zero)
        iter:           iteration number
        maxIter:        maximum iteration number (optional input, default = 1)
        objFun:         function handle to evaluate the objective
                        (not required for maxIter = 1 )
        gradFun:        function handle to evaluate the gradient
                        (not required for maxIter = 1 )
        lowerBound:     lower bound for the parameters (optional input)
        upperBound:     upper bound for the parameters (optional input)
        stopGrad:       stopping criterion based on 2-norm of gradient vector
                        (default 10^-6)
        alg:            algorithm used
        __version__:    version of the code
    ==============================================================================
    Methods:
     Public:
        performIter:    performs all the iterations inside a for loop
        getGradHist:    returns gradient history (default is zero)
        getMoments:     returns history of moments
        Inherited:
            getParam:       returns the parameter values
            getObj:         returns the current objective value
            getGrad:        returns the current gradient information
            getParamHist:   returns parameter update history
     Private: (should not be called outside this class file)
        __init__:       initialization
        update:         performs one iteration of Adam
        Inherited:
            evaluateObjFn:      evaluates the objective function
            evaluateGradFn:     evaluates the gradients
            satisfyBounds:      satisfies the parameter bounds
            learningSchedule:   learning schedule
            stopCrit:           check stopping criteria
    ==============================================================================
    """

    def __init__(self, nu1=0.8, nu2=0.7, m=0.0, v=0.0, beta1=0.9, beta2=0.99, **kwargs):

        self.alg = 'QHAdam'
        SGD.printAlg(self)

        self.nu1 = nu1
        self.nu2 = nu2
        Adam.__init__(self, **kwargs)

    def update(self):
        """ Perform one iteration of Adam
        """
        SGD.learningSchedule(self)
        # Moment updates
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * self.grad  # Update biased first moment estimate
        self.mHat = self.m / (
                1.0 - self.beta1 ** (self.currentIter + 1))  # Compute bias-corrected first moment estimate

        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * np.multiply(self.grad,
                                                                        self.grad)
        self.vHat = self.v / (
                1.0 - self.beta2 ** (self.currentIter + 1))  # Compute bias-corrected second moment estimate
        # Parameter updates
        self.params = self.params - np.divide((self.etaCurrent * self.mHat), (np.sqrt(self.vHat)) + self.epsilon)
        SGD.satisfyBounds(self)
        self.paramHist = np.append(self.paramHist, np.reshape(self.params, (2, 1)), axis=1)

    def run(self):
        """
        Performs all the iterations of Adam
        """
        t = time.time()
        for i in range(self.iter, self.maxIter, 1):
            if self.stop:
                print('Complete: Time Elapsed = ' + str(
                    np.around(time.time() - t, decimals=4)) + 's' + ', Objective = ' +
                      str(np.around(self.obj[self.currentIter - 1], decimals=6)))
                break
            self.update()
            self.currentIter = i + 1

            # Update the objective and gradient
            if self.maxIter > 1:  # since objFun and gradFun are optional for 1 iteration
                SGD.evaluateObjFn(self)
                SGD.evaluateGradFn(self)
                SGD.stopCrit(self)

    def getMoments(self):
        """
        This returns the updated moments
        """
        return self.m, self.v


class GradDeconvolutor:
    def __init__(self, data: torch.Tensor, x: torch.Tensor, startPoint, optimizer: SGD):
        self.data = torch.clone(data)
        self.x = torch.clone(x)
        self.startPoint = startPoint
        self.optimizer = optimizer
        self.optimizer.initialize(self.startPoint, self.MSE, self.grad)

    def grad(self, point, mask=None):
        params = torch.clone(point)

        def dF_dA(amp, w, mu, G):
            return voigt(self.x, torch.tensor([1]), w, mu, G)

        def dF_dW(amp, w, mu, G):
            res = amp * (torch.square(self.x - mu) * (
                    gauss(self.x, torch.tensor([1]), w, mu) / torch.pow(w / width_sigma, 3) * G + \
                    2 * torch.square( lorentz(self.x, torch.tensor([1]), w, mu)) * (1 - G) / torch.pow(w / width_lambda, 3)) - (
                    gauss(self.x, torch.tensor([1]), w, mu) / (w / width_sigma) * G +
                    lorentz(self.x, torch.tensor([1]), w, mu) * (1 - G) / (w / width_lambda))
                    )
            return res

        def dF_dmu(amp, w, mu, G):
            res = (self.x - mu) * (gauss(self.x, amp, w, mu) * G / torch.square((w / width_sigma)) + \
                   2 * amp * torch.square(lorentz(self.x, torch.tensor([1]), w, mu) / w * width_lambda) * (1 - G))
            return res

        def dF_dG(amp, w, mu, G):
            return gauss(self.x, amp, w, mu) - lorentz(self.x, amp, w, mu)

        nbands = params.shape[1]
        grad = np.array([])
        # val = self.data.sub(voigt_(self.x, *params))
        # gradF = val / torch.abs(val)
        gradF = - 2 * self.data.sub(voigt_(self.x, *params))
        for j, func in enumerate([dF_dA, dF_dW, dF_dmu, dF_dG]):
            for i in range(nbands):
                if mask is not None and mask[j, i]:
                    grad = np.append(grad, torch.tensor(0))
                else:
                    grad = np.append(grad, (gradF * func(*params[:, i])).mean())

        return torch.tensor(grad).reshape(self.optimizer.size)

    def MSE(self, params):
        approx = voigt_(self.x, *params)
        mse = torch.square(approx.sub(self.data)).mean()
        return mse

    def run(self):
        self.optimizer.run()
        return self.optimizer
