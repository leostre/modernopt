from BaselineRemoval import BaselineRemoval 
import numpy as np
from scipy.spatial import ConvexHull
from scipy import sparse
from scipy.sparse.linalg import spsolve
# rubberband

def baseline_zhang(y, polynomial_degree=2):
    """
    :param: numpy.array of floats
    :return: numpy.array of floats

	adaptive iteratively reweighted Penalized Least Squares
	"""
    baseObj = BaselineRemoval(y)
    return baseObj.ZhangFit()


def baseline_rubberband(x, y):
    """
    :param x: numpy.array of floats
    :param y: numpy.array of floats
    :return: numpy.array of floats - data without baseline
    """
    base = ConvexHull(list(zip(x, y))).vertices
    base = np.roll(base, -base.argmax() - 1)
    base1 = base[base.argmin():]
    base2 = base[:base.argmin() + 1]
    if len(base2) > 1:
        base1 = list(base1 if y[base1[1]] < y[base2[1]] else base2)
    else:
        base1 = list(base1)
    base1 = [len(x) - 1] + base1 + [0]
    new_y = y - np.interp(x, x[base1], y[base1])
    return new_y


def baseline_alss(y, lam=1e6, p=1e-3, niter=10):
    """
    :param y: numpy.array of floats - data
    :param lam: float - smoothing degree
    :param p: float - asymmetry coefficient. Preferably in range [0.001, 0.1]
    :param niter: int - number of aloruthm iterations
    :return: numpy.array of floats - data without baseline

    Asymmetric Least Squares Smoothing by P. Eilers and H. Boelens
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return y - z
