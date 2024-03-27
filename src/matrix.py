import numpy as np
from pandas import DataFrame
from spectrum import Spectrum
from enumerations import Scale, BaseLineMode, NormMode
from scan import get_spectra_list, get_spectra_dict
from miscellaneous import scale_change
from smoothing import Smoother
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Matrix:
    def __init__(self, spectra):
        self.spectra = spectra

    @property
    def shape(self):
        if self.spectra:
            return len(self.spectra), len(self.spectra[0])
        return 0, 0

    @classmethod
    def create_matrix(cls, raw_spectra, config, predicate=None):
        """
        :param raw_spectra: iterable of Spectrum
        :param config: dict - spectra modifications description
        :param predicate: (spc) -> bool - function to determine whether to include the spectrum
        :return: Matrix
        """
        if not predicate:
            predicate = lambda x: True
        matrix = []
        if not raw_spectra:
            return
        for spc in raw_spectra:
            # spc.transform
            if not predicate(spc):
                continue
            if 'smooth' in config:
                spc.smooth(**config['smooth'])
            if 'baseline' in config:
                spc.correct_baseline(**config['baseline'])
            if 'normalize' in config:
                spc.normalize(**config['normalize'])
            if 'derivative' in config:
                spc.get_derivative(**config['derivative'])
            if 'range' in config:
                spc = spc.range(*config['range'])
            matrix.append(spc)
        return Matrix(matrix)

    @property
    def sample_spectrum(self):
        """
        Yet to be designed.
        :return:
        """
        pass

    @classmethod
    def create_from_dataframe(cls, df: DataFrame):
        """
        :param df: pandas.DataFrame
        :return: Matrix
        """
        spectra = []
        classes = df.pop(df.columns[0])
        scale = df.columns.astype(float)
        for i in range(df.shape[0]):
            spectra.append(
                Spectrum(
                    scale.to_numpy(),
                    df.iloc[i, :],
                    clss=classes.iloc[i]
                )
            )
        return Matrix(spectra)

    def differentiate(self, n=2):
        """
        :param n: derivative order

        in-place differentiation
        """
        for spc in self.spectra:
            spc.get_derivative(n=n)

    def similarity_hist(self):
        """
        Plot the histogram of pairwise cosine similarity
        """
        similarities = []
        n = len(self.spectra)
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(self.spectra[i] ^ self.spectra[j])
        plt.hist(similarities, bins=n)
        plt.show()

    @classmethod
    def read_csv(cls, path, scale_type=Scale.WAVENUMBERS):
        """
        :param path: filepath to the csv file
        :param scale_type: Scale
        :return: (scale, data, class)

        Creates a generator to get separately essential characteristics from a big size csv file
        """
        with open(path, 'r') as csv:
            scale = csv.readline().split(',')
            scale_t, *scale = scale
            if scale_t:
                scale_type = scale_t

            if scale_type == Scale.WAVENUMBERS.value:
                f = float
            elif scale_type == Scale.WAVELENGTH_um.value:
                f = lambda x: 10_000. / float(x)
            else:
                f = lambda x: 10_000_000. / float(x)
            scale = np.array(list(map(f, scale)))
            while True:
                spc = csv.readline().strip()
                if len(spc) == 0:
                    break
                clss, *data = spc.split(',')
                data = np.array(list(map(float, data)))
                yield scale, data, clss

    def save_matrix(self, path='matrix.csv', scale_type=Scale.WAVENUMBERS):
        """
        :param path: str - file destination
        :param scale_type: Scale

        Writes the Matrix as a csv fie
        """
        if not self.spectra:
            return
        sc = self.spectra[0]
        f = scale_change(scale_type)
        scale = list(map(f, sc.wavenums))
        with open(path, 'w') as out:
            print(scale_type.value, *scale, sep=',', file=out)
            for spc in self.spectra:
                print(spc.clss, *spc.data, sep=',', file=out)

    def as_2d_array(self, predicate=lambda x: True):
        """
        :param predicate: (spc) -> bool: which spectra to include
        :return: 2d numpy.array
        """
        return np.vstack([spc.data for spc in self.spectra if predicate(spc)])

    def __corr_half(self, predicate):
        """
        :return: a half of the correlation matrix within the selection
        """
        C = 4
        r = self.corr(predicate) + C
        r = np.triu(r, 1) - C
        return np.array(list(filter(lambda x: x >= -1, r.flatten())))

    def average_by_point(self):
        """
        :return: Spectrum - the averaged spectrum
        """
        res = np.zeros(self.shape[1])
        for spc in self.spectra:
            res += spc.data
        return res / self.shape[0]

    def corr(self, predicate):
        """
        :param predicate:  (spc) -> bool: which spectra to include
        :return: 2d numpy.array - correlation matrix
        """
        mtr = self.as_2d_array(predicate)
        mtr = pd.DataFrame(mtr)
        return mtr.corr(method='spearman').to_numpy()

    def one_spectrum(self, spc):
        """
        :param spc: Spectrum
        :return: 2d numpy.array

        intraspectrum correlation
        """
        avgs = self.average_by_point()
        res = np.zeros(shape=(len(spc), len(spc)))
        y = spc.data
        for i in range(len(spc)):
            for j in range(len(spc)):
                res[i][j] = (y[i] - avgs[i]) * (y[j] - avgs[j])
        return res

    def get_stat_matrix(self):
        mtr = self.as_2d_array()
        res = np.zeros(shape=(self.shape[1], self.shape[1]))
        avgs = self.average_by_point()
        res_std = res.copy()
        for i in range(res.shape[0]):
            for j in range(res.shape[0]):
                distr = -(mtr[:, i] - mtr[:, j])
                res[i][j] = distr.mean()
                res_std[i][j] = distr.std()
            res[i, :] += avgs[i]
        return {
            'mean': res,
            'std': res_std
        }

    def get_random_values(self, correls, d):
        stds = d['std']
        means = d['mean']
        size = stds.shape[0]
        res = np.zeros(shape=(size,))

        for i in range(size):
            res[i] = np.random.normal(
                (means[:, i] * np.abs(correls[:, i])).sum() / np.abs(correls[:, i]).sum(),
                stds[:, i].mean()
            )
        return res

if __name__ == '__main__':
    print('Matrix')
    from scan import get_spectra_list
    from enumerations import Smooth, BaseLineMode, NormMode
    from smoothing import Smoother
    spa = get_spectra_list(path='../new_data', recursive=True)
    mtr = Matrix.create_matrix(spa, {
        'smooth': {'method': Smoother.savgol},
        'baseline': {'method': BaseLineMode.ALSS},
        'normalize': {'method': NormMode.MINMAX}
    }).save_matrix()









