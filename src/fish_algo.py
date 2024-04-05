import torch
from spectrum import Spectrum
from miscellaneous import voigt_
import sys
class FishSchool:
    little_val = -1e10
    def __init__(self, nfish, nbands, fitness_func):
        self.tol = 1e-9
        self.nparam_per_band = 4
        self.nfish = nfish
        self.nbands = nbands
        self.best = FishSchool.little_val
        self._fitness_func = fitness_func
        
        self.positions = torch.FloatTensor(nfish,
                                            nbands * self.nparam_per_band,
                                            ).double()
        self.weights = torch.full([nfish], 1.).double()
        self.delta_fitness = torch.zeros_like(self.weights).add_(self.tol).double()
        self.boundaries = None
        self.total_weight = 0.
        # self.new_position = torch.FloatTensor(nfish, nbands * self.nparam_per_band, dtype=torch.float64)
        self.delta_pos = torch.zeros_like(self.positions).add_(self.tol).double()
        self.fitness = torch.full([nfish], FishSchool.little_val).double()
        self.step_vol = .5
        self.step_ind = .1
        self.init = False
        self.optimal_position = None

    def _calc_fitness(self, positions):
        fitness = torch.zeros_like(self.fitness)
        for i, position in enumerate(positions):
            fitness[i] = self._fitness_func(position)
        return fitness

    def _upd_optimal(self):
        i = torch.argmax(self.fitness)
        if self.best < self.fitness[i]:
            self.optimal_position = self.positions[i, :]
            self.best = self.fitness[i].detach().item()

    def _individual_swimming(self):
        newp = self._individual()
        delta_pos = newp - self.positions
        newf = self._calc_fitness(newp)
        mask = newf > self.fitness
        # assert mask.any()

        self.delta_pos = torch.zeros_like(self.positions)
        self.delta_pos[mask, :] = delta_pos[mask, :]
        
        self.delta_fitness = (newf - self.fitness)
        self.delta_fitness[~mask] = 0.

        self.fitness[mask] = newf[mask]
        self.positions[mask, :] = newp[mask, :]
    
    def _collective_swimming(self):
        self.positions = self._instinctive()
        self.positions = self._collective()
        # self._update_total_weight()   
        self._feed()
        self._update_total_weight()

    def _feed(self):
        max_d_fitness = self.delta_fitness.max()
        if max_d_fitness.item()!= 0:
            self.weights += self.delta_fitness / max_d_fitness
        else:
            self.weights[:] = 1.

    def _update_total_weight(self):
        self.total_weight = self.weights.sum().item()

    def _collective(self):
        total = self.weights.sum()
        barycenter =  self.positions * self.weights[:, None] / total
        search = (-1) ** (total.item()  <= self.total_weight)
        lmb = torch.rand_like(self.weights)[:, None]
        new_position = self._clip(self.positions + (self.positions - barycenter) \
            * self.step_vol * lmb * search)
        return new_position

    def run(self, max_iter=1000):
        self._init_weights()
        while max_iter: #and self._check_stop():
            max_iter -= 1
            self._individual_swimming()
            self._collective_swimming()
            # self._feed()
            # self._update_total_weight()
            self._upd_optimal()

    def init_boundaries(self, pd):
        nbands = self.nbands
        self.boundaries = torch.tensor([
            pd[k] for k in ['amp', 'w', 'x0', 'gau']])[:, None, :].repeat(1, nbands, 1).view(-1, 2).t()

    def _init_weights(self):
        self.positions = (self.boundaries[1, :] - self.boundaries[0, :]).repeat(self.nfish, 1) * torch.rand(self.nfish, self.nbands * self.nparam_per_band)\
              + self.boundaries[0, :].repeat(self.nfish, 1)

    def _individual(self):
        lmb = torch.rand_like(self.positions).mul_(2).sub_(1).mul_(self.step_ind)
        return self._clip(self.positions + lmb)

    def _instinctive(self):
        total_delta = self.delta_fitness.sum().item()
        # print(self.delta_pos.size(), self.delta_fitness.size())
        instinct = self.delta_pos * self.delta_fitness[:, None]#.view(self.nfish)
        if total_delta:
            instinct /= total_delta
        return self._clip(self.positions + instinct)

    def _check_stop(self):
        return self.delta_pos.max() < self.tol

    def _clip(self, values):
        return torch.clamp_(values, min=self.boundaries[0,:], max=self.boundaries[1,:])
    

class Deconvolutor:
    def __init__(self, spectrum:Spectrum, nfish:int, boundaries:dict, mse_transform=None,):
        self.spectrum = spectrum
        self.reference = torch.tensor(spectrum.data)
        self.x = torch.tensor(spectrum.wavenums)
        self.nbands = self.__count_peaks()
        self.school = FishSchool(nfish, self.nbands, self.get_fitness())
        self.school.init_boundaries(boundaries)
        self.mse_transform = mse_transform

    def __count_peaks(self):
        spc = self.spectrum * 1
        spc.get_derivative(2)
        return len(spc.get_extrema(minima=True,)[0])
    
    def get_fitness(self):
        x, reference = self.x, self.reference
        def _fitness_f(params):
            approx = voigt_(x, *params.view(4, -1), True)
            mse = torch.square_(approx.sub_(reference)).mean()
            if not self.mse_transform:
                return mse.neg_()
            elif self.mse_transform == 'neg_log':
                return mse.add_(1e-8).log_().neg_()
            elif self.mse_transform == 'reciprocal':
                return torch.reciprocal_(mse)
            else:
                raise NotImplementedError
        return _fitness_f
    
    def run(self, max_iter):
        self.school.run(max_iter)
