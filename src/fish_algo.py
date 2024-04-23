import torch
from spectrum import Spectrum
from miscellaneous import voigt_
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from bisect import bisect_left

little_val = -1e6

class FishSchool:
    little_val = little_val
    def __init__(self, nfish, nbands,
                fitness_func,
                step_ind=1., step_vol=0.5, 
                explorators_prop=0.):
        self.tol = 1e-7
        self.nparam_per_band = 4
        self.nfish = nfish
        self.nbands = nbands
        self.best = FishSchool.little_val
        self._fitness_func = fitness_func
        
        self._positions = torch.FloatTensor(nfish,
                                            nbands * self.nparam_per_band,
                                            ).double()
        
        self._weights = torch.full([nfish], FishSchool.little_val).double()
        self._delta_fitness = torch.zeros_like(self._weights).add_(self.tol).double()
        self.boundaries = None
        self._total_weight = 0.
        self._delta_pos = torch.zeros_like(self._positions).add_(self.tol).double()
        self._fitness = torch.full([nfish], FishSchool.little_val).double()
        self.init_step_vol = step_vol
        self.init_step_ind = step_ind
        self.optimal_position = None
        self.history = []
        self.explorators_prop = explorators_prop
        self.scheduler = None

    def _calc_fitness(self, positions):
        fitness = torch.zeros_like(self._fitness)
        for i, position in enumerate(positions):
            fitness[i] = self._fitness_func(position)
        return fitness

    def _update_optimal(self):
        i = torch.argmax(self._fitness)
        fit = self._fitness[i].item()
        self.history.append(fit)
        if self.best < fit:
            self.optimal_position = self._positions[i, :]
            self.best = fit

    def _regroup(self, newp,):
        delta_pos = newp - self._positions
        newf = self._calc_fitness(newp)
        mask = newf > self._fitness
        # add_mask = torch.rand_like(self._fitness) < exploratory_prob
        explorators = torch.rand_like(self._fitness) < self.explorators_prop
        mask[explorators] = True

        self._delta_pos = torch.zeros_like(self._positions)
        self._delta_pos[mask, :] = delta_pos[mask, :]
        
        self._delta_fitness = (newf - self._fitness)
        self._delta_fitness[~mask] = 0.

        self._fitness[mask] = newf[mask]
        self._positions[mask, :] = newp[mask, :]

    def _individual(self):
        lmb = torch.rand_like(self._positions).mul_(2).sub_(1).mul_(
            (self.boundaries[1, :] - self.boundaries[0, :])[None, :]
        ).mul_(self.init_step_ind)
        return self._clip(self._positions + lmb)

    def _instinctive(self):
        total_delta = self._delta_fitness.sum().item()
        if total_delta < self.tol:
            return self._positions
        instinct = self._delta_pos * self._delta_fitness[:, None]#.view(self.nfish)
        # instinct = self._delta_pos[torch.argmax(self._delta_fitness), :][None, :]
        instinct /= total_delta
        instinct = instinct.sum(0)[None, :]

        return self._clip(self._positions + instinct)
    
    def _collective(self):
        total = self._weights.sum()
        barycenter =  self._positions * self._weights[:, None] / total
        search = -1. if total.item() < self._total_weight else 1.
        lmb = 1 # torch.rand_like(self.weights)[:, None] 
        step_vol = self.init_step_vol * (self.boundaries[1, :] - self.boundaries[0, :])\
              * torch.rand([self.boundaries.size(1)])
        new_position = self._clip(self._positions + (self._positions - barycenter) \
             * lmb * step_vol[None, :] * search)
        return new_position

    def _individual_swimming(self):
        newp = self._individual()
        self._regroup(newp)
    
    def _collective_swimming(self):
        self._positions = self._instinctive()
        self._feed() #
        # self._positions = self._collective()
        self._regroup(self._collective())
        # self._feed()

    def _feed(self):
        max_d_fitness = self._delta_fitness.max()
        if max_d_fitness.item()!= 0:
            self._weights += self._delta_fitness / max_d_fitness
        else:
            self._weights[:] = FishSchool.little_val

    def _update_total_weight(self):
        self._total_weight = self._weights.sum().item()

    def run(self, max_iter=1000, init=True):
        if init:
            self._init_pos()
        for i in tqdm(range(max_iter), desc='Iteration: ', leave=False): #and self._check_stop():
            self._individual_swimming()
            self._collective_swimming()
            # self._feed()
            self._update_total_weight()
            self._update_optimal()
            if self.scheduler:
                self.scheduler.step()
            if i and self._check_stop():
                break

    def init_boundaries(self, params):
        if isinstance(params, dict):
            pd = params
            nbands = self.nbands
            self.boundaries = torch.tensor([
                pd[k] for k in ['amp', 'w', 'x0', 'gau']])[:, None, :].repeat(
                    1, nbands, 1).view(-1, 2).t()
        else:
            params = torch.tensor(params)
            assert params.size() == (2, self.nparam_per_band * self.nbands)
            self.boundaries = params
        

    def _init_pos(self):
        range = (self.boundaries[1, :] - self.boundaries[0, :])
        lower =  self.boundaries[0, :]
        n = self.nbands
        #amp
        offset = 0.1
        range[:n] = range[:n] - offset
        lower[:n] = offset
        # ws
        # tmp = range[n: 2 * n]
        mean = range[n] / n
        lower[n: 2 * n] = mean
        range[n: 2 * n] = mean * 0.1
        # x0
        # gau
        rng = 0.05 
        lower[3 * n:] = .5 - rng
        range[3 * n:] = rng
 
        self._positions = range.repeat(self.nfish, 1) * torch.rand(self.nfish, self.nbands * self.nparam_per_band)\
              + lower.repeat(self.nfish, 1)


    def _check_stop(self, explore=False):
        flag = self._delta_fitness.abs().max() < self.tol
        if explore and flag:
            self.init_step_ind *= 1.2
            self._individual()
            self.init_step_ind /= 1.2
            return False
        return flag

    def _clip(self, values):
        return torch.clamp_(values, min=self.boundaries[0,:], max=self.boundaries[1,:])
    

class Deconvolutor:
    def __init__(self, spectrum:Spectrum, nfish:int,
                 boundaries:dict = None, mse_transform='neg', p=2,
                 opt_params=None, amp_range=0.5, pos_range=.1,
                 scheduler_regime=None, scheduler_sch=None, **sch_kw ):
        self.spectrum = spectrum
        self.reference = torch.tensor(spectrum.data)
        self.x = torch.tensor(spectrum.wavenums)
        self.__pot_ind = self.__potential_peaks()
        self.nbands = len(self.__pot_ind)
        self.__p = p
        self.school = FishSchool(nfish, self.nbands, self.get_fitness(p), **(opt_params if opt_params else {}))
        self.school.init_boundaries(boundaries if boundaries is not None else self.__form_boundaries(pos_range, amp_range))
        self.mse_transform = mse_transform
        self.scheduler = Scheduler(self.school, 
        scheduler_sch if scheduler_sch is not None else {},
        cond=scheduler_regime, **sch_kw
        )
        self.school.scheduler = self.scheduler
    
    def __potential_peaks(self, eps=.1):
        spc = self.spectrum
        der4 = spc * 1
        der2 = spc * 1
        der4.get_derivative(4)
        der2.get_derivative(2)
        idx, ext = der4.get_extrema(minima=False)
        tmp = torch.stack(
            ext + (der2.get_extrema(minima=True)[1])
            ).sort().values
        sorted = (tmp[1:] - tmp[:-1]).sort()
        poses = tmp[sorted.indices[sorted.values < eps]]
        ws = spc.wavenums.tolist()
        idx = [bisect_left(ws, pos) for pos in poses]
        return idx
    
    def __form_boundaries(self, pos_range=.1, amp_range=0.5):
        spc, ind = self.spectrum, self.__pot_ind
        pos = spc.wavenums[ind]
        amps = spc.data[ind]
        # ['amp', 'w', 'x0', 'gau']
        boundaries = torch.zeros((2, self.nbands * 4))

        boundaries[1, :self.nbands] = amps 

        boundaries[0, self.nbands * 2 : self.nbands * 3] = pos * (1 - pos_range)
        boundaries[1, self.nbands * 2 : self.nbands * 3] = pos * (1 + pos_range)

        boundaries[1, 3 * self.nbands:] = 1.

        boundaries[1, self.nbands : 2 * self.nbands] = abs(spc.wavenums[-1] - spc.wavenums[0]) / self.nbands
        boundaries[0, self.nbands : 2 * self.nbands] = 0.01
        return boundaries
    
    def get_fitness(self, p=2):
        x, reference = self.x, self.reference
        def _fitness_f(params):
            approx = voigt_(x, *params.view(4, -1), True)
            se = (approx.sub_(reference).abs_()).pow_(p)
            mse = se.mean()
            if self.mse_transform == 'neg':
                return mse.neg_()
            elif self.mse_transform == 'neg_log':
                return mse.add_(1e-10).log_().neg_()
            elif self.mse_transform == 'reciprocal':
                return torch.reciprocal_(mse.add_(1e-10))
            elif self.mse_transform == 'max_neg':
                return se.max().neg()
            elif self.mse_transform == 'neg_root':
                return mse.sqrt_().neg_()
            else:
                raise NotImplementedError
        return _fitness_f
    
    def run(self, max_iter, init=True):
        self.school.run(max_iter, init)

    @property
    def result(self):
        return self.school.optimal_position

    def plot_history(self):
        plt.plot(self.school.history)
        plt.title('Fitness')
        plt.ylabel(f'{self.mse_transform} mse')
        plt.xlabel('Iteration')
        return plt.gca()

    def plot_comparison(self):
        plt.plot(self.x, voigt_(self.x, *self.school.optimal_position.view(4, -1)), label='recon')
        plt.plot(self.x, self.reference, label='orig')
        plt.xlabel('Wavenumber')
        plt.ylabel('Intensity')
        plt.legend()
        return plt.gca()
    
    def reset_fitness_func(self, method, p=2):
        self.mse_transform = method
        self.school._fitness_func = self.get_fitness(p)
        self.school._weights[:] = little_val
    
class Scheduler:
    def __init__(self, obj, param_coef: dict, cond=None, each_n: int=1, memory=5, checkout=''):
        self.target = obj
        self.i = 0
        self.cond = {
            None: lambda : True,
            'no_change': lambda : max(self.memory) - min(self.memory) < obj.tol and self.burnout <=0,
            'each_n': lambda : self.i % each_n == 0 and self.burnout <= 0
        }[cond]
        self.param_coef = param_coef
        self.memory = [0] * memory
        self.memory[0] = float('inf')
        self.checkout = checkout
        self.burnout = memory
    
    def step(self):
        self.i += 1
        self.burnout -= 1
        self.memory[self.i % len(self.memory)] = getattr(self.target, self.checkout)
        if self.cond():
            for param, coef in self.param_coef.items():
                setattr(self.target, param, getattr(self.target, param) * coef)
            self.burnout = len(self.memory)





