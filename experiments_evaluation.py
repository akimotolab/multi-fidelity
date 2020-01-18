# For CPU timing, call: OMP_NUM_THREADS=1 python experiments.py
import sys
import time

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from ddcma import DdCma, Checker, Logger
from constraint_handling import McrArchConstraintHandler
from compliance_minimization import ComplianceMinimization, NGNet
from adaptive_simulator_switcher import AbstractSimulator, AdaptiveSimulationSwitcher

class NGNetComplianceMinimization(AbstractSimulator):
    def __init__(self, i):
        """
        Parameters
        ----------
        n_basis_x, n_basis_y : int
            number of bases on each coordinate, n_basis_x * n_basis_y bases in total
        nelx, nely : int
            number of elements in each coordinate
        scale : float
            scale parameter of the Gaussian kernel
        volfrac : float, in [0, 1]
            total amount of material, V(x), is constrained to volfrac * V(1)
        eps : float, default 1e-3
            tolerance of equality constraint
        """
        self.nelx=12
        self.nely=4
        self.ng_x = 24
        self.ng_y = 8
        self.dim = self.ng_x * self.ng_y
        self.volfrac = 0.4
        self.co = ComplianceMinimization(self.nelx * i, self.nely * i, self.volfrac, penal=1.)
        self.ngnet = NGNet(self.ng_x, self.ng_y, self.nelx * i, self.nely * i, scale=1.0)
        
    def __call__(self, solution):
        """Compute the compliance and volume fraction
        
        Parameter
        ---------
        solution : object
            solution._x_repaired (input, 1d array-like) : height vector of NGNet, constrained in [-1, 1]
            solution.compliance (output, float)         : compliance
            solution.volution_fraction (output, float)  : volume fraction
            solution._f (output, float)                 : f(x)
            solution._g (output, 1d array-like)         : [g(x)]
        """
        fx, gx, _, _ = self.co(self.ngnet(solution._x_repaired))
        solution.compliance = fx
        solution.volume_fraction = gx
        solution._f = fx
        solution._g = [gx - self.volfrac]


class Solution:
    def __init__(self, x=None):
        self._x = x
        self._x_repaired = None
        self._f = None
        self._quak_penalty = None
        self._quak_violation = []
        self._qrsk_violation = []
    def clone(self):
        cl = Solution()
        cl._x = np.array(self._x, copy=True)
        cl._x_repaired = np.array(self._x_repaired, copy=True)
        cl._f = self._f
        cl._quak_penalty = self._quak_penalty
        cl._quak_violation = np.array(self._quak_violation, copy=True)
        cl._qrsk_violation = np.array(self._qrsk_violation, copy=True)
        return cl

def postprocess(dat):
    idx = dat[:, 0]
    time = dat[:, 1]
    ff = dat[:, 2]
    gg = dat[:, 3]
    xx = dat[:, 4:]
    reeval = np.empty((dat.shape[0], 7))
    reeval[:, 0] = idx[:]
    reeval[:, 1] = time[:]
    reeval[:, 2] = ff[:]
    reeval[:, 3] = gg[:]

    bestsofar = np.inf
    for i in tqdm(range(xx.shape[0])):
        sol = Solution(xx[i])
        sol._x_repaired = sol._x
        simulator(sol)
        reeval[i, 4] = sol._f
        reeval[i, 5] = sol._g[0]
        if sol._g[0] <= 0 and sol._f < bestsofar:
            bestsofar = sol._f
        reeval[i, 6] = bestsofar
    return reeval

simulator = NGNetComplianceMinimization(20)
seed_list = list(range(1, 11))
fidelity_list = [1, 4, 7, 11, 14, 17, 20]
lr_u_list = [1, 3 ,5, 7, 9]
maxsec = 86400
for fidelity in fidelity_list:
    for seed in seed_list:
        results = '../dat/ddcma_arch_mcr_fix{}_seed{}_maxsec{}.txt'.format(str(fidelity), str(seed), str(maxsec))
        reeval = postprocess(np.loadtxt(results))
        np.savetxt('../dat/ddcma_arch_mcr_fix{}_seed{}_maxsec{}_reeval.txt'.format(str(fidelity), str(seed), str(maxsec)), reeval)        

for lr_u in lr_u_list:
    for seed in seed_list:
        results = '../dat/ddcma_arch_mcr_adapt{}_seed{}_maxsec{}.txt'.format(str(lr_u), str(seed), str(maxsec))
        reeval = postprocess(np.loadtxt(results))
        np.savetxt('../dat/ddcma_arch_mcr_adapt{}_seed{}_maxsec{}_reeval.txt'.format(str(lr_u), str(seed), str(maxsec)), reeval) 

