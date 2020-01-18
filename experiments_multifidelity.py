# For CPU timing, call: OMP_NUM_THREADS=1 python experiments.py
import sys
import time

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


lr_u = float(sys.argv[1])
seed = (int(sys.argv[2]) + 123) * 456
maxsec = int(sys.argv[3])
results = '../dat/ddcma_arch_mcr_adapt{}_seed{}_maxsec{}.txt'.format(sys.argv[1], sys.argv[2], sys.argv[3])
np.random.seed(seed)

# number of simulators
m = 20 
# list of different approximate functions
func_list = [NGNetComplianceMinimization(i) for i in range(1, m + 1)]
# Create the switcher (function wrapper)
adaptf = AdaptiveSimulationSwitcher(func_list, tau_lr_u=lr_u/10.0) 

# Upper and Lower Bounds
N = adaptf.simulator_list[0].dim
lbound = -1.0 * np.ones(N)
ubound = 1.0 * np.ones(N)

# DD-CMA
ddcma = DdCma(xmean0=(lbound + ubound)/2.,
            sigma0=(ubound - lbound)/5., 
            flg_variance_update=True, 
            flg_covariance_update=True,
            flg_active_update=True)

# ARCH + MCR Constraint Handling
# volume_fraction can be treated as either inequality or equality qrsk constraint.
ch = McrArchConstraintHandler(dim=ddcma.N, 
                            weight=ddcma.w, 
                            fobjective=adaptf.f, 
                            bound=(lbound, ubound),
                            linear_ineq_quak=None, 
                            linear_eq_quak=None, 
                            nonlinear_ineq_quak_list=None, 
                            nonlinear_eq_quak_list=None,
                            ineq_qrsk_list=[adaptf.get_gi(0)], 
                            eq_qrsk_list=None
                            )

# Execution
with open(results, 'w') as f:
    pass
issatisfied = False
fbestsofar = np.inf
t_total = 0.0
while not issatisfied:
    idx_fidelity = adaptf.idx_current + 1
    #----------------------------------------------------
    t_start = time.perf_counter()
    xx, yy, zz = ddcma.sample()
    sol_list = [Solution(x) for x in xx]
    xcov = ddcma.transform(ddcma.transform(np.eye(N)).T)
    ch.prepare(ddcma.xmean, xcov)
    for sol in sol_list:
        ch.repair_quak(sol)
    adaptf.batcheval(sol_list)        
    for sol in sol_list:
        ch.evaluate_f_and_qrsk(sol)
    ranking = ch.total_ranking(sol_list)
    idx = np.argsort(ranking)
    ddcma.update(idx, xx, yy, zz)
    adaptf.update(sol_list, ch.do)
    t_end = time.perf_counter()
    t_total += t_end - t_start
    #----------------------------------------------------
    with open(results, 'a') as f:
        f.write(str(idx_fidelity) + ' ' + repr(t_total) + ' '
                + repr(sol_list[idx[0]]._f) + ' ' + repr(sol_list[idx[0]]._qrsk_violation[0]) + ' '
                + ' '.join([repr(x) for x in sol_list[idx[0]]._x_repaired])
                + '\n')
    print(idx_fidelity, t_total, sol_list[idx[0]]._f, sol_list[idx[0]]._qrsk_violation[0])

    ddcma.t += 1        
    ddcma.neval += ddcma.lam        
    ddcma.arf = np.array([sol._f for sol in sol_list])
    ddcma.arx = np.array([sol._x for sol in sol_list])

    if t_total > maxsec:
        break