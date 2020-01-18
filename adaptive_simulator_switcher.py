import math
import time
import numpy as np
from scipy.stats import kendalltau  # tau-b in scipy v1.1.0


class AbstractSimulator:
    """Abstract Simulator
    
    Required to implement: `__call__`
    Optional: `batcheval` to support parallel evaluation
    """

    def __call__(self, solution):
        """Evaluate the given solution
        
        Parameter
        ---------
        solution : object
            solution._x_repaired    : (input) QUAK-feasible solution
            solution._f             : (output) f(x)
            solution._g[i]          : (output) gi(x)
        """
        raise NotImplementedError

    def eval(self, solution):
        """Serial Evaluation"""
        self(solution)

    def batcheval(self, solution_list):
        """Parallel Evaluation"""
        for solution in solution_list:
            self(solution)

    def f(self, solution):
        if not hasattr(solution, '_f'):
            self(solution)
        return solution._f

    def get_gi(self, i):
        def g(solution):
            if not hasattr(solution, '_g'):
                self(solution)
            return solution._g[i]
        return g


class AdaptiveSimulationSwitcher:
    """Adaptive Switching Strategy 

    In simulation-basedf optimization we often have access to multiple 
    simulators or surrogate models that approximate a computationally 
    expensive or intractable objective function with different trade-offs
    between the fidelity and computational time. Such a setting is called
    multi-fidelity optimization. 

    This is a python code of the strategy proposed in the following reference
    to adaptively select which simulator to use during optimization of 
    comparison-based evolutionary algorithms. Our adaptive switching strategy 
    works as a wrapper of multiple simulators: optimization algorithms 
    optimize the wrapper function and the adaptive switching strategy selects 
    a simulator inside the wrapper.        
    
    Reference
    ---------
    Y. Akimoto, T. Shimizu, T. Yamaguchi
    Adaptive Objective Selection for Multi-Fidelity Optimization
    In Proceedings of Genetic and Evolutionary Computation Conference (2019)        
    """
    
    def __init__(self, simulator_list, err_list=None, runtime_estimation=True, beta=1.0, tau_thresh=0.5, tau_margin=0.2, tau_lr=0.1, tau_lr_u=0.5):
        """
        Parameters
        ----------
        simulator_list : list of instances of AbstractSimulator
            simulators sorted in the ascending order of expected runtime
            
        err_list : list of float [a1, ..., am-1]
            estimated runtime ratios, mi = runtime(fi+1) / runtime(fi)
            If there is not good estimated values, they can be all 0,
            and use `runtime_estimation = True`
            
        runtime_estimation : bool
            compute elapsed time in runtime to decide when to check Kendall's tau
        
        beta : float
            Kendall's tau is computed if the current f has spent beta times 
            the elapsed time for the next f
            
        tau_thresh : float 
            threshold for tau value to switch to the next level
        tau_margin : float
            threshold margin. 
        """
        if err_list is not None:
            assert len(simulator_list) - 1 == len(err_list)
        else:
            err_list = np.zeros(len(simulator_list) - 1)
            runtime_estimation = True
        self.simulator_list = simulator_list
        self.err_list = err_list
        self.cum_err_list = np.hstack(([1.], np.cumprod(err_list)))
        self.runtime_estimation = runtime_estimation
        self.beta = beta
        self.tau_thresh = tau_thresh
        self.tau_margin = tau_margin
        self.tau_avg_lower = 0.
        self.tau_avg_upper = 1.
        self.tau_lr = tau_lr
        self.tau_lr_u = tau_lr_u
        self.idx_current = 0
        self.t_current = 0
        self.t_next = 0
        self.t_previous = 0
        # The following values are used only for logging
        self._time_list = np.zeros((len(simulator_list), 3)) # T0 and T1 for each mode
        self._nfes_list = np.zeros((len(simulator_list), 3), dtype=int) # f0 and f1 FEs for each mode

    def eval(self, solution, idx=0):
        assert 0 <= self.idx_current + idx < len(self.simulator_list), "index out of bound"
        # evaluate current f
        sim = self.simulator_list[self.idx_current + idx]
        st = time.time()
        sim(solution)
        et = time.time()
        # keep elapsed time (or number of evaluations)
        elapsed = et - st if self.runtime_estimation else self.cum_err_list[self.idx_current + idx]
        solution._elapsed = elapsed
        # log
        self._time_list[self.idx_current, idx] += elapsed
        self._nfes_list[self.idx_current, idx] += 1
        # time
        if idx == 0:
            self.t_current += elapsed
        elif idx == -1:
            self.t_previous += elapsed
        elif idx == 1:
            self.t_next += elapsed

    def batcheval(self, solution_list, idx=0):
        assert 0 <= self.idx_current + idx < len(self.simulator_list), "index out of bound"
        # evaluate current f
        sim = self.simulator_list[self.idx_current + idx]
        st = time.time()
        sim.batcheval(solution_list)
        et = time.time()
        # keep elapsed time (or number of evaluations)
        elapsed = et - st if self.runtime_estimation else self.cum_err_list[self.idx_current + idx]
        # log
        self._time_list[self.idx_current, idx] += elapsed
        self._nfes_list[self.idx_current, idx] += len(solution_list)
        # time
        if idx == 0:
            self.t_current += elapsed
        elif idx == -1:
            self.t_previous += elapsed
        elif idx == 1:
            self.t_next += elapsed

    def f(self, solution, idx=0):
        return self.simulator_list[self.idx_current + idx].f(solution)

    def get_gi(self, i, idx=0):
        return self.simulator_list[self.idx_current + idx].get_gi(i)

    def update(self, solution_list, ranking_evaluator):
        """update the fidelity level
        
        Parameters
        ----------
        solution_list : list of object
            populations to be used to check whether the fidelity level should increase/decrease
            solution.clone() : (method) clone the solution
        ranking_evaluator : callable
            function to compute the final ranking after constraint handling
        """
        # initialize the first estimate of the runtime for next level
        if (self.t_next == 0 and self.idx_current + 1 < len(self.simulator_list)):
            self.t_next = self.err_list[self.idx_current] * self.t_current
        elif (self.t_next == 0 and self.idx_current + 1 == len(self.simulator_list)):
            self.t_next = np.inf
        if (self.t_previous == 0 and self.idx_current > 0 and self.err_list[self.idx_current - 1] > 0):
            self.t_previous = self.t_current / self.err_list[self.idx_current - 1]
        elif (self.t_previous == 0 and self.idx_current == 0):
            self.t_previous = np.inf

        if (self.beta * self.t_next < self.t_current):
            rank1 = ranking_evaluator(solution_list)
            # evaluate on an upper fidelity simulation
            mf_solution_list = [sol.clone() for sol in solution_list]
            self.batcheval(mf_solution_list, idx=1)
            rank2 = ranking_evaluator(mf_solution_list)
            # compute tau and update idx
            tau, p_value = kendalltau(rank1, rank2)
            self.tau_avg_upper += (tau - self.tau_avg_upper) * self.tau_lr_u
            if self.tau_avg_upper < self.tau_thresh:
                self.idx_current += 1
                self.t_current = self.t_next = self.t_previous = 0
                self.tau_avg_upper = 1.0
                self.tau_avg_lower = 0.0

        elif (self.beta * self.t_previous < self.t_current):
            rank1 = ranking_evaluator(solution_list)
            # evaluate on a lower fidelity simulation            
            mf_solution_list = [sol.clone() for sol in solution_list]
            self.batcheval(mf_solution_list, idx=-1)
            rank3 = ranking_evaluator(mf_solution_list)
            # compute tau and update idx
            tau, p_value = kendalltau(rank1, rank3)
            self.tau_avg_lower += (tau - self.tau_avg_lower) * self.tau_lr
            if self.tau_avg_lower > self.tau_thresh + self.tau_margin:
                self.idx_current -= 1
                self.t_current = self.t_next = self.t_previous = 0
                self.tau_avg_upper = 1.0
                self.tau_avg_lower = 0.0

if __name__ == '__main__':
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from ddcma import DdCma, Checker, Logger
    except:
        raise ImportError("https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518.js")

    def ranking_evaluator(solution_list):
        ff = [sol._f for sol in solution_list]
        n_better_f = np.asarray([np.sum(ff < f) for f in ff])
        n_equal_f = np.asarray([np.sum(ff == f) for f in ff])
        rff = n_better_f + (n_equal_f - 1) / 2.0
        return rff

    def f_true(x):
        return np.dot(x, x)

    class Solution:
        def __init__(self, x=None):
            self._x = x
            self._x_repaired = x
            self._f = None
        def clone(self):
            cl = Solution()
            cl._x = np.array(self._x, copy=True)
            cl._x_repaired = np.array(self._x_repaired, copy=True)
            cl._f = self._f
            return cl

    # Monte Calro Simulator
    # True objective f(x) = ||x||^2
    # Noisy objective fn(x) = f(x) + N(0, 1)
    # Approximate objective f_k = (1/k) * sum_{i=1}^{k} fn(x) = f(x) + N(0, 1/k)
    class MonteCarloSimulator(AbstractSimulator):
        def __init__(self, k):
            self.std = k**(-0.5)
        def __call__(self, solution):
            solution._f = f_true(solution._x_repaired) + np.random.randn(1)[0] * self.std
            solution._g = []
    # number of simulators
    m = 50 
    # list of different approximate functions
    func_list = [MonteCarloSimulator(2**i) for i in range(m)]
    # list of approximate runtime ratio (if available)
    err_list = [2] * (m-1)
    # err_list = None
    # Create the switcher (function wrapper)
    # Give this to your optimization algorithm as an objective function.
    adaptf = AdaptiveSimulationSwitcher(func_list, err_list, runtime_estimation=False) 

    # Main loop
    N = 10
    ddcma = DdCma(xmean0=np.random.randn(N), sigma0=np.ones(N)*2.)
    checker = Checker(ddcma)
    logger = Logger(ddcma)

    issatisfied = False
    fbestsofar = np.inf
    while not issatisfied:
        xx, yy, zz = ddcma.sample()

        sol_list = [Solution(x) for x in xx]
        adaptf.batcheval(sol_list)
        f_list = [adaptf.f(sol) for sol in sol_list]
        adaptf.update(sol_list, ranking_evaluator)
        idx = np.argsort(f_list)
        ddcma.update(idx, xx, yy, zz)

        ddcma.t += 1        
        ddcma.neval += ddcma.lam        
        ddcma.arf = np.array(f_list)
        ddcma.arx = xx
        fbest = np.min(ddcma.arf)      
        fbestsofar = min(fbest, fbestsofar)
        if fbest < -np.inf:
            issatisfied, condition = True, 'ftarget'
        elif ddcma.t > 1000:
            issatisfied, condition = True, 'maxiter'
        else:
            issatisfied, condition = checker()
        if ddcma.t % 10 == 0:
            print(ddcma.t, ddcma.neval, fbest, fbestsofar)
            logger()
    print(ddcma.t, ddcma.neval, fbest, fbestsofar)
    print("Terminated with condition: " + condition)
    logger(condition)

    # Produce a figure
    fig, axdict = logger.plot()
    for key in axdict:
        if key not in ('xmean'):
            axdict[key].set_yscale('log')
    plt.savefig(logger.prefix + '.pdf', tight_layout=True)