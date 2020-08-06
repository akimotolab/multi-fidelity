from scipy import optimize
from scipy.stats import kendalltau  # tau-b in scipy v1.1.0
from scipy.stats import norm
import numpy as np
import gc

class McrArchConstraintHandler:
    
    @staticmethod
    def lin2nonlin(A, b):
        def gi(i):
            def g(x):
                return np.dot(A[i], x) - b[i]
            def grad_g(x):
                return A[i]
            return (g, grad_g)
        return [gi(i) for i in range(len(b))]

    @staticmethod
    def box2lin(lb, ub):
        dim = len(lb)
        A_box_lb = np.eye(dim) * -1.
        A_box_ub = np.eye(dim)
        A_box = np.block([[A_box_lb], [A_box_ub]])
        b_box_lb = - lb
        b_box_ub = ub
        b_box = np.block([b_box_lb, b_box_ub])
        return A_box, b_box

    def __init__(self, dim, weight, fobjective, bound=None, 
                 linear_ineq_quak=None, linear_eq_quak=None, 
                 nonlinear_ineq_quak_list=None, nonlinear_eq_quak_list=None,
                 ineq_qrsk_list=None, eq_qrsk_list=None):
        """
        Parameters
        ----------
        dim : int
        weight : 1d array-like
        fobjective : callable (a solution object -> float)
            f(x) 
        bound : tuple (lb, ub)
            lb[i] <= x[i] <= ub[i]
        linear_ineq_quak : tuple (A, b)
            A * x - b <= 0, A: 2D, b: 1D
            If both linear_quak and nonlinear_quak_list are given, 
            linear_quak is transformed to nonlinear_quak internally.
        linear_eq_quak : tuple (A, b)
            A * x - b = 0, A: 2D, b: 1D
            If both linear_(ineq/eq)_quak and nonlinear_(ineq/eq)_quak_list are given, 
            linear_(ineq/eq)_quak is transformed to nonlinear_(ineq/eq)_quak internally.
        nonlinear_ineq_quak_list : list of tuple (gi, grad_gi)
            gi(x) <= 0 : argument is an array
            If gradient of gi is not available, set `None` to grad_gi.
        nonlinear_eq_quak_list : list of tuple (gi, grad_gi)
            gi(x) = 0 : argument is an array
            If gradient of gi is not available, set `None` to grad_gi.
        ineq_qrsk_list : list of callable (a solution object -> float)
            gi(x) <= 0 
        eq_qrsk_list : list of callable (a solution object -> float)
            gi(x) == 0 
        """
        # Input Preprocessing
        self.f = fobjective

        # Linear Ineq. QUAK
        if linear_ineq_quak is not None:
            self.linear_ineq_quak = linear_ineq_quak
        else:
            self.linear_ineq_quak = (np.empty((0, dim)), np.empty(0))
        if bound is not None:
            A, b = self.box2lin(bound[0], bound[1])
            self.linear_ineq_quak = (np.vstack((A, self.linear_ineq_quak[0])), np.hstack((b, self.linear_ineq_quak[1])))
        
        # Linear Eq. QUAK
        if linear_eq_quak is not None:
            self.linear_eq_quak = linear_eq_quak
        else:
            self.linear_eq_quak = (np.empty((0, dim)), np.empty(0))

        # Nonlinear Ineq. QUAK
        if nonlinear_ineq_quak_list is not None:
            self.nonlinear_ineq_quak_list = nonlinear_ineq_quak_list
        else:
            self.nonlinear_ineq_quak_list = []
        
        # Nonlinear Eq. QUAK
        if nonlinear_eq_quak_list is not None:
            self.nonlinear_eq_quak_list = nonlinear_eq_quak_list
        else:
            self.nonlinear_eq_quak_list = []

        # Linear to Nonlinear
        if len(self.nonlinear_ineq_quak_list) + len(self.nonlinear_eq_quak_list) > 0:
            if len(self.linear_ineq_quak[1]) + len(self.linear_eq_quak[1]) > 0:
                self.nonlinear_ineq_quak_list += self.lin2nonlin(*self.linear_ineq_quak)
                self.nonlinear_eq_quak_list += self.lin2nonlin(*self.linear_eq_quak)
                self.linear_ineq_quak = (np.empty((0, dim)), np.empty(0))
                self.linear_eq_quak = (np.empty((0, dim)), np.empty(0))

        # QRSK
        if ineq_qrsk_list is not None:
            self.ineq_qrsk_list = ineq_qrsk_list
        else:
            self.ineq_qrsk_list = []
        if eq_qrsk_list is not None:
            self.eq_qrsk_list = eq_qrsk_list
        else:
            self.eq_qrsk_list = []
        
        # Setup QUAK Constraint Handler
        if len(self.linear_ineq_quak[1]) + len(self.linear_eq_quak[1]) > 0:
            self.quak_handler = ARCHLinear(self.linear_ineq_quak[0], self.linear_ineq_quak[1], self.linear_eq_quak[0], self.linear_eq_quak[1], weight)
        elif len(self.nonlinear_ineq_quak_list) + len(self.nonlinear_eq_quak_list) > 0:
            self.quak_handler = ARCHNonLinear(dim, weight, self.nonlinear_ineq_quak_list, self.nonlinear_eq_quak_list)
        else:
            self.quak_handler = ARCHBase(dim, weight, 0, 0)
        # Setup QRSK Constraint Handler
        if len(self.ineq_qrsk_list) + len(self.eq_qrsk_list) > 0:
            # self.qrsk_handler = MCR(self.ineq_qrsk_list, self.eq_qrsk_list)
            self.qrsk_handler = MCRMOD(self.ineq_qrsk_list, self.eq_qrsk_list)
            # self.qrsk_handler = AMCR(self.ineq_qrsk_list, self.eq_qrsk_list)
            # self.qrsk_handler = MCRMAX(self.ineq_qrsk_list, self.eq_qrsk_list)
        else:
            self.qrsk_handler = AMCRBase(self.ineq_qrsk_list, self.eq_qrsk_list)

    # TODO: only for debug
    def setcma(self, cma):
        self._cma = cma
    def __call__(self, solution_list):
        for sol in solution_list:
            sol._x = sol.arxraw
        xcov = self._cma.transform(self._cma.transform(np.eye(self._cma.N)))
        self.prepare(self._cma.xmean, xcov)
        arfit = self.do(solution_list)
        for i, sol in enumerate(solution_list):
            sol.arfit = arfit[i]
            sol.arffeas = sol._f
      
    def do(self, solution_list):
        """Do all the job"""
        for sol in solution_list:
            self.repair_quak(sol)
            self.evaluate_f_and_qrsk(sol)
        return self.total_ranking(solution_list)

    def prepare(self, xmean, xcov):
        self.quak_handler.prepare(xmean, xcov)
        
    def repair_quak(self, solution):
        """Evaluate QUAK constraint violations and repair
        
        Parameter
        ---------
        solution : object
            solution._x (input) : original solution
            solution._x_repaired (output) : repaired solution
            solution._quak_violation (output) : list of constraint violations
            solution._quak_penalty (output) : penalty for quak violation
        """
        if self.quak_handler:
            solution._x_repaired = self.quak_handler.repair(solution._x)
            solution._quak_violation = self.quak_handler.compute_violation(solution._x)
            solution._quak_penalty = self.quak_handler.compute_penalty(solution._x, solution._x_repaired)
        else:
            solution._x_repaired = solution._x
            solution._quak_violation = []
            solution._quak_penalty = 0
            
    def evaluate_f_and_qrsk(self, solution):
        """Evaluate objective and QRSK constraint violations
        
        Parameter
        ---------
        solution : object
            solution._x_repaired (input) : repaired solution
            solution._f (output) : objective value
            solution._qrsk_violation (output) : list of constraint violations
        """
        if np.all(self.quak_handler.compute_violation(solution._x_repaired) <= 0):
            solution._f = self.f(solution)
            solution._qrsk_violation = self.qrsk_handler.compute_violation(solution)
        else:
            solution._f = np.inf
            solution._qrsk_violation = [np.inf]

    def total_ranking(self, solution_list):
        """Compute the total ranking
        
        Parameter
        ---------
        solution_list : list of object
            solution._f (input) : objective value
            solution._qrsk_violation (input) : list of constraint violations
            solution._quak_penalty (input) : penalty for quak violation

        Return
        ------
        final ranking : list of ranking (float)
        """
        ff = [sol._f for sol in solution_list]
        rs = [sol._qrsk_violation for sol in solution_list]
        ua = [sol._quak_penalty for sol in solution_list]
        rt = self.qrsk_handler.total_ranking(ff, rs)
        rt = self.quak_handler.total_ranking(rt, ua)
        return rt
        

class ARCHBase:
    """Adaptive Ranking Based Constraint Handling for Explicit Constraints
    
    It is a base (abstract) implementation of Quantifiable/Unrelaxable/Apriori/Known 
    equality and inequality constraints. 
    
    Main Functionarity
    ------------------
    * compute_violation : compute violation values
    * prepare       : set the current distribution and update the penalty factor
    * repair        : repair infeasible solutions
    * compute_penalty : compute the mahalanobis distance betwen original and repaired solutions
    * total_ranking : compute total ranking
    
    Reference
    ---------
    N. Sakamoto and Y. Akimoto: Adaptive Ranking Based Constraint Handling 
    for Explicitly Constrained Black-Box Optimization, GECCO 2019. 
    https://doi.org/10.1145/3321707.3321717
    
    """
    def __init__(self, dim, weight, num_of_eqcons, num_of_ineqcons, 
                 eps_min=1e-15, eps_max=1e-4, tol_for_eqcons=1e-4, tol_for_ineqcons=0.0, maxiter=300):
        """
        Parameters
        ----------
        dim : int
            dimension of the search space
        weight : array-like
            recombination weights
        num_of_eqcons, num_of_ineqcons : int
            number of equality and inequality constraints
        
        Optional Parameters
        -------------------
        eps_min, eps_max : float
            the repaired point is forced to be inside from the boundary with margin eps.
            the margin is adapted with in the range [eps_min, eps_max].
        tol_for_eqcons, tol_for_ineqcons : float
            the tolerance for equality and inequality constraint violations
            tol_for_eqcons must be small positive number, while tol_for_ineqcons can be 0.
        maxiter : int
            the maximum number of iterations for the internal optimization process,
            given to SLSQP.
        """
        ww = np.array(weight, copy=True)
        ww[ww < 0] = 0
        ww /= np.sum(np.abs(ww))
        self.N = dim
        self.lam_def = 4. + np.floor(3. * np.log(self.N))
        self.lam = len(ww)
        self.mueff = 1.0 / np.sum(ww**2)
        self.sqreSNF = sigma_normalization_factor(self.N, ww)**2

        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps = self.eps_min
        self.num_of_fail = 0
        self.maxiter = maxiter
        
        self.tol_for_eqcons = tol_for_eqcons
        self.tol_for_ineqcons = tol_for_ineqcons
        self.num_of_ineq = num_of_ineqcons
        self.num_of_eq = num_of_eqcons

        self.alpha = 1.0
        self.d_alpha = 1.0 / self.N
        self.dm_threshold = 1.0
        
        self._xmean = None
        self._sqrtC = None
        self._sqrtCinv = None

    def compute_violation(self, x):
        """Compute the violation
        
        Parameters
        ----------
        x : 1d or 2d array-like
        
        Returns
        -------
        violation (float) : g(x) - tol_for_ineqcons for inequality constraints, 
                            abs(g(x)) - tol_for_eqcons for equality constraints
        """
        if np.ndim(x) == 1:
            return np.empty((0))
        elif np.ndim(x) == 2:
            return np.empty((x.shape[0], 0))
        else:
            raise ValueError
    
    def prepare(self, xmean, xcov):
        """Preparation 
 
        Parameters
        ----------
        xmean : 1d array-like
            mean vector of the sampling distribution
        xcov : 2d array-like
            covariance matrix of the sampling distribution
            x sim N(xmean, xcov) sim xmean + sqrt(xcov) * N(0, I)
        """
        self._xmean = xmean
        D, B = np.linalg.eigh(xcov)
        self._sqrtC = np.dot(B * np.sqrt(D), B.T)
        self._sqrtCinv = np.dot(B / np.sqrt(D), B.T)
        self._update_alpha()
    
    def _update_alpha(self):
        """Update alpha (Section 4.3)"""
        if np.all(self.compute_violation(self._xmean) <= 0):
            self.dm = 0.0
        else:
            mfeas, J_done = self.repair(self._xmean, info=True, is_xmean=True)
            num_of_act = len(J_done) + self.num_of_eq
            N = len(self._xmean)
            self.dm = self.compute_penalty(self._xmean, mfeas)
            self.dm *= self.sqreSNF  # sqreSNF = (optimal normalized step-size / n)^2
            self.dm *= 2. * N / (N + 2. * num_of_act)
            self.dm *= np.exp(self.lam_def / self.lam - 1.0)
        
        if not hasattr(self, 'dm_old'):
            self.dm_old = 0.0  # Initialization
        if np.sign(self.dm - self.dm_old) == np.sign(self.dm - self.dm_threshold) or self.dm == 0.0:
            self.alpha *= np.exp(np.sign(self.dm - self.dm_threshold) * self.d_alpha)
        self.dm_old = self.dm
        self.alpha = np.clip(self.alpha, 1.0 / self.lam, self.lam) #TODO: lower and upper should be independent of lam

        # for next iteration repair procedure
        if self.num_of_fail <= np.ceil(self.lam / 10):
            self.eps /= 2.
        else:
            self.eps *= 10.
        self.eps = np.clip(self.eps, self.eps_min, self.eps_max)
        self.num_of_fail = 0
    
    def repair(self, x, info=False, is_xmean=False):
        """Repair infeasible solutions and return feasible solutions (Section 4.1)
        
        Parameters
        ----------
        x : 1d or 2d array-like
            solutions or list of solutions to be repaired
        
        Returns
        -------
        X (same shape as x) : repaired solutions
        """
        if info:
            if np.ndim(x) == 1:
                return x, []
            elif np.ndim(x) == 2:
                return x, [[] * len(x)]
            else:
                raise ValueError
        else:
            return self.repair(x, info=True, is_xmean=is_xmean)[0]

    def compute_penalty(self, x, x_repaired):
        """Compute the Maharanobis distance between x and x_repaired"""
        assert x.shape == x_repaired.shape
        if np.array_equal(x, x_repaired):
            return 0
        else:
            dx = x - x_repaired
            dz = np.dot(dx, self._sqrtCinv)
            pen = np.sum(dz * dz, axis=-1)
            return pen
    
    def total_ranking(self, f_list, penalty_list):
        ff = np.asarray(f_list)
        pp = np.asarray(penalty_list)
        n_better_f = np.asarray([np.sum(ff < f) for f in ff])
        n_equal_f = np.asarray([np.sum(ff == f) for f in ff])
        n_better_p = np.asarray([np.sum(pp < p) for p in pp])
        n_equal_p = np.asarray([np.sum(pp == p) for p in pp])
        rff = n_better_f + (n_equal_f - 1) / 2.0
        rpp = n_better_p + (n_equal_p - 1) / 2.0
        return rff + self.alpha * rpp
        # return rff + rpp
 

class ARCHLinear(ARCHBase):
    """Adaptive Ranking Based Constraint Handling for Explicit Constraints
    
    It is an implementation of Quantifiable/Unrelaxable/Apriori/Known 
    linear equality and inequality constraints. 
    
    Main Functionarity
    ------------------
    * compute_violation : compute violation values
    * prepare       : set the current distribution and update the penalty factor
    ** _update_alpha : update the penalty factor
    * repair        : repair infeasible solutions
    ** _nearest_feasible : find the nearest feasible solution in terms of MD
    * compute_penalty : compute the mahalanobis distance betwen original and repaired solutions
    * total_ranking : compute total ranking
    
    Reference
    ---------
    N. Sakamoto and Y. Akimoto: Adaptive Ranking Based Constraint Handling 
    for Explicitly Constrained Black-Box Optimization, GECCO 2019. 
    https://doi.org/10.1145/3321707.3321717
    
    """
    def __init__(self, matA, vecb, matAeq, vecbeq, weight, **kwargs):
        """Linear constraint handler for A * x <= b

        Parameters
        ----------
        matA : 2d array-like
        vecb : 1d array-like
            A and b in A * x <= b
        matAeq : 2d array-like
        vecbeq : 1d array-like
            A and b in A * x == b
        weight : 1d array-like
            recombination weights for CMA-ES 
        kwargs : dict
            optional parameters for ARCHBase
        """
        super(ARCHLinear, self).__init__(len(matA[0]), weight, len(vecbeq), len(vecb), **kwargs)
        Ai = np.asarray(matA, dtype=float).reshape(-1, self.N)
        Ae = np.asarray(matAeq, dtype=float).reshape(-1, self.N)
        bi = np.asarray(vecb, dtype=float).ravel()
        be = np.asarray(vecbeq, dtype=float).ravel()
        self.A = np.vstack((Ai, Ae, -Ae))
        self.b = np.hstack((bi + self.tol_for_ineqcons, be + self.tol_for_eqcons, -be + self.tol_for_eqcons))

    def compute_violation(self, x):
        """Compute the violation
        
        Parameters
        ----------
        x : 1d or 2d array-like
        
        Returns
        -------
        violation (float) : A * x - b - tolerance for inequality constraints,
                            abs(A * x - b) - tolerance for equality constraints
        """
        i = self.num_of_ineq
        ie = self.num_of_ineq + self.num_of_eq
        vi = np.dot(x, self.A[:i].T) - self.b[:i]
        ve1 = np.dot(x, self.A[i:ie].T) - self.b[i:ie]
        ve2 = np.dot(x, self.A[ie:].T) - self.b[ie:]
        v = np.hstack((vi, np.fmax(ve1, ve2)))
        return v
            
    def repair(self, x, info=False, is_xmean=False):
        """Repair infeasible solutions and return feasible solutions (Section 4.1)
        
        Note that Eq (5) always finds a solution as long as constraints are redundant.
        Henc Eq (6) is not implemented.
        
        Parameters
        ----------
        x : 1d or 2d array-like
            solutions or list of solutions to be repaired
        info : bool, optional
        is_xmean : bool, optional
        
        Returns
        -------
        X (same shape as x) : repaired solutions
        J_done (list of int) : list of index of active inequality constraints
        """
        if info:
            if np.ndim(x) == 2:
                tmp = [self.repair(x[i], info=True, is_xmean=is_xmean) for i in range(x.shape[0])]
                X = np.array([tmp[i][0] for i in range(len(tmp))])
                J_done = np.array([tmp[i][1] for i in range(len(tmp))])
                return X, J_done[J_done < self.num_of_ineq]
            elif np.ndim(x) == 1:
                if np.all(self.compute_violation(x) <= 0):
                    return x, []
                else:
                    res = self._nearest_feasible(x)
                    if np.any(self.compute_violation(res[0]) > 0) and not is_xmean:
                        self.num_of_fail += 1
                    return res
            else:
                raise ValueError
        else:
            return self.repair(x, info=True, is_xmean=is_xmean)[0]
        
    def _nearest_feasible(self, x):
        """Find the nearest feasible solution in terms of the Euclidean distance

        Parameters
        ----------
        x : ndarray (1D)

        Returns
        -------
        xnear : ndarray
            Repaired x. If `out` is passed, its reference is returned.

        J_done : list
            list of index of active inequality constraints
        """

        assert (np.ndim(x) == 1)
        dim = self.A.shape[-1]
        num_of_constraint = self.A.shape[0]
        J_done = []
        ynear = np.dot(self._sqrtCinv, x)
        A = np.dot(self.A, self._sqrtC)
        b = self.b - self.eps
        pinvA = np.empty((0, dim))
        for _ in range(num_of_constraint):
            violation = np.dot(A, ynear) - b
            # Termination check
            if np.all(violation <= self.eps):
                break
            # index of violated or handled constraints
            J = [j for j in range(num_of_constraint)
                 if j in J_done or violation[j] >= self.eps]
            pinvA = np.linalg.pinv(A[J])
            ynear -= np.dot(pinvA, violation[J])
            J_done = J.copy()
        J_done = np.asarray(J_done)
        return np.dot(self._sqrtC, ynear), J_done[J_done < self.num_of_ineq]


class ARCHNonLinear(ARCHBase):
    """ARCH for Nonlinear Explicit Constraints (ECJ submitted version)"""
    
    def __init__(self, dim, weight, ineq_list, eq_list, **kwargs):
        super(ARCHNonLinear, self).__init__(dim, weight, len(eq_list), len(ineq_list), **kwargs)
        def gi(g):
            return lambda x: g(x) + self.eps - self.tol_for_ineqcons
        def slsqp_gi(gi):
            return lambda y: -gi(y + self._xmean)
        def slsqp_grad_gi(grad_g):
            return lambda y: -grad_g(y + self._xmean) if grad_g is not None else None
        
        # Inequality constraint list
        self.ineq_list = [(gi(g[0]), g[1]) for g in ineq_list]
        self.slsqp_ineq_func_list = [slsqp_gi(g[0]) for g in self.ineq_list]
        self.slsqp_ineq_grad_list = [slsqp_grad_gi(g[1]) for g in ineq_list]

        # Equality constraint list
        self.eq_list = eq_list
        self.slsqp_eq_func_list = [slsqp_gi(g[0]) for g in self.eq_list]
        self.slsqp_eq_grad_list = [slsqp_grad_gi(g[1]) for g in eq_list]

    def compute_violation(self, x):
        """Compute the violation
        
        Parameters
        ----------
        x : 1d or 2d array-like
        
        Returns
        -------
        violation (float) : g(x) - tol_for_ineqcons for inequality constraints, 
                            abs(g(x)) - tol_for_eqcons for equality constraints
        """
        if np.ndim(x) == 1:
            violation = [gi[0](x) - self.eps for gi in self.ineq_list]
            violation += [np.abs(gi[0](x)) - self.tol_for_eqcons for gi in self.eq_list]
            return np.asarray(violation)
        elif np.ndim(x) == 2:
            return np.asarray([self.compute_violation(xi) for xi in x])
        else:
            raise ValueError


    def repair(self, x, info=False, is_xmean=False):    
        """Repair infeasible solutions and return feasible solutions (Section 4.1)
        
        Parameters
        ----------
        x : 1d or 2d array-like
            solutions or list of solutions to be repaired
        info : bool, optional
        is_xmean : bool, optional
        
        Returns
        -------
        X (same shape as x) : repaired solutions
        J_done (list of int) : list of index of active inequality constraints
        """    
        if info:
            violation = self.compute_violation(x)
            if np.all(violation <= 0):
                return x, []
            else:
                # index of violated or handled constraints
                J_eq = list(np.argwhere(violation[:self.num_of_ineq] > 0.0).flatten())
                J_ineq = []

                yinfeas = np.array(x - self._xmean, copy=True)
                def f(y):
                    dz = np.dot(self._sqrtCinv, y - yinfeas)
                    return np.dot(dz, dz)
                def derivative(y):
                    return 2.0 * np.dot(self._sqrtCinv, np.dot(self._sqrtCinv, y - yinfeas))

                # repair on boundaries
                for _ in range(self.num_of_ineq):
                    ynear = self._nearest_feasible_in_y(y=yinfeas,
                                                        f=f,
                                                        derivative=derivative,
                                                        J_eq=J_eq,
                                                        J_ineq=J_ineq)

                    # check termination
                    violation = self.compute_violation(ynear + self._xmean)
                    if np.all(violation <= 0):
                        del f, derivative
                        gc.collect()
                        return self._xmean + ynear, J_eq + J_ineq
                    else:
                        tmp = J_ineq + [j for j in np.argwhere(violation[:self.num_of_ineq] > 0.0).flatten() if j not in J_eq]
                        if len(tmp) == len(J_ineq):
                            break
                        else:
                            J_ineq = tmp

                # repair to nearest feasible domain
                J_ineq = J_eq # TODO: check
                J_eq = []
                for _ in range(self.num_of_ineq):
                    ynear = self._nearest_feasible_in_y(y=yinfeas,
                                                        f=f,
                                                        derivative=derivative,
                                                        J_eq=J_eq,
                                                        J_ineq=J_ineq)

                    # check termination
                    violation = self.compute_violation(ynear + self._xmean)
                    if np.all(violation <= 0):
                        del f, derivative
                        gc.collect()
                        return self._xmean + ynear, J_ineq
                    else:
                        tmp = J_ineq + [j for j in np.argwhere(violation[:self.num_of_ineq] > 0.0).flatten()]
                        if len(tmp) == len(J_ineq):
                            break
                        else:
                            J_ineq = tmp

                if not is_xmean:
                    self.num_of_fail += 1
                    
                del f, derivative
                gc.collect()
                return self._xmean + ynear, J_ineq
            
        else:
            return self.repair(x, info=True)[0]

    def _nearest_feasible_in_y(self, y, f, derivative, J_eq, J_ineq):
        # Set the parameters that SLSQP will need
        # meq, mieq: number of equality and inequality constraints
        meq = len(J_eq)
        mieq = len(J_ineq)
        # m = The total number of constraints
        m = meq + mieq
        # la = The number of constraints, or 1 if there are no constraints
        la = np.array([1, m]).max()
        # n = The number of independent variables
        n = self.N

        # Define the workspaces for SLSQP
        n1 = n + 1
        mineq = m - meq + n1 + n1
        len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
                + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
        if len_w <= 0:
            return y
        else:
            cons = tuple({'type': 'eq', 
                          'fun': self.slsqp_eq_func_list[j], 
                          'jac': self.slsqp_eq_grad_list[j]}
                         for j in range(self.num_of_eq))
            cons += tuple({'type': 'eq',
                           'fun': self.slsqp_ineq_func_list[j],
                           'jac': self.slsqp_ineq_grad_list[j]}
                          for j in J_eq)
            cons += tuple({'type': 'ineq',
                           'fun': self.slsqp_ineq_func_list[j],
                           'jac': self.slsqp_ineq_grad_list[j]}
                          for j in J_ineq)

            try:
                ynear = optimize.minimize(fun=f, 
                                          x0=y, 
                                          jac=derivative, 
                                          constraints=cons,
                                          bounds=None,
                                          method="SLSQP", 
                                          options={'maxiter': self.maxiter}).x
                assert np.all(np.isfinite(ynear))
            except:
                ynear = optimize.minimize(fun=f, 
                                          x0=y, 
                                          jac=derivative, 
                                          constraints=cons,
                                          bounds=None,
                                          method="SLSQP", 
                                          options={'maxiter': self.maxiter//20}).x

            del cons
            gc.collect()
            return ynear


class AMCRBase:
    def __init__(self, ineq_qrsk_list, eq_qrsk_list, tol_for_eqcons=1e-4, tol_for_ineqcons=0.0):
        self.ineq_qrsk_list = ineq_qrsk_list
        self.eq_qrsk_list = eq_qrsk_list
        self.tol_for_ineqcons = tol_for_ineqcons
        self.tol_for_eqcons = tol_for_eqcons
    def compute_violation(self, solution):
        return []
    def total_ranking(self, f_list, qrsk_list):
        ff = np.asarray(f_list)
        n_better_f = np.asarray([np.sum(ff < f) for f in ff])
        n_equal_f = np.asarray([np.sum(ff == f) for f in ff])
        rff = n_better_f + (n_equal_f - 1) / 2.0
        return rff


class MCR(AMCRBase):
    """Original MCR
    
    Main functionalities are
    * compute_violation : compute the qrsk constraint violation value list
    * total_ranking : compute the total ranking of a solution list
    """
    def __init__(self, ineq_qrsk_list, eq_qrsk_list, **kwargs):
        super(MCR, self).__init__(ineq_qrsk_list, eq_qrsk_list, **kwargs)

    def compute_violation(self, solution):
        violation = [g(solution) - self.tol_for_ineqcons for g in self.ineq_qrsk_list]
        violation += [g(solution) - self.tol_for_eqcons for g in self.eq_qrsk_list]
        return violation

    def total_ranking(self, f_list, qrsk_list):
        ff = np.asarray(f_list)
        gg = np.asarray(qrsk_list)
        gg[gg < 0] = 0
        nvv = np.sum(gg > 0, axis=1)
        n_better_f = np.asarray([np.sum(ff < f) for f in ff])
        n_better_gg = np.asarray([np.sum((gg < g), axis=0) for g in gg])
        n_better_nvv = np.asarray([np.sum((nv < nvv), axis=0) for nv in nvv])
        rff = n_better_f
        rgg = n_better_gg
        rnv = n_better_nvv
        if np.any(nvv == 0):
            return rff + np.sum(rgg, axis=1) + rnv
        else:
            return np.sum(rgg, axis=1) + rnv


class MCRMOD(AMCRBase):
    """MCR-mod (GECCO 2019)"""
    def __init__(self, ineq_qrsk_list, eq_qrsk_list, **kwargs):
        super(MCRMOD, self).__init__(ineq_qrsk_list, eq_qrsk_list, **kwargs)

    def compute_violation(self, solution):
        violation = [g(solution) - self.tol_for_ineqcons for g in self.ineq_qrsk_list]
        violation += [g(solution) - self.tol_for_eqcons for g in self.eq_qrsk_list]
        return violation

    def total_ranking(self, f_list, qrsk_list):
        ff = np.asarray(f_list)
        gg = np.asarray(qrsk_list)
        gg[gg < 0] = 0
        nvv = np.sum(gg > 0, axis=1)
        n_better_f = np.asarray([np.sum(ff < f) for f in ff])
        n_better_gg = np.asarray([np.sum((gg < g), axis=0) for g in gg])
        n_better_nvv = np.asarray([np.sum((nv < nvv), axis=0) for nv in nvv])
        rff = n_better_f
        rgg = n_better_gg
        rnv = n_better_nvv
        beta = np.sum(nvv == 0) / len(nvv)
        return beta * rff + (1 - beta) * (np.mean(rgg, axis=1) + rnv)


class AMCR(AMCRBase):
    """Itoh MCR
    
    Main functionalities are
    * compute_violation : compute the qrsk constraint violation value list
    * total_ranking : compute the total ranking of a solution list
    """
    def __init__(self, ineq_qrsk_list, eq_qrsk_list, **kwargs):
        super(AMCR, self).__init__(ineq_qrsk_list, eq_qrsk_list, **kwargs)
        self.beta = np.ones(len(ineq_qrsk_list) + len(eq_qrsk_list))

    def compute_violation(self, solution):
        violation = [g(solution) - self.tol_for_ineqcons for g in self.ineq_qrsk_list]
        violation += [g(solution) - self.tol_for_eqcons for g in self.eq_qrsk_list]
        return violation

    def total_ranking(self, f_list, qrsk_list):
        ff = np.asarray(f_list)
        gg = np.asarray(qrsk_list)
        tau = np.asarray([kendalltau(ff, gg[:, i])[0] for i in range(gg.shape[1])])
        #beta-->
        freq = np.mean(gg > 0, axis=0)
        self.beta = np.clip(self.beta * np.exp(freq - 0.9), 1.0, 10.0)
        #>--beta
        n_better_f = np.asarray([np.sum(ff < f) for f in ff])
        n_equal_f = np.asarray([np.sum(ff == f) for f in ff])
        n_better_gg = np.asarray([np.sum((gg < g) * (0 < gg), axis=0) for g in gg])
        n_equal_gg = np.asarray([np.sum((gg == g) * (0 < gg), axis=0) for g in gg])
        rff = n_better_f + (n_equal_f - 1) / 2.0
        rgg = n_better_gg + (n_equal_gg - 1) / 2.0
        return rff + np.dot(rgg, self.beta - tau)


class MCRMAX(AMCRBase):
    """MCR using Worst Constraint Violation
    
    Main functionalities are
    * compute_violation : compute the qrsk constraint violation value list
    * total_ranking : compute the total ranking of a solution list
    """
    def __init__(self, ineq_qrsk_list, eq_qrsk_list, **kwargs):
        super(MCRMAX, self).__init__(ineq_qrsk_list, eq_qrsk_list, **kwargs)

    def compute_violation(self, solution):
        violation = [g(solution) - self.tol_for_ineqcons for g in self.ineq_qrsk_list]
        violation += [g(solution) - self.tol_for_eqcons for g in self.eq_qrsk_list]
        return violation

    def total_ranking(self, f_list, qrsk_list):
        ff = np.asarray(f_list)
        gg = np.asarray(qrsk_list)

        n_better_f = np.asarray([np.sum(ff < f) for f in ff])
        n_equal_f = np.asarray([np.sum(ff == f) for f in ff])
        n_better_gg = np.asarray([np.sum((gg < g) * (0 < gg), axis=0) for g in gg])
        n_equal_gg = np.asarray([np.sum((gg == g) * (0 < gg), axis=0) for g in gg])

        rff = n_better_f + n_equal_f / 2.0
        rgg = n_better_gg + n_equal_gg / 2.0
        rggidx = np.argmax(rgg, axis=1)
        return np.fmax(rff, rgg[rggidx])

class NormalOrderStatistics(object):
    """Compute Moments of Normal Order Statistics

    Requires
    --------
    numpy
    scipy.stats.norm
    """

    def __init__(self, n):
        """Normal Order Statistics from `n` populations

        Normal order statistics of population size `n` are the ordered 
        random variables
            N_{1:n} < N_{2:n} < ... < N_{n:n}
        that are drawn from the standard normal distribution N(0, 1)
        independently.

        Parameters
        ----------
        n : int
            population size
        """
        self._n = n
        self._pr = np.arange(1, n + 1, dtype=float) / (n + 1)
        self._q0r = norm.ppf(self._pr)
        self._q1r = 1.0 / norm.pdf(self._q0r)
        self._q2r = self._q0r * self._q1r**2
        self._q3r = (1.0 + 2.0 * self._q0r**2) * self._q1r**3
        self._q4r = self._q0r * (7.0 + 6.0 * self._q0r**2) * self._q1r**4

    def exp(self):
        """Expectation of the normal order statistics, using Taylor Expansion.

        Returns
        -------
        1D ndarray : array of expectation of the normal order statistics

        Algorithm
        ---------
        Eq. (4.6.3)--(4.6.5) combined with Example 4.6 in "Order Statistics".
        """
        result = self._q0r
        result += self._pr * (1 - self._pr) * self._q2r / (2 * self._n + 4)
        result += self._pr * (1 - self._pr) * (
            1 - 2 * self._pr) * self._q3r / (3 * (self._n + 2)**2)
        result += (self._pr *
                   (1 - self._pr))**2 * self._q4r / (8 * (self._n + 2)**2)
        return result

    def var(self):
        """Variance of the normal order statistics, using Taylor Expansion.

        Returns
        -------
        1D ndarray : array of variance of the normal order statistics

        Algorithm
        ---------
        Eq. (4.6.3)--(4.6.5) combined with Example 4.6 in "Order Statistics".
        """
        result = self._pr * (1 - self._pr) * self._q1r**2 / (self._n + 2)
        result += self._pr * (1 - self._pr) * (
            1 - 2 * self._pr) * 2 * self._q1r * self._q2r / ((self._n + 2)**2)
        result += (self._pr * (1 - self._pr))**2 * (
            self._q1r * self._q3r + self._q2r**2 / 2) / (self._n + 2)**2
        return result

    def cov(self):
        """Covariance of the normal order statistics, using Taylor Expansion.

        Returns
        -------
        2D ndarray : array of covariance of the normal order statistics

        Algorithm
        ---------
        Eq. (4.6.3)--(4.6.5) combined with Example 4.6 in "Order Statistics".
        """
        result = np.outer(self._pr**2 * self._q2r, (1 - self._pr)
                          **2 * self._q2r) / 2
        result += np.outer(self._pr * (1 - 2 * self._pr) * self._q2r,
                           (1 - self._pr) * self._q1r)
        result += np.outer(self._pr * self._q1r,
                           (1 - self._pr) * (1 - 2 * self._pr) * self._q2r)
        result += np.outer(self._pr**2 * (1 - self._pr) * self._q3r,
                           (1 - self._pr) * self._q1r) / 2
        result += np.outer(self._pr * self._q1r, self._pr * (1 - self._pr)
                           **2 * self._q3r) / 2
        result /= (self._n + 2)**2
        result += np.outer(self._pr * self._q1r,
                           (1 - self._pr) * self._q1r) / (self._n + 2)
        return np.triu(result) + np.triu(result, k=1).T

    def blom(self):
        """Blom's Approximation of the Expectation of Normal Order Statistics

        Returns
        -------
        1D ndarray : array of expectation of the normal order statistics
        """
        alpha = 0.375
        pir = (np.arange(1, self._n + 1) - alpha) / (self._n + 1 - 2 * alpha)
        return norm.ppf(pir)

    def davis_stephens(self):
        """Refinement of Covariance Matrix by Algorithm 128

        Returns
        -------
        2D ndarray : array of covariance of the normal order statistics

        See
        ---
        https://statistics.stanford.edu/sites/default/files/SOL%20ONR%20254.pdf
        """
        result = self.cov()
        n = self._n
        for i in range((n + 1) // 2):
            rowsum = np.sum(result[i])
            free = np.sum(result[i, i:n - i])
            result[i, i:n - i] *= 1 + (1 - rowsum) / free
            result[i:n - i, i] = result[i, i:n - i]
            result[n - i - 1, i:n - i] = (result[i, i:n - i])[::-1]
            result[i:n - i, n - i - 1] = (result[i, i:n - i])[::-1]
        return result


def sigma_normalization_factor(dim, weights, cm=1.):
    """sigma = snf * ||xmean|| """
    nos = NormalOrderStatistics(len(weights))
    nlam = nos.blom()
    beta = -np.dot(nlam, weights)
    if len(weights) < 50:
        nnlam = nos.davis_stephens()
        gamma = beta**2 + np.dot(np.dot(nnlam, weights), weights)
    else:
        gamma = beta**2
    muw = np.sum(np.abs(weights)) ** 2 / np.dot(weights, weights)
    return beta * muw / (dim - 1 + gamma * muw) / cm


def quadratic_optimal_sigma(hess, xmean, weights, cm=1.):
    """Optimal Sigma for Quadratic
    If hess is proportional to [1, ..., 1] or the identity matrix,
    the result should be the same as 
    sigma_normalization_factor * ||xmean||.
    """    
    nos = NormalOrderStatistics(len(weights))
    nlam = nos.blom()
    beta = -np.dot(nlam, weights)
    if len(weights) < 50:
        nnlam = nos.davis_stephens()
        gamma = beta**2 + np.dot(np.dot(nnlam, weights), weights)
    else:
        gamma = beta**2
    muw = np.sum(np.abs(weights)) ** 2 / np.dot(weights, weights)
    if np.ndim(hess) == 1:
        e = hess * xmean
        eae = np.dot(hess, e * e) / np.dot(e, e) / np.sum(hess)
        g = np.linalg.norm(e) / np.sum(hess)
    else:
        e = np.dot(hess, xmean)
        eae = np.dot(e, np.dot(hess, e)) / np.dot(e, e) / np.trace(hess)
        g = np.linalg.norm(e) / np.trace(hess)
    return (beta * muw / (1 - eae + eae * gamma * muw) / cm) * g


def quadratic_optimal_normalized_sigma(hess, weights, cm=1.):
    """Estimate the optimal sigma and normalized quality gain
    The mean vector is supposed to be on the long axis of the ellipsoid.
    """
    if np.ndim(hess) == 1:
        idx = np.argmin(hess)
        xmean = np.zeros(hess.shape[0])
        xmean[idx] = 1.0
        g = hess[idx] / np.sum(hess)
    else:
        d, b = np.linalg.eigh(hess)
        idx = np.argmin(d)
        xmean = b[:, idx]
        g = d[idx] / np.sum(d)
    nos = NormalOrderStatistics(len(weights))
    nlam = nos.blom()
    beta = -np.dot(nlam, weights)
    optns = quadratic_optimal_sigma(hess, xmean, weights, cm=cm) * (cm / g)
    optnqg = beta * optns / 2.0
    return optns, optnqg, g
        
        
if __name__ == '__main__':
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from ddcma import DdCma, Checker, Logger
    except:
        raise ImportError("https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518.js")

    N = 20
    CASE = 2

    class Solution:
        def __init__(self, x):
            self._x = x
            self._x_repaired = None
            self._f = None
            self._quak_penalty = None
            self._quak_violation = []
            self._qrsk_violation = []
            
    def f(solution):
        return np.dot(solution._x_repaired, solution._x_repaired)

    def qrsk(solution):
        return - solution._x_repaired[1] + 2.0

    lbound = np.ones(N) * (-5.0)
    ubound = np.ones(N) * 5.0
    A = np.array([[-1.0] + [0.0] * (N-1)])
    b = np.array([-1.0])

    # --------------------------------------------------------------------- #
    if CASE == 1:
        print("Case 1: QRSK(1 ineq) + QUAK(1 linear ineq + bound)")
        ddcma = DdCma(xmean0=(lbound + ubound)/2.,
                    sigma0=(ubound - lbound)/4., 
                    flg_variance_update=True, 
                    flg_covariance_update=True,
                    flg_active_update=True)

        ch = McrArchConstraintHandler(dim=ddcma.N, 
                                    weight=ddcma.w, 
                                    fobjective=f, 
                                    bound=(lbound, ubound),
                                    linear_ineq_quak=(A, b),
                                    ineq_qrsk_list=[qrsk],
                                    )
        checker = Checker(ddcma)
        logger = Logger(ddcma, variable_list=['xmean', 'D', 'S', 'sigma', 'beta'])

        issatisfied = False
        fbestsofar = np.inf
        while not issatisfied:
            xx, yy, zz = ddcma.sample()
            sol_list = [Solution(x) for x in xx]
            xcov = ddcma.transform(ddcma.transform(np.eye(N)).T)
            ch.prepare(ddcma.xmean, xcov)
            ranking = ch.do(sol_list)
            idx = np.argsort(ranking)
            ddcma.update(idx, xx, yy, zz)
            
            ddcma.t += 1        
            ddcma.neval += ddcma.lam        
            ddcma.arf = np.array([sol._f for sol in sol_list])
            ddcma.arx = np.array([sol._x for sol in sol_list])
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

    # --------------------------------------------------------------------- #
    elif CASE == 2:
        print("Case 2: QRSK(1 ineq) + QUAK(1 nonlinear ineq + bound)")
        ddcma = DdCma(xmean0=(lbound + ubound)/2.,
                    sigma0=(ubound - lbound)/4., 
                    flg_variance_update=True, 
                    flg_covariance_update=True,
                    flg_active_update=True)

        ch = McrArchConstraintHandler(dim=ddcma.N, 
                                    weight=ddcma.w, 
                                    fobjective=f, 
                                    bound=(lbound, ubound),
                                    nonlinear_ineq_quak_list=McrArchConstraintHandler.lin2nonlin(A, b),
                                    ineq_qrsk_list=[qrsk],
                                    )
        checker = Checker(ddcma)
        logger = Logger(ddcma, variable_list=['xmean', 'D', 'S', 'sigma', 'beta'])

        issatisfied = False
        fbestsofar = np.inf
        while not issatisfied:
            xx, yy, zz = ddcma.sample()
            sol_list = [Solution(x) for x in xx]
            xcov = ddcma.transform(ddcma.transform(np.eye(N)).T)
            ch.prepare(ddcma.xmean, xcov)
            ranking = ch.do(sol_list)
            idx = np.argsort(ranking)
            ddcma.update(idx, xx, yy, zz)
            
            ddcma.t += 1        
            ddcma.neval += ddcma.lam        
            ddcma.arf = np.array([sol._f for sol in sol_list])
            ddcma.arx = np.array([sol._x for sol in sol_list])
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

    else:
        raise ValueError("CASE <= 2")
