import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg._isolve.utils import make_system
import time

class ConjugateGradientMethod():
    def __init__(self, Vis, Weights, kernel_row, FT, x0=None):
        self._FT = FT
        self._N2 = self._FT._N2

        self._Vis = Vis
        self._Weights = Weights
        self._kernel_row = kernel_row

        self._x0 = x0

    def _get_atol_rtol(self, name, b_norm, atol=0., rtol=1e-5):
        """
        A helper function to handle tolerance normalization
        """
        if atol == 'legacy' or atol is None or atol < 0:
            msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
                "if set, `atol` must be a real, non-negative number.")
            raise ValueError(msg)

        atol = max(float(atol), float(rtol) * float(b_norm))

        return atol, rtol

    def cg(self, A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=None, M=None, callback=None):
        """Use Conjugate Gradient iteration to solve ``Ax = b``.

        Parameters
        ----------
        A : {sparse matrix, ndarray, LinearOperator}
            The real or complex N-by-N matrix of the linear system.
            ``A`` must represent a hermitian, positive definite matrix.
            Alternatively, ``A`` can be a linear operator which can
            produce ``Ax`` using, e.g.,
            ``scipy.sparse.linalg.LinearOperator``.
        b : ndarray
            Right hand side of the linear system. Has shape (N,) or (N,1).
        x0 : ndarray
            Starting guess for the solution.
        rtol, atol : float, optional
            Parameters for the convergence test. For convergence,
            ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
            The default is ``atol=0.`` and ``rtol=1e-5``.
        maxiter : integer
            Maximum number of iterations.  Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved.
        M : {sparse matrix, ndarray, LinearOperator}
            Preconditioner for A.  The preconditioner should approximate the
            inverse of A.  Effective preconditioning dramatically improves the
            rate of convergence, which implies that fewer iterations are needed
            to reach a given error tolerance.
        callback : function
            User-supplied function to call after each iteration.  It is called
            as callback(xk), where xk is the current solution vector.

        Returns
        -------
        x : ndarray
            The converged solution.
        info : integer
            Provides convergence information:
                0  : successful exit
                >0 : convergence to tolerance not achieved, number of iterations

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.sparse import csc_matrix
        >>> from scipy.sparse.linalg import cg
        >>> P = np.array([[4, 0, 1, 0],
        ...               [0, 5, 0, 0],
        ...               [1, 0, 3, 2],
        ...               [0, 0, 2, 4]])
        >>> A = csc_matrix(P)
        >>> b = np.array([-1, -0.5, -1, 2])
        >>> x, exit_code = cg(A, b, atol=1e-5)
        >>> print(exit_code)    # 0 indicates successful convergence
        0
        >>> np.allclose(A.dot(x), b)
        True

        """
        A, M, x, b, postprocess = make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)

        atol, _ = self._get_atol_rtol('cg', bnrm2, atol, rtol)

        if bnrm2 == 0:
            return postprocess(b), 0

        n = len(b)

        if maxiter is None:
            maxiter = n*10

        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        matvec = A.matvec
        psolve = M.matvec
        r = b - matvec(x) if x.any() else b.copy()

        # Dummy value to initialize var, silences warnings
        rho_prev, p = None, None

        for iteration in range(maxiter):
            if np.linalg.norm(r) < atol:  # Are we done?
                return postprocess(x), 0

            z = psolve(r)
            rho_cur = dotprod(r, z)
            if iteration > 0:
                beta = rho_cur / rho_prev
                p *= beta
                p += z
            else:  # First spin
                p = np.empty_like(r)
                p[:] = z[:]

            q = matvec(p)
            alpha = rho_cur / dotprod(p, q)
            x += alpha*p
            r -= alpha*q
            rho_prev = rho_cur

            if callback:
                callback(x)

        else:  # for loop exhausted
            # Return incomplete progress
            return postprocess(x), maxiter
    

    def linear_op_A(self):
        def SNI(vis_model):
            return np.array([ 
                    np.dot(np.add(
                        self._kernel_row(i)*self._Weights, np.eye(1, self._N2, i).flatten()), vis_model)
                    for i in range(self._N2)
                    ])
        
        return LinearOperator((self._N2, self._N2 ), matvec= SNI)
    
    def linear_op_b(self):
        def SNV(vis_gridded):
            return np.array([
                    np.dot(
                        self._kernel_row(i)*self._Weights, vis_gridded)
                    for i in range(self._N2)
                    ])
        bop = LinearOperator((self._N2, self._N2), matvec= SNV)
        return bop.matvec(self._Vis)

    def linear_op_A_precond(self):
        def diag_SNI(vis_model):
            return np.array([ 
                    ((self._kernel_row(i)[i] * self._Weights[i] + 1)**(-1) * vis_model[i])
                    for i in range(self._N2)
                    ])

        return LinearOperator((self._N2, self._N2), matvec=diag_SNI)


    def solve(self, rtol):
        print("  *  Constructing linear operators...")
        start_time = time.time()
        A = self.linear_op_A()
        b = self.linear_op_b()
        A_precond =  self.linear_op_A_precond()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'     --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        print("  *  Solving linear system...")
        start_time = time.time()
        if self._x0 is None:
            x, info = self.cg(A, b, M = A_precond, rtol = rtol)
        else:
            x, info = self.cg(A, b, x0 = self._x0)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'     --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        fit_correctly = np.allclose(A.dot(x), b)

        print("  --> CGM converged?  ", info == 0)
        print("  --> Fit correctly?  ", fit_correctly)
        if fit_correctly:
            print("                                                 fit correctly       !!!!!!")

        return x