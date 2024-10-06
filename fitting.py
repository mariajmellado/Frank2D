import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg._isolve.utils import make_system
import time

class IterativeSolverMethod():
    def __init__(self, Vis, Weights, kernel_row, FT, method = 'cg', rtol = 1e-7,  x0=None):
        self._FT = FT
        self._N2 = self._FT._N2

        self._Vis = Vis
        self._Weights = Weights
        self._kernel_row = kernel_row
        
        self._method = method
        self._x0 = x0
        self._rtol = rtol
    
    def get_method(self):
        method = self._method
        if method == 'cg':
            return self.cg
        elif method == 'bicg':
            return self.bicg
        elif method == 'bicgstab':
            return self.bicgstab

    def _get_atol_rtol(self, name, b_norm, atol=0., rtol=1e-5):
        """
        A helper function to handle tolerance normalization
        """
        if atol == 'legacy' or atol is None or atol < 0:
            msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
                "if set, `atol` must be a real, non-negative number.")
            raise ValueError(msg)

        print("atol: ",float(atol))
        print("rtol: ", float(rtol) * float(b_norm))

        atol = max(float(atol), float(rtol) * float(b_norm))
        print("final atol: ", atol)

        return atol, rtol

    def cg(self, A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=None, M=None, callback=None):
        A, M, x, b, postprocess = make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)

        atol, _ = self._get_atol_rtol('cg', bnrm2, atol, rtol)

        if bnrm2 == 0:
            return postprocess(b), 0

        n = len(b)

        if maxiter is None:
            maxiter = 20
        print("maxiter: ", maxiter)

        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        matvec = A.matvec
        psolve = M.matvec
        r = b - matvec(x) if x.any() else b.copy()

        # Dummy value to initialize var, silences warnings
        rho_prev, p = None, None

        for iteration in range(maxiter):
            print("iteration: ", iteration)
            if np.linalg.norm(r) < atol:  # Are we done?
                print("  --> CGM converged in ", iteration, " iterations")
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
    
    def bicg(self, A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=None, M=None, callback=None):
        A, M, x, b, postprocess = make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)

        atol, _ = self._get_atol_rtol('bicg', bnrm2, atol, rtol)

        if bnrm2 == 0:
            return postprocess(b), 0

        n = len(b)
        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        if maxiter is None:
            maxiter = n*10

        matvec, rmatvec = A.matvec, A.rmatvec
        psolve, rpsolve = M.matvec, M.rmatvec

        rhotol = np.finfo(x.dtype.char).eps**2

        # Dummy values to initialize vars, silence linter warnings
        rho_prev, p, ptilde = None, None, None

        r = b - matvec(x) if x.any() else b.copy()
        rtilde = r.copy()

        for iteration in range(maxiter):
            if np.linalg.norm(r) < atol:  # Are we done?
                return postprocess(x), 0

            z = psolve(r)
            ztilde = rpsolve(rtilde)
            # order matters in this dot product
            rho_cur = dotprod(rtilde, z)

            if np.abs(rho_cur) < rhotol:  # Breakdown case
                return postprocess, -10

            if iteration > 0:
                beta = rho_cur / rho_prev
                p *= beta
                p += z
                ptilde *= beta.conj()
                ptilde += ztilde
            else:  # First spin
                p = z.copy()
                ptilde = ztilde.copy()

            q = matvec(p)
            qtilde = rmatvec(ptilde)
            rv = dotprod(ptilde, q)

            if rv == 0:
                return postprocess(x), -11

            alpha = rho_cur / rv
            x += alpha*p
            r -= alpha*q
            rtilde -= alpha.conj()*qtilde
            rho_prev = rho_cur

            if callback:
                callback(x)

        else:  # for loop exhausted
            # Return incomplete progress
            return postprocess(x), maxiter

    def bicgstab(self, A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=None, M=None, callback=None):
        A, M, x, b, postprocess = make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)

        atol, _ = self._get_atol_rtol('bicgstab', bnrm2, atol, rtol)

        if bnrm2 == 0:
            return postprocess(b), 0

        n = len(b)

        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        if maxiter is None:
            maxiter = n*10

        matvec = A.matvec
        psolve = M.matvec

        # These values make no sense but coming from original Fortran code
        # sqrt might have been meant instead.
        rhotol = np.finfo(x.dtype.char).eps**2
        omegatol = rhotol

        # Dummy values to initialize vars, silence linter warnings
        rho_prev, omega, alpha, p, v = None, None, None, None, None

        r = b - matvec(x) if x.any() else b.copy()
        rtilde = r.copy()

        for iteration in range(maxiter):
            if np.linalg.norm(r) < atol:  # Are we done?
                return postprocess(x), 0

            rho = dotprod(rtilde, r)
            if np.abs(rho) < rhotol:  # rho breakdown
                return postprocess(x), -10

            if iteration > 0:
                if np.abs(omega) < omegatol:  # omega breakdown
                    return postprocess(x), -11

                beta = (rho / rho_prev) * (alpha / omega)
                p -= omega*v
                p *= beta
                p += r
            else:  # First spin
                s = np.empty_like(r)
                p = r.copy()

            phat = psolve(p)
            v = matvec(phat)
            rv = dotprod(rtilde, v)
            if rv == 0:
                return postprocess(x), -11
            alpha = rho / rv
            r -= alpha*v
            s[:] = r[:]

            if np.linalg.norm(s) < atol:
                x += alpha*phat
                return postprocess(x), 0

            shat = psolve(s)
            t = matvec(shat)
            omega = dotprod(t, s) / dotprod(t, t)
            x += alpha*phat
            x += omega*shat
            r -= omega*t
            rho_prev = rho

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


    def solve(self):
        print("  *  Constructing linear operators...")
        start_time = time.time()
        A = self.linear_op_A()
        b = self.linear_op_b()
        A_precond =  self.linear_op_A_precond()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'     --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        # Solve the linear system
        solver = self.get_method()
        print("  *  Solving linear system...")
        start_time = time.time()

        print("rtol: ", self._rtol)
        if self._x0 is None:
            x, info = solver(A, b, M = A_precond, rtol = self._rtol)
        else:
            x, info = solver(A, b, M = A_precond, x0 = self._x0, rtol = self._rtol)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'     --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        # Report on the success of the fitting
        fit_correctly = np.allclose(A.dot(x), b)
        print("  --> CGM converged?  ", info == 0)
        print("  --> Fit correctly?  ", fit_correctly)
        if fit_correctly:
            print("                                                !!!!!!!!!!!      fit correctly       !!!!!!!!")

        return x