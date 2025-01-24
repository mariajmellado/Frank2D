import numpy as np

class CovarianceMatrix():
    def __init__(self, N, Rmax, params, FT):
        self._q = FT.q
        self._u = FT._Un
        self._v = FT._Vn
        self._min_freq = self._v[1]
    
    def power_spectrum(self, q, m, c):
        if not np.isscalar(q):  
            q[q == 0] = self._min_freq
        elif q == 0:
            q = self._min_freq
        return (q**m)*np.exp(c)


class SquareExponential(CovarianceMatrix):
    def __init__(self, N, Rmax, params, FT):
        super().__init__(N, Rmax, params, FT)

        # SE Kernel params
        self._m, self._c, self._l, self._k = params

        self._ul = self._u/self._l
        self._vl = self._v/self._l

        self._power_spectrum_q1 = self.power_spectrum(self._q, self._m, self._c)
                    

    def kernel(self, i):
        amp = np.sqrt(self._power_spectrum_q1 * self.power_spectrum(self._q[i], self._m, self._c))

        return amp * np.exp(-0.5 * ((self._ul - self._ul[i]) ** 2 + (self._vl - self._vl[i]) ** 2))
    
    def kernel_polar(self, i):
        amp = np.sqrt(self._power_spectrum_q1 * self.power_spectrum(self._q[i], self._m, self._c))
        q_i = (self._ul - self._ul[i]) ** 2 + (self._vl - self._vl[i]) ** 2
        theta = np.arctan2(self._v, self._u)/self._k
        theta_i = (theta - theta[i])**2

        return  amp * np.exp(-0.5 * (q_i + theta_i))


class Wendland(CovarianceMatrix):
    def __init__(self, N, Rmax, params, FT):
        super().__init__(N, Rmax, params, FT)
        
        self._m, self._c, self._l = params

        self._j, self._k = 4, 1
        self._H = 2*1.897367*self._l

        self._uh = self._u/self._H
        self._vh = self._v/self._H
        self._power_spectrum_q1 = self.power_spectrum(self._q, self._m, self._c)

    def P_k(self, r, k):
        if k == 0:
            return np.ones_like(r)  # P_0(r) = 1
        elif k == 1:
            return 4*r +1  # P_1(r) = 4r + 1
        elif k == 2:
            return (35/3)*r**2 +6*r + 1  # P_2(r) = (35/3)r^2 + 6r + 1
        else:
            raise ValueError("k must be 0, 1, or 2.")

    def kernel(self, i):
        amp = np.sqrt(self._power_spectrum_q1 * self.power_spectrum(self._q[i], self._m, self._c))

        r_normalized = np.sqrt((self._uh-self._uh[i])**2 + (self._vh-self._vh[i])**2)
        factor = (1 - r_normalized)**self._j
        factor[r_normalized > 1] = 0

        return amp * factor * self.P_k(r_normalized, self._k)
    





