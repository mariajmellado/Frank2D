import numpy as np
from fourier2d import DiscreteFourierTransform2D as ft2d

class CovarianceMatrix():
    def __init__(self, N, Rmax, params):
        FT = ft2d(Rmax, N)
        self._u_ft_N2 =  FT._Un
        self._v_ft_N2 =  FT._Vn

        self._params = params
        self._min_freq = self._v_ft_N2[1]
                 

    def SE_kernel(self, i):
        m, c, l = self._params
        u, v = self._u_ft_N2, self._v_ft_N2
        min_freq = np.min(v)
        N = np.sqrt(len(u))
        block = int(i // N)

        u_i = u[i]
        v_i = v[i]
        
        q_1i = np.hypot(u, v)
        q_2i = np.hypot(u_i, v_i)

        def power_spectrum(q, m, c):
            if not np.isscalar(q):  
                q[q == 0] = self._min_freq
            elif q == 0:
                q = self._min_freq
            return np.exp(m*np.log(q) + c)

        p_1i = power_spectrum(q_1i, m, c)
        p_2i = power_spectrum(q_2i, m, c)
        
        factor = np.sqrt(p_1i * p_2i)

        SE_Kernel_row =  np.exp(-0.5 * ((u - u_i) ** 2 + (v - v_i) ** 2) / l ** 2) * factor

        return SE_Kernel_row
    
    def RadialBasis_kernel(self, i):
        pass 