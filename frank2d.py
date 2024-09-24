import constants as const
from fourier2d import DiscreteFourierTransform2D as ft2d
from preprocess_vis import Gridding
from gaussian_process import CovarianceMatrix
from fit import ConjugateGradientMethod

from frank.radial_fitters import FrankFitter
from frank.geometry import SourceGeometry
from frank.plot import sweep_profile

import numpy as np
import matplotlib.pyplot as plt
import time


class Frank2D(object):
    def __init__(self, N, Rmax, geometry = [None, None, None, None]):
        self._Nx = self._Ny =  N
        self._N2 = self._Nx*self._Ny
        self._Rmax = Rmax/const.rad_to_arcsec #rad
        self._FT = ft2d(self._Rmax, self._Nx)
        self._geometry = geometry

        self._kernel = None
        self._x0 = None
        self._method = None

        self.sol_visibility = None
        self.sol_intensity = None

    def preprocess_vis(self, u, v, Vis, Weights):
        start_time = time.time()
        grid = Gridding(self._Nx, self._Rmax)
        u_gridded, v_gridded, vis_gridded, weights_gridded = grid.run(u, v, Vis, Weights)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'  --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        return u_gridded, v_gridded, vis_gridded, weights_gridded

    def set_kernel(self, kernel_params = [-5, 60, 1e4]):
        print("Setting kernel...")
        kernel = CovarianceMatrix(self._Nx, self._Rmax, params = kernel_params)
        self._kernel =  kernel.SE_kernel

    def set_guess(self, u, v, Vis, Weights):
        self._x0 = self.guess_frank1d(u, v, Vis, Weights)


    def set_fit_method(self, u, v, Vis, Weights, kernel_params = None):
        _, _, vis_gridded, weights_gridded = self.preprocess_vis(u, v, Vis, Weights)

        if kernel_params is not None:
            self.set_kernel(kernel_params = kernel_params)
        else:
            self.set_kernel()

        print("Setting fit method...")
        self._method = ConjugateGradientMethod(vis_gridded, weights_gridded, self._kernel, self._FT, x0 = self._x0)


    def fit(self, u, v, Vis, Weights, kernel_params = None, rtol = 1e-7, transform = "2fft"):

        self.set_fit_method(u, v, Vis, Weights, kernel_params = kernel_params)
        
        print("Fitting...")
        vis_model = self._method.solve(rtol = rtol)
        self.sol_visibility = vis_model 

        # Image model.
        print("Inverting...")
        if transform == "2fft":
            start_time = time.time()
            I_model =  self._FT.transform_fast(vis_model, direction = 'backward')
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'  --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')
        elif transform == "2dft":
            start_time = time.time()
            I_model =  self._FT.transform(vis_model, direction = 'backward')
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'  --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        self.sol_intensity = I_model.real.reshape(self._Nx, self._Ny)
    
    # TO DO
    def guess_frank1d(self, u, v, Vis, Weights, alpha = 1.3, w_smooth = 1e-1, n_pts = 300):
        FT = self._FT
        inc, pa, dra, ddec = self._geometry
        geom = SourceGeometry(inc = inc, PA = pa, dRA = dra, dDec = ddec)
        Rout = self._Rmax*const.rad_to_arcsec
        FF = FrankFitter(Rout, n_pts, geom, alpha = alpha, weights_smooth = w_smooth)
        sol = FF.fit(u, v, Vis, Weights)

        I2D, _, _ = sweep_profile(sol.r, sol.mean,
                                   xmax = FT._Xmax*const.rad_to_arcsec,
                                   ymax = FT._Ymax*const.rad_to_arcsec,
                                   dr = (FT._x[1] - FT._x[0])*const.rad_to_arcsec)
        
        return I2D.flatten()
