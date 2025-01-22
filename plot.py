import numpy as np
import matplotlib.pyplot as plt
from frank2d import Frank2D
from frank2d import const
import time
from matplotlib.colors import LogNorm

class Plot():
    def __init__(self, Frank2D):
        self._frank2d = Frank2D
        
    def get_image(self, title = "Model", size = 7,  add_fourier_resolution = False, log_norm = False):
        frank2d = self._frank2d
        I = frank2d.sol_intensity
        Nx = frank2d._Nx
        Ny = frank2d._Ny
        dx = frank2d._FT._dx*const.rad_to_arcsec
        dy = frank2d._FT._dy*const.rad_to_arcsec
        Rout = frank2d._Rmax*const.rad_to_arcsec
        x = frank2d._FT._x*const.rad_to_arcsec
        y = frank2d._FT._y*const.rad_to_arcsec

        # Coordenadas del pixel que quieres mostrar
        pixel_x, pixel_y = Nx//2, Ny//2
        pixel_value = I[pixel_y, pixel_x]

        # Crear una figura con dos subplots
        if add_fourier_resolution:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})
        else:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
            axs = [ax]
        
        norm = LogNorm() if log_norm else None

        # Primer subplot.
        plot = axs[0].pcolormesh(x, y, I, cmap='magma', norm=norm)
        axs[0].invert_xaxis()
        cmap = plt.colorbar(plot, ax=axs[0])
        cmap.set_label(r'I [Jy $sr^{-1}$]', size=15)

        axs[0].set_title(title)
        axs[0].set_xlabel("dRa ['']")
        axs[0].set_ylabel("dDec ['']")

        xlim = axs[0].get_xlim()
        ylim = axs[0].get_ylim()

        axs[0].text(
            xlim[0] + 0.1 * (xlim[1] - xlim[0]),  # 10% desde el borde izquierdo
            ylim[0] + 0.1 * (ylim[1] - ylim[0]),  # 10% desde el borde inferior
            r' FOV: ' + str(2 * Rout) + ' arcsecs ' + r'| $N^{2}$ pixels,  N = ' + str(Nx) + '  ',
            bbox={'facecolor': 'white', 'pad': 4, 'alpha': 0.8}
        )

        if add_fourier_resolution:
            # Segundo subplot: el pixel específico
            # Creamos una matriz de ceros y luego establecemos el valor del píxel deseado en I[pixel_y, pixel_x]
            pixel_image = np.zeros((1, 1))
            pixel_image[0, 0] = pixel_value
            img = axs[1].imshow(pixel_image, cmap="magma", extent=[0, 1, 0, 1])
            axs[1].set_title(f'Pixel at ({pixel_x}, {pixel_y})')
            axs[1].set_xticks([])
            axs[1].set_yticks([])

            # Colorbar para el pixel específico
            cmap_pixel = plt.colorbar(img, ax=axs[1], shrink=0.5)
            cmap_pixel.set_label(r'I [Jy $sr^{-1}$]', size=9)

            # Configurar notación científica en la barra de color
            cmap_pixel.formatter.set_powerlimits((-2, 2))  # Limitar la notación científica a potencias entre -3 y 3
            cmap_pixel.update_ticks()

            # Mostrar el valor del pixel como leyenda
            axs[1].text(0.5, -0.3, f'Intensity: {pixel_value:.4}', ha='center', va='center', transform=axs[1].transAxes, fontsize=9, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3})

            # Agregar indicaciones del largo del pixel en los ejes x y y
            axs[1].annotate('', xy=(0, 0), xytext=(1, 0), arrowprops=dict(arrowstyle='<->', color='black'))
            axs[1].text(0.5, -0.05, f'{dx: .5f} ["]', ha='center', va='top', transform=axs[1].transAxes)

            axs[1].annotate('', xy=(0, 0), xytext=(0, 1), arrowprops=dict(arrowstyle='<->', color='black'))
            axs[1].text(-0.05, 0.5, f'{dy: .5f}  ["]', ha='right', va='center', transform=axs[1].transAxes, rotation='vertical')

            # Ajustar el tamaño de los subplots
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2)
        
        plt.gca().set_aspect('equal') 

        plt.show()


    def get_several_images(self, data, params, n_plots = 9, rtol = 1e-7, transform = "2fft"):
        u, v, Vis, Weights = data
        m, c, l = params

        I_array = []
        Vis_array = []
        frank2d = self._frank2d

        for i in range(0, n_plots):
            start_time = time.time()
            #----------------------
            print("fit : "+ str(i))
            frank2d.fit(u, v, Vis, Weights, kernel_params = [m[i], c[i], l[i]],
                         rtol = rtol, transform = transform)

            Vis_array.append(frank2d.sol_visibility)
            I_array.append(frank2d.sol_intensity)
            #----------------------
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'total time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')
            print(' with ' + f' m = {m[i]:.1f}, c = {c[i]:.1f}, l = {l[i]:.1e} ') 
            print(".....................")


        n_plots = int(np.sqrt(n_plots))

        fig, axs = plt.subplots(n_plots, n_plots, figsize=(18, 14))

        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        x, y = frank2d._FT._x*const.rad_to_arcsec, frank2d._FT._y*const.rad_to_arcsec

        k = 0
        for i in range(n_plots):
            for j in range(n_plots):
                I = I_array[k].reshape(frank2d._Nx, frank2d._Ny).T
                plot = axs[i, j].pcolormesh(y, x, I, cmap='magma')
                
                axs[i, j].invert_xaxis()
                
                cmap = plt.colorbar(plot, ax=axs[i, j])

                if i == j and i == (0):
                    axs[i, j].set_title(f'AS209 at 1mm')
                    axs[i, j].set_xlabel("dRa ['']")
                    axs[i, j].set_ylabel("dDec ['']")
                    cmap.set_label(r'I [Jy $sr^{-1}$]', size=10)

                    Rout =  frank2d._Rmax*const.rad_to_arcsec

                    axs[i, j].text(1.7, -1.5, f' N = {frank2d._Nx}, FOV: {2*Rout:.2f} ', 
                                bbox={'facecolor': 'white', 'pad': 2, 'alpha': 0.8})

                axs[i, j].text(1.7, -1.8, f' m = {m[k]:.1f}, c = {c[k]:.1f}, l = {l[k]:.1e}', 
                            bbox={'facecolor': 'white', 'pad': 2, 'alpha': 0.8})

                axs[i, j].text(1.7, 1.6, f'{k}', bbox={'facecolor': 'white', 'pad': 2, 'alpha': 1})
                
                k += 1

        plt.show()
        
        return Vis_array

        



    