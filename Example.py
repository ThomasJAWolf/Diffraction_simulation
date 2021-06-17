"""
Code example for the simulation of electron diffraction patterns based on molecular geometries 
within the independent atom model. The simulations yield results compatible with the electron 
beam parameters of the MeV Ultrafast Electron Diffraction (UED) facility at SLAC National 
Accelerator Laboratory (https://lcls.slac.stanford.edu/instruments/mev-ued). 
Created by Thomas Wolf, 02/26/2020
Modified by Thomas Wolf, 12/29/2020
Modified by Thomas Wolf, 01/01/2021
"""
        
############################################################################################################
## Example code for diffraction of 1,3-cyclohexadiene ######################################################
############################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from Diffraction import *

fname = 'CHD_6-31Gd.xyz'

CHD_geo = mol_geom()
CHD_geo.loadxyz(fname)
CHD_Diff = Diffraction(CHD_geo)

# Figure showing atomic scattering cross-sections

plt.figure()
for el, Xsect in CHD_Diff.XSects.items():
    plt.plot(Xsect[0,:],Xsect[1,:], label= el)

plt.title('Differential Atomic scattering cross-sections from ELSEPA')
plt.legend(loc='best')
plt.xlim(0,0.5)
plt.ylabel('Differential cross-section / $a_{0}^{2}/{sr}$')
plt.xlabel('Scattering angle / $^{\circ}$')

CHD_Diff.make_1D_diffraction()

# Figure showing 1D diffraction
plt.figure()

plt.subplot(2,2,1)
plt.title('Atomic contribution')
plt.semilogy(CHD_Diff.s,CHD_Diff.I_at_1D)
plt.xlabel('s / $\AA^{-1}$')
plt.ylabel('$\sigma$ / $a_{0}^{2}/{sr}$')

plt.subplot(2,2,2)
plt.title('Molecular contribution')
plt.plot(CHD_Diff.s,CHD_Diff.I_mol_1D)
plt.xlabel('s / $\AA^{-1}$')
plt.ylabel('$\sigma$ / $a_{0}^{2}/{sr}$')

plt.subplot(2,2,3)
plt.title('Total signal')
plt.semilogy(CHD_Diff.s,CHD_Diff.I_mol_1D+CHD_Diff.I_at_1D)
plt.xlabel('s / $\AA^{-1}$')
plt.ylabel('$\sigma$ / $a_{0}^{2}/{sr}$')

plt.subplot(2,2,4)
plt.title('Modified diffraction')
plt.plot(CHD_Diff.s,CHD_Diff.sM_1D, label='Modified diffraction')
plt.plot(CHD_Diff.zcross, np.zeros_like(CHD_Diff.zcross), '.', label='Zero crossings')
plt.xlabel('s / $\AA^{-1}$')
plt.ylabel('sM(s)')
plt.legend(loc='best')
plt.tight_layout()

CHD_Diff.make_2D_diffraction()

# Figure showing 2D diffraction. The limits of the color bar (vmin, vmax) will have to
# be adjusted for other molecules.
plt.figure()
plt.subplot(2,2,1)
plt.title('Atomic contribution')
plt.pcolormesh(CHD_Diff.sy,CHD_Diff.sz,CHD_Diff.I_at_2D, shading='auto')
plt.xlabel('s / $\AA^{-1}$')
plt.ylabel('s / $\AA^{-1}$')
plt.colorbar()

plt.subplot(2,2,2)
plt.title('Molecular contribution')
plt.pcolormesh(CHD_Diff.sy,CHD_Diff.sz,CHD_Diff.I_mol_2D,vmin=-1000,vmax = 10000, shading='auto')
plt.xlabel('s / $\AA^{-1}$')
plt.ylabel('s / $\AA^{-1}$')
plt.colorbar()

plt.subplot(2,2,3)
plt.title('Total signal')
plt.pcolormesh(CHD_Diff.sy,CHD_Diff.sz,CHD_Diff.I_mol_2D+CHD_Diff.I_at_2D,vmin=0,vmax = 10000, shading='auto')
plt.xlabel('s / $\AA^{-1}$')
plt.ylabel('s / $\AA^{-1}$')
plt.colorbar()

plt.subplot(2,2,4)
plt.title('Modified diffraction')
plt.pcolormesh(CHD_Diff.sy,CHD_Diff.sz,CHD_Diff.sM_2D, shading='auto')
plt.xlabel('s / $\AA^{-1}$')
plt.ylabel('s / $\AA^{-1}$')
plt.colorbar()
plt.show()
