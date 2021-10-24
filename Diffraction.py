"""
Classes and functions for the simulation of electron diffraction patterns based on molecular geometries 
within the independent atom model. The simulations yield results compatible with the electron 
beam parameters of the MeV Ultrafast Electron Diffraction (UED) facility at SLAC National 
Accelerator Laboratory (https://lcls.slac.stanford.edu/instruments/mev-ued). 
Created by Thomas Wolf, 02/26/2020
Modified by Thomas Wolf, 12/29/2020
Modified by Thomas Wolf, 01/01/2021
"""

import numpy as np
from scipy.interpolate import interp1d
import os
from scipy.io import loadmat

############################################################################################################
## Classes and functions ###################################################################################
############################################################################################################

class mol_geom():
    """
    Creates a molecular geometry object.
    Arguments: 
    filename: Path to a molecular geometry (*.xyz) file. See below for the file format expected
    by the code.
    """
    def __init__(self):
        pass
        
    def loadxyz(self,filename):
        """
        Function to load geometry data from an *.xyz file. The code assumes the file synthax 
        as read and written by programs like Molden (http://cheminf.cmbi.ru.nl/molden/). It 
        ignores the first two lines of the *.xyz file. The first line usually contains the 
        number of atoms in the molecular geometry, the second line contains comments. The code
        expects a line for each atom of the molecular geometry in the remainder of the file.
        Each line contains the following information in the exact order: Element letter, x, y, and z 
        coordinates. The different items are separated by spaces.
        """
        # Load geometry file as strings
        with open(filename,'r') as geofile:
            geostr = geofile.readlines()

        # Extract element information (elements) and coordinates (geom)
        geostr2 = geostr[2:]
        self.coordinates = np.zeros((len(geostr)-2,3))
        self.elements = []
        for i in np.arange(len(geostr2)):
            arr = geostr2[i].split()
            self.elements.append(arr[0])
            self.coordinates[i,0] = float(arr[1])
            self.coordinates[i,1] = float(arr[2])
            self.coordinates[i,2] = float(arr[3])
            
    def set_geometry(self, coords, elements):
        """
        Function to set arbitrary coordinates.
        Arguments:
        coords  : 2D numpy array with element in first dimension and xyz coordinates in second
                  dimension.
        elements: list of element symbols as strings.
        """
        self.coordinates = coords
        self.elements = elements
            
############################################################################################################

class Diffraction():
    """
    Creates a diffraction object.
    Arguments:
    geom:   mol_geom object
    Xsectfile: File containing atomic scattering cross-sections. The code expects a dictionary returning a 2D
               array with scattering angles in the first row and scattering cross-sections in the second row.
    Npixel: Length of Q-array
    Max_Q:  Maximum Q in inverse Angstroms
    """
    def __init__(self,geom,Npixel=120,Max_s=12,Xsectfile='Xsects.mat'):
        """
        Function to initialize Diffraction object.
        """
        self.coordinates = geom.coordinates
        self.elements = geom.elements
        dirname = os.path.dirname(__file__)
        self.XSects = loadmat(dirname + '/' + Xsectfile)
        del self.XSects['__header__'], self.XSects['__version__'], self.XSects['__globals__']
        self.U = 3.7 # Electron kinetic energy
        self.Max_s = Max_s
        self.Npixel = Npixel
        
        E=self.U*1e6*1.6022*1e-19
        m=9.1094e-31
        h=6.6261e-34
        c=299792458

        lambdaEl=h/np.sqrt(2*m*E)/np.sqrt(1+E/(2*m*c**2)) # Electron wavelength
        k=2*np.pi/lambdaEl # Electron wave vector

        thetarad = self.XSects['C'][0,:]/360*2*np.pi # Could be any other element. 
        self.a = 4*np.pi/lambdaEl*np.sin(thetarad/2)/1E10
        
        
    def make_1D_diffraction(self):
        """
        Function to create a 1D diffraction pattern assuming an ensemble of randomly oriented
        molecules.
        """
        natom = len(self.elements)
        self.s = np.linspace(0,np.float(self.Max_s),self.Npixel)

        

        self.I_at_1D = np.zeros((len(self.s),)) # Atomic scattering contribution to diffraction signal
        fmap = []
        for element in self.elements:
            f = interp1d(self.a,np.sqrt(self.XSects[element][1,:]))
            fmap.append(f(self.s))
            self.I_at_1D += np.square(abs(f(self.s)))

        # Contribution from interference between atoms to diffaction signal:
        self.I_mol_1D = np.zeros_like(self.I_at_1D) 
	# Set zero values to a small nonzero value to avoid division by zero
        for k in range(len(self.s)):
	    if (abs(self.s[k]) < 1.0e-18):
	        self.s[k]=1.0e-18
        for i in np.arange(natom):
            for j in np.arange(natom):
                if i!=j:
                    dist = np.sqrt(np.square(self.coordinates[i,:]-self.coordinates[j,:]).sum())
		    self.I_mol_1D += abs(fmap[i])*abs(fmap[j])*np.sin(dist*self.s)/(dist*self.s)
        self.sM_1D = self.s*self.I_mol_1D/self.I_at_1D # Modified molecular diffraction
        self.get_zero_crossings()
        
    def make_2D_diffraction(self):
        """
        Function to create a 2D diffraction pattern assuming an ensemble of randomly oriented
        molecules.
        """
        self.sy,self.sz = np.meshgrid(np.arange(-1*self.Max_s,self.Max_s,2*float(self.Max_s)/self.Npixel), \
                            np.arange(-1*self.Max_s,self.Max_s,2*float(self.Max_s)/self.Npixel))
        self.sr = np.sqrt(np.square(self.sy)+np.square(self.sz))
        natom = len(self.elements)

        self.I_at_2D = np.zeros_like(self.sr) # Atomic scattering contribution to diffraction signal
        fmap = []
        for element in self.elements:
            f = interp1d(self.a,np.sqrt(self.XSects[element][1,:]))
            fmap.append(f(self.sr))
            self.I_at_2D += np.square(abs(f(self.sr)))

        # Contribution from interference between atoms to diffaction signal:
        self.I_mol_2D = np.zeros_like(self.I_at_2D) 
        for i in np.arange(natom):
            for j in np.arange(natom):
                if i!=j:
                    dist = np.sqrt(np.square(self.coordinates[i,:]-self.coordinates[j,:]).sum())
                    self.I_mol_2D += abs(fmap[i])*abs(fmap[j])*np.sin(dist*self.sr)/(dist*self.sr) 

        self.sM_2D = self.sr*self.I_mol_2D/self.I_at_2D # Modified molecular diffraction
        
    def get_zero_crossings(self):
        """
        Function to get the zero crossings of the 1D modified molecular diffraction.
        """
        self.zcross = []
        for i in np.arange(len(self.sM_1D)-1):
            if self.sM_1D[i]<=0 and self.sM_1D[i+1]>0:
                ind = abs(np.array([self.sM_1D[i], self.sM_1D[i+1]])).argmin()
                self.zcross.append(self.s[i+ind])
            elif self.sM_1D[i]>=0 and self.sM_1D[i+1]<0:
                ind = abs(np.array([self.sM_1D[i], self.sM_1D[i+1]])).argmin()
                self.zcross.append(self.s[i+ind])
        self.zcross = np.array(self.zcross)
