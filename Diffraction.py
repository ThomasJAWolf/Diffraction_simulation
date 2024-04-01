"""
Classes and functions for the simulation of electron diffraction patterns based on molecular geometries 
within the independent atom model. The simulations yield results compatible with the electron 
beam parameters of the MeV Ultrafast Electron Diffraction (UED) facility at SLAC National 
Accelerator Laboratory (https://lcls.slac.stanford.edu/instruments/mev-ued). 
Created by Thomas Wolf, 02/26/2020
Modified by Thomas Wolf, 12/29/2020
Modified by Thomas Wolf, 01/01/2021
Modified by Xinxin Cheng, 05/26/2022: Include simulation capabilities for X-ray scattering
Modified by Thoma sWolf, 04/01/2024
"""

import numpy as np
from scipy.interpolate import interp1d
import os
from scipy.io import loadmat
import pickle

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
        self.coordinates = np.array(coords)
        self.elements = elements
            
############################################################################################################

class Diffraction():
    """
    Creates a diffraction object.
    Arguments:
    geom           : mol_geom object
    Npixel         :  Length of Q-array
    Max_Q          : Maximum Q in inverse Angstroms
    diffractionType: Type of desired diffraction, can be 'electron' or'xray'.
                     Electron scattering cross-sections are calculated using 
                     the elsepa program (xhttps://github.com/eScatter/elsepa) 
                     assuming a kinetic energy of 3.7 MeV.
    """
    def __init__(self,geom,Npixel=120,Max_s=12,diffractionType='electron'):
        """
        Function to initialize Diffraction object.
        """
        self.coordinates = geom.coordinates
        self.elements = geom.elements
        self.diffractionType = diffractionType
        self.Max_s = Max_s
        self.Npixel = Npixel
        dirname = os.path.dirname(__file__)
        
        if (self.diffractionType == 'electron'):
            Xsectfile='Xsects.mat'
            self.XSects = loadmat(dirname + '/' + Xsectfile)
            del self.XSects['__header__'], self.XSects['__version__'], self.XSects['__globals__']
            self.U = 3.7 # Electron kinetic energy, the saved cross-section was calculated with 3.7 MeV, although we are currently using 4.2 MeV for real experiment

            E=self.U*1e6*1.6022*1e-19
            m=9.1094e-31
            h=6.6261e-34
            c=299792458

            lambdaEl=h/np.sqrt(2*m*E)/np.sqrt(1+E/(2*m*c**2)) # Electron wavelength
            k=2*np.pi/lambdaEl # Electron wave vector

            thetarad = self.XSects['C'][0,:]/360*2*np.pi # Could be any other element. Convert degree to rad.
            self.a = 4*np.pi/lambdaEl*np.sin(thetarad/2)/1E10
            
        elif (self.diffractionType == 'xray'):
            Xsectfile='Xsects.pkl'
            with open(dirname + '/' + Xsectfile, "rb") as f:
                self.XSects = pickle.load(f)
            self.a = self.XSects['C'][0,:]
        
    def make_1D_diffraction(self, geom=None):
        """
        Function to create a 1D diffraction pattern assuming an ensemble of randomly oriented
        molecules.
        Arguments:
        geom: Optional for generating series of diffraction patterns from e.g. a AIMD trajectory. It assumes
              identical number of atoms and atomic ordering for all geometries, but saves time by not completely
              reevaluating e.g. the atomic scattering.
        """
        if geom:
            self.coordinates= geom.coordinates
        if not hasattr(self, 'natom'):
            natom = len(self.elements)
            self.s = np.linspace(0,float(self.Max_s),self.Npixel)
            self.s[(self.s<1.0e-18)] = 1.0e-18
            self.I_at_1D = np.zeros((len(self.s),)) # Atomic scattering contribution to diffraction signal
            fmap = []
            for element in self.elements:
                f = interp1d(self.a,np.sqrt(self.XSects[element][1,:]))
                fmap.append(f(self.s))
                self.I_at_1D += np.square(abs(f(self.s)))
            self.fmap = np.array(fmap)
        

        

        # Contribution from interference between atoms to diffaction signal:
        self.I_mol_1D = np.zeros_like(self.I_at_1D)
        # Set zero values to a small nonzero value to avoid division by zero
        atoms = np.arange(natom)
        pairs = np.array([[a, b] for idx, a in enumerate(atoms) for b in atoms[idx + 1:]])
        self.I_mol_1D = np.zeros_like(self.s)
        if len(pairs)!=0:
            dists = np.sqrt(np.square(self.coordinates[pairs[:, 0], :]-self.coordinates[pairs[:, 1], :]).sum(1))
            dist_s = dists*np.tile(self.s, (len(dists), 1)).T
            self.I_mol_1D = (abs(self.fmap[pairs[:, 0]])*abs(self.fmap[pairs[:, 1]])*(np.sin(dist_s)/dist_s).T).sum(0)*2
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
