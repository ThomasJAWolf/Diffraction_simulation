# Diffraction_simulation
Simple code to simulate electron diffraction signals from molecular geometry files. The code
is geared towards users of the MeV ultrafast electron diffraction facility at SLAC National
Accelerator Laboratory (https://lcls.slac.stanford.edu/instruments/mev-ued)
Details can be found in the comments of Diffraction.py and Example.py. Diffraction.py contains classes and functions. Example.py contains example code of how to use Diffraction.py to create diffraction patterns. The repository contains a file Xsects.mat containing angle-dependent scattering cross-sections for all elements of the periodic table, which were simulated (thanks to Pedro Nunes!) using the ELSEPA program 
(https://github.com/eScatter/elsepa) for different elements and a .xyz geometry file of
1,3-cyclohexadiene.
