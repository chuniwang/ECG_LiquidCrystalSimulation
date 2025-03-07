# **Accessing the electronic structure of liquid crystalline semiconductors with bottom-up electronic coarse-graining**
This repository provides the CG MD models, ECG models, and kMC estimators of a liquid crystal material (BTBT). Please refer to the published paper ([Chun-I Wango, J. Charlie Maier, and Nicholas E. Jackson, *Chem. Sci.*, 2024, 15, 8390-8403](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d3sc06749a)) for more details. Below are brief introductions for each folder.

## **CG_MD**
This folder includes the files to perform the Bottom-up coarse-grained (CG) molecular dynamics simulations of BTBT in isotropic (***NVT_700K***), smectic A (***NVT_555K***), and smectic E (***NVT_515K***) phase by using LAMMPS.

## **ECG**
This folder contains the electronic coarse-graining (ECG) models for predicting HOMO energy (***HOMO_Energy***), HOMO-HOMO couplings (***HOMO_Coupling***), and the sign of the coupling value(***Sign_Classification***). The Python files define the model architecture, while the *.pth files store the parameters of the trained models.

## **kMC_Mobility_Estimators**
This folder contains all the scripts for constructing the electronic Hamiltonian, estimating charge mobility using Kinetic Monte Carlo, and analyzing the corresponding structural and electronic properties.
