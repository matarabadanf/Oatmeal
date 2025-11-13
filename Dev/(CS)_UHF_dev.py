from Dev.CSUHF import CS_UHF
from py_mods.src.SCF.RHF import plot_map
from pyscf import gto, scf, ao2mo
import numpy as np 

spin = 2

mol_He= gto.M(atom = 'He 0 0 0;', spin=spin, charge=0, basis='aug-cc-pvqz') #  basis='6-311g') # basis='aug-cc-pVqZ')

# mol_He.basis = {'He': gto.basis.parse(He_tempered_str)}
mol_He.build()

kin = mol_He.intor('int1e_kin')
vnuc = mol_He.intor('int1e_nuc')
overlap = mol_He.intor('int1e_ovlp')
eri = mol_He.intor('int2e')

rhf_He = scf.UHF(mol_He)
e_He = rhf_He.kernel()

nelec = 2
theta = 0

occupations = [np.array([1,1,0,0,0,0,0,0,0]), np.array([0,0,0])]

converged, E_elec_comp, e_alpha, e_beta, P_LR_alph, P_LR_beta, P_LR_alph_0 , pp = CS_UHF(overlap, kin, vnuc, eri, nelec, occupation=occupations, mult=spin, verbose=True, threshold=1E-10, max_iter=500,  conv_type='DIIS')

P_total = P_LR_alph + P_LR_beta
P_minus = P_LR_alph - P_LR_beta


plot_map(pp.reshape(kin.shape).real)
plot_map(P_LR_alph_0.reshape(kin.shape).real)
plot_map(overlap.reshape(kin.shape).real)
plot_map(P_minus.reshape(kin.shape).real)


