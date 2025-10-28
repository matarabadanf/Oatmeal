from pyscf import gto, scf
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

mol_Be = gto.M(atom = 'Sc 0 0 0', spin=1, charge=0, basis='6-31g')

kin = mol_Be.intor('int1e_kin')
vnuc = mol_Be.intor('int1e_nuc')
overlap = mol_Be.intor('int1e_ovlp')
eri = mol_Be.intor('int2e')

# print(overlap)

print(overlap[5][5:14])

plt.imshow(overlap, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Value')
plt.title('Heatmap of Sc def2-SVP [5s,3p,2d,1f]')
plt.show()