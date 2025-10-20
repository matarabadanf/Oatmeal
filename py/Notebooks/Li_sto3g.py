import pyscf.gto as gto

mol_Li = gto.M(atom = 'Li 0 0 0', spin=1)

kin = mol_Li.intor('int1e_kin')
vnuc = mol_Li.intor('int1e_nuc')
overlap = mol_Li.intor('int1e_ovlp')
eri = mol_Li.intor('int2e')

print(kin)
