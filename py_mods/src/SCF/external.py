from py_mods.src.SCF.CSRHF import CS_RHF_ContextClass
from py_mods.src.SCF.CSUHF import CS_UHF_ContextClass
from pyscf import gto


def RHF_context_from_pyscf(atom="He 0 0 0", spin=0, charge=0, basis="cc-pvdz"):
    mol = gto.M(atom=atom, spin=spin, charge=charge, basis=basis)
    n_elec = sum(mol.nelec)

    kin = mol.intor("int1e_kin")
    vnuc = mol.intor("int1e_nuc")
    overlap = mol.intor("int1e_ovlp")
    eri = mol.intor("int2e")

    return_class = CS_RHF_ContextClass(overlap, kin, vnuc, eri, n_electrons=n_elec)

    return return_class


def UHF_context_from_pyscf(atom="He 0 0 0", spin=0, charge=0, basis="cc-pvdz"):
    mol = gto.M(atom=atom, spin=spin, charge=charge, basis=basis)
    n_elec = sum(mol.nelec)

    kin = mol.intor("int1e_kin")
    vnuc = mol.intor("int1e_nuc")
    overlap = mol.intor("int1e_ovlp")
    eri = mol.intor("int2e")

    return_class = CS_UHF_ContextClass(overlap, kin, vnuc, eri, n_electrons=n_elec)

    return return_class
