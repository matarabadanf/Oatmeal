from typing import Union
from py_mods.src.SCF.types import CSRHFContext
from py_mods.src.SCF.CSUHF import CS_UHF_ContextClass
from pyscf import gto
from pyscf.lib.exceptions import BasisNotFoundError


def RHF_context_from_pyscf(
    atom="He 0 0 0",
    spin=0,
    charge=0,
    basis="cc-pvdz",
    verbose=0,
    parseable_basis: Union[None, str] = None,
):
    mol = gto.M(atom=atom, spin=spin, charge=charge, basis=basis)
    n_elec = sum(mol.nelec)

    if parseable_basis is not None:
        try:
            mol.basis = gto.basis.parse(parseable_basis)
            mol.build()
        except BasisNotFoundError:
            print("Invalid custom basis string. Reverting to default cc-pvdz")
            mol = gto.M(atom=atom, spin=spin, charge=charge, basis=basis)
            mol.build()

    kin = mol.intor("int1e_kin")
    vnuc = mol.intor("int1e_nuc")
    overlap = mol.intor("int1e_ovlp")
    eri = mol.intor("int2e")

    return_class = CSRHFContext(overlap, kin, vnuc, eri, n_electrons=n_elec)

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


if __name__ == "__main__":
    pass
