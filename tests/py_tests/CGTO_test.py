from py_mods.src.integrals.CGTO import (
    create_CGTOClass,
    S_GTO_mat,
    _generate_angular_momentum_projections,
    T_GTO_mat,
    V_GTO_mat,
    Eri_GTO_tensor,
)

from pyscf import gto
import numpy as np


def test_1s1s():
    # STO3g example:
    l = 0
    l_tags = ["S", "P", "D", "F", "G", "H"]
    l_projs = len(_generate_angular_momentum_projections(l))

    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([0.0, 0.0, 1.4])
    H_exps = np.array([3.42525091, 0.62391373, 0.16885540])
    H_coeffs = np.array([0.15432897, 0.53532814, 0.44463454])

    atom_pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    atom_charges = np.array([1.0, 1.0])

    # Construct the CGTO
    H_1s_1 = create_CGTOClass(r1, H_exps, H_coeffs, l)
    H_1s_2 = create_CGTOClass(r2, H_exps, H_coeffs, l)

    S_1s1s_sto3g_self = S_GTO_mat(H_1s_1, H_1s_2)
    T_1s1s_sto3g_self = T_GTO_mat(H_1s_1, H_1s_2)
    V_1s1s_sto3g_self = V_GTO_mat(H_1s_1, H_1s_2, atom_pos, atom_charges)
    Eri_1s1s1s1s_self = Eri_GTO_tensor(H_1s_1, H_1s_2, H_1s_1, H_1s_1)

    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        unit="Bohr",
        basis={
            "H": gto.basis.parse(
                f"""
                    H {l_tags[l]}
                    3.42525091    0.15432897
                    0.62391373    0.53532814
                    0.16885540    0.44463454
                """
            )
        },
        cart=True,
    )

    mol.build()

    overlap = mol.intor("int1e_ovlp")
    kin = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    ref_eri = mol.intor("int2e")

    # pyscf does not normalize in cartesian like this, we have to renormalize to compare
    norm_vec = 1.0 / np.sqrt(np.diag(overlap))

    overlap *= norm_vec[:, None]
    overlap *= norm_vec[None, :]

    kin *= norm_vec[:, None]
    kin *= norm_vec[None, :]

    V *= norm_vec[:, None]
    V *= norm_vec[None, :]

    ref_eri *= norm_vec[:, None, None, None]
    ref_eri *= norm_vec[None, :, None, None]
    ref_eri *= norm_vec[None, None, :, None]
    ref_eri *= norm_vec[None, None, None, :]

    assert np.allclose(S_1s1s_sto3g_self, overlap[l_projs:, :l_projs])
    assert np.allclose(T_1s1s_sto3g_self, kin[l_projs:, :l_projs])
    assert np.allclose(V_1s1s_sto3g_self, V[l_projs:, :l_projs])
    assert np.allclose(
        Eri_1s1s1s1s_self, ref_eri[l_projs:, :l_projs, l_projs:, l_projs:]
    )


def test_2p2p():
    # STO3g example:
    l = 1
    l_tags = ["S", "P", "D", "F", "G", "H"]
    l_projs = len(_generate_angular_momentum_projections(l))

    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([0.0, 0.0, 1.4])
    H_exps = np.array([3.42525091, 0.62391373, 0.16885540])
    H_coeffs = np.array([0.15432897, 0.53532814, 0.44463454])

    atom_pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    atom_charges = np.array([1.0, 1.0])

    # Construct the CGTO
    H_1s_1 = create_CGTOClass(r1, H_exps, H_coeffs, l)
    H_1s_2 = create_CGTOClass(r2, H_exps, H_coeffs, l)

    S_1s1s_sto3g_self = S_GTO_mat(H_1s_1, H_1s_2)
    T_1s1s_sto3g_self = T_GTO_mat(H_1s_1, H_1s_2)
    V_1s1s_sto3g_self = V_GTO_mat(H_1s_1, H_1s_2, atom_pos, atom_charges)
    Eri_1s1s1s1s_self = Eri_GTO_tensor(H_1s_1, H_1s_2, H_1s_1, H_1s_1)

    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        unit="Bohr",
        basis={
            "H": gto.basis.parse(
                f"""
                    H {l_tags[l]}
                    3.42525091    0.15432897
                    0.62391373    0.53532814
                    0.16885540    0.44463454
                """
            )
        },
        cart=True,
    )

    mol.build()

    overlap = mol.intor("int1e_ovlp")
    kin = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    ref_eri = mol.intor("int2e")

    # pyscf does not normalize in cartesian like this, we have to renormalize to compare
    norm_vec = 1.0 / np.sqrt(np.diag(overlap))

    overlap *= norm_vec[:, None]
    overlap *= norm_vec[None, :]

    kin *= norm_vec[:, None]
    kin *= norm_vec[None, :]

    V *= norm_vec[:, None]
    V *= norm_vec[None, :]

    ref_eri *= norm_vec[:, None, None, None]
    ref_eri *= norm_vec[None, :, None, None]
    ref_eri *= norm_vec[None, None, :, None]
    ref_eri *= norm_vec[None, None, None, :]

    assert np.allclose(S_1s1s_sto3g_self, overlap[l_projs:, :l_projs])
    assert np.allclose(T_1s1s_sto3g_self, kin[l_projs:, :l_projs])
    assert np.allclose(V_1s1s_sto3g_self, V[l_projs:, :l_projs])
    assert np.allclose(
        Eri_1s1s1s1s_self, ref_eri[l_projs:, :l_projs, l_projs:, l_projs:]
    )


if __name__ == "__main__":
    test_1s1s()
    test_2p2p()
