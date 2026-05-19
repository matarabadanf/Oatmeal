def create_and_run_Uncontracted_matrix_element_test(positions, angular_momenta_list, charges) -> Bool:
    mol = gto.M(
    atom="H 0 0 0; H 0 0 1.4",
    unit="Bohr",
    basis={
        "H": gto.basis.parse(f"""
        H S
        0.3    1
        # 1    1
        H P 
        0.3    1
        """),
    },
    cart=True,
    )

    # Get overlap and kinetic energy matrices
    mol.build()
    print(mol.ao_labels())

    S_pyscf = mol.intor("int1e_ovlp")
    T_pyscf = mol.intor("int1e_kin")
    V_pyscf = mol.intor("int1e_nuc")
    ref_eri = mol.intor("int2e")

    norm_vec = 1.0 / np.sqrt(np.diag(S_pyscf))

    ref_eri *= norm_vec[:, None, None, None]
    ref_eri *= norm_vec[None, :, None, None]
    ref_eri *= norm_vec[None, None, :, None]
    ref_eri *= norm_vec[None, None, None, :]
    
    return False 