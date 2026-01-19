# Features, design, issues, limitations, and future prospects

This document aims to provide a brief overview of features, design decisions made during development, known limitations, and possible future improvements.

## Features
The current implementation's features are briefly listed below. For specific results see [Results](../docs/Results.md), for examples see [Tutorial](../docs/Tutorial.md), and the `notebooks` folder contains explanatory notebooks demonstrating the code and reproducing results.

As of 17 January 2026, the implementation includes:
- **(CS)RHF and (CS)UHF:** Complex-scaled restricted and unrestricted Hartree–Fock calculations.
- **Convergence acceleration:** DIIS and CROP algorithms.
- **(CS)RMP2 and (CS)UMP2:** Complex-scaled and standard second-order Møller–Plesset (MP2) perturbation theory.
- **PySCF interface:** Integrals and basis sets are obtained via PySCF. A small interface converts PySCF integrals to internal classes for the calculations.
- **Excited-state convergence:** A simple occupation-mask approach builds the density to target certain excited determinants.

## Design
This section describes the main design decisions taken during the development of this project, as well as the reasoning behind them. Most decisions were based on standard SCF/MP2 literature, but some design choices were made to overcome specific limitations.

As of 17 January 2026, this is a list of these decisions:

- **Implementation of a RHF guess (for UHF calculations):** In order to improve the initial guess for the SCF procedure, a RHF guess was implemented. This guess is based on a standard RHF over some cycles, and the resulting orbitals are used as an initial guess for the complex-scaled calculation. This was implemented due to systems such as $\mathrm{Li}$, where the core Hamiltonian guess combined with acceleration algorithms led to convergence to an incorrect solution. After the implementation of the RHF guess, the $\mathrm{Li}$ atom yielded correct results.
- **UHF density breaking:** Regarding the previous RHF guess, in order to break symmetry for UHF calculations, the current approach is similar, but not identical, to the one mentioned in [PySCF's forum](https://github.com/pyscf/pyscf.github.io/blob/master/examples/scf/32-break_spin_symm.py). In that case, since they have an implemented guess based on [a superposition of atomic densities](https://pyscf.org/user/scf.html), they have a better initial guess than the *Core* guess. They simply break symmetry by setting the density of some part of the beta density matrix to zero: `dm_beta[:2,:2] = 0`. This particular example was performed for $\mathrm{H}_2$ (hence the indices 2). Trying this approach in larger systems resulted in the numerically poor guess issue described in the following section; therefore, a different approach was implemented. The current approach is to average out the alpha and beta density matrices, and then eliminate some part of the beta density. In the currently tested cases, convergence to an unrestricted state was achieved with no further issues. Further approaches could be implemented in the future (see future prospects).
- **First unscaled and then scaled calculations:** In the current implementation, the SCF procedure is performed in two steps. First, a standard unscaled calculation is performed, and the resulting orbitals are used as an initial guess for the scaled calculation. This approach was chosen to avoid optimization in the complex plane from a poor guess (since the currently implemented guesses are the core Hamiltonian and the previously mentioned RHF guess). Optimizing both at the same time presented problems exemplified in $\mathrm{Li}$, where the poor guess combined with the complex-scaled calculation led to convergence at an incorrect solution.
- **Pseudo overlap calculation:** In order to converge to excited states such as the $\mathrm{He}\;2s^2$ resonance, the density was calculated using the occupation determinant as a mask (i.e., unoccupied orbitals, by order, don't contribute to the density). This approach has worked for the current tests in noble gases. However, a proper MOM should be implemented.
- **MGS reorthonormalization of degenerate orbitals:** In order to obtain a correct description of orbitals, the Modified Gram-Schmidt reorthonormalization procedure was implemented to reorthonormalize degenerate orbitals after each diagonalization of the Fock matrix. However, there are two (potential) issues with the current implementation. The first is that since we are working with complex-scaled calculations, eigenvalues can be complex, and thus the degeneracy criterion is currently a bit loose. The second is that tests have been carried out mostly on atoms, where degeneracy between MOs is always due to different shells, not accidental degeneracies that could arise in molecules. Therefore, there is room for improvement in this section.

## Known issues
This section provides a brief description of currently known issues and possible ways to tackle them. It might be interconnected with the future prospects, as in many cases an issue might be the consequence of something that has not been implemented yet. Any workarounds performed for specific issues are described here.

As of 17 January 2026, this is the list of acknowledged issues in the current implementation:
- **Contamination of the imaginary component in certain non-scaled calculations:** In some cases, when performing calculations without complex scaling, a small imaginary component appears in the results. The current theory is that the general eigenproblem solver `eig`, when used on Hermitian matrices using the `complex` type, leads to small numerical inaccuracies. The current workaround is to enforce the use of the Hermitian solver `eigh` when no complex scaling is used, which seems to solve the issue. Further tests are required to determine the appearance of this issue. A warning has been implemented to alert the user when the generalized `eig` is used on a matrix that could be diagonalized with `eigh`.
- **Portability:** The current state of the code is completely based in Python, which leads to a portability problem regarding other languages. Even though the code has been written trying to use the least amount of Python features, some parts of the code still require them. For the moment, the ideas considered to solve this issue are:
    - Performing strict memory management instead of using Python's pop/append in lists.
    - Using temporal matrices allocated at the beginning of the calculations instead of creating new ones in each step.
- **Performance:** The current implementation is much slower than established codes like PySCF.
- **Issues in certain dissociation curves of large elements:** The choice of an incorrect guess can lead to numerical instability and break the "diagonality of the diagonalized Fock matrix" (i.e., the product $C^TFC \neq \varepsilon \delta_{ij}$). Currently, the code is implemented to halt the calculation and raise an error. However, the nature of this numerical instability should be determined.

## Future improvements
This section describes possible future improvements that could be implemented in the code. Some appear due to the known issues and previous decisions, while others are the logical next steps in the development of the code.

- **Implementation of a proper MOM:** The current implementation uses a trick to converge to excited states. However, a proper Maximum Overlap Method should be implemented to improve the convergence to excited states.
- **Improved symmetry breaking methods:** The current method to break symmetry in UHF calculations is somewhat arbitrary and might not work in all cases. Therefore, other methods should be explored, such as adding random noise to the density matrix or using a more sophisticated approach based on perturbation theory.
- **BSE API:** Currently, basis sets are obtained from PySCF or manually defined. However, a small script to obtain basis sets from the [Basis Set Exchange](https://www.basissetexchange.org/) should not be too hard to implement and would be quite convenient.
- **Native integral support:** Currently, the code relies on PySCF to perform the integral calculations. However, a native implementation of the integral calculations could allow the use of other approaches, such as GTOs with complex exponents.
- **Performance improvements:** The current implementation is way too slow compared to established codes. We must learn how to optimize the code as much as possible and perhaps eventually port the most intensive parts.

# References
For a complete list of references used during development, see [References](../docs/References.md).