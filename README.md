# Oatmeal: a tiny CS-SCF toy model



## Features
For now the implemented features are:
- Restricted Hartree-Fock (RHF).
- Complex Scaled RHF (CS-RHF) with occupation selection.
- Support for convergence acceleration methods:
    - DIIS ([Pulay](https://doi.org/10.1002/jcc.540030413)).
    - CROP ([Ettenhuber, Jorgensen](https://doi.org/10.1021/ct501114q)).

## Installation
Instal as a module to be able to import it. Clone, then:
```bash
cd Oatmeal
pip install -e .
```

## Project Structure

```
Oatmeal/
|- py_mods/
|    ─ src/
|        - SCF/                
|            ─ RHF.py         
|            - CSRHF.py        
|- tests/
|    - py_tests/              # Implementation tests
|- notebooks/                 # Jupyter notebooks with examples
|- setup.py
```

## Usage 
The current implementation depends on the integrals to be calculated elsewhere. In particular, PySCF's integrals were used.

For usage examples see see the Jupyter notebooks in `notebooks/`

## Tests
Tests can be run using `pytest` on the parent directory:
```bash
pytest .
```

## References
- Basis sets ([Basis set exchange](https://www.basissetexchange.org/)).
- [PySCF](https://doi.org/10.1063/5.0006074).
- DIIS ([Pulay](https://doi.org/10.1002/jcc.540030413)).
- CROP ([Ettenhuber, Jorgensen](https://doi.org/10.1021/ct501114q)).

