# Tutorial of the current code implementation
This document provides a basic description of the implementation's functionality through simple examples. For concrete results see the [Results](Results.md) section.

## Core functionality
This code is focused on complex scaled calculations. In order to describe the implementation, it is necessary to consider how it runs under the hood. 

The calculation takes place using an input struct and creates an output struct. The input struct will be from now on referred as context. The contexts currently available are: `CSRHFContext`, `CSUHFContext`, `CSRMP2Context`, and `CSUMP2Context`. These are the initial contexts for the complex scaled Restricted and Unrestricted Hartree-Fock and MP2 calculations respectively.

## RHF examples
We will start with simple RHF calculations, introducing eventually scaling and custom occupation. The current implementation of the (CSSCF) is:

![alt text](Figures/CSSCF-workflow.png)

### Non-scaled RHF calculation of Helium atom


First of all, we need to define a context. Consider that this code is based (at this point) on assuming that integrals are calculated elsewhere, so we will assume that integrals `S`, `T`, `V`, and `eris` exist and are in a `ndarray`. These integrals could have been read from files, or calculated by PySCF as we will comment later. The number of electrons needs to be specified too. 

```python
from py_mods.src.SCF.types import CSRHFContext

He_cxt = CSRHFContext(S, T, V, eris, 2)

```

Also, if we want to use directly PySCF to precompute the integrals we can use a function that interfaces with pyscf passing a dictionary, which returns a `CSRHFContext` object: 

```python
from py_mods.src.SCF.external import RHF_context_from_pyscf

pyscf_args = {"atom": "He 0 0 0", "spin": 0, "charge": 0, "basis": "sto-3g"}

He_cxt = RHF_context_from_pyscf(**pyscf_args)
```

Now the calculation is ready to run. For this we call the `CS_RHF` function:

```python
from py_mods.src.SCF.CSRHF import CS_RHF

He_results = CS_RHF(He_cxt)

```

That results in: 

```
 -------------------------------------------------------------------------------------
|  Iter  |             E_iter            |            Delta_e            |  norm(r_i) |
 -------------------------------------------------------------------------------------
      1      0.000000E+00+0.000000E+00j      0.000000E+00+0.000000E+00j     0.0000E+00
      2     -2.750138E+00+0.000000E+00j     -2.750138E+00+0.000000E+00j     1.6570E+00
      3     -2.860139E+00+0.000000E+00j     -1.100010E-01+0.000000E+00j     1.2436E-01
      4     -2.861481E+00+0.000000E+00j     -1.342423E-03+0.000000E+00j     1.3808E-02
      5     -2.861513E+00+0.000000E+00j     -3.210857E-05+0.000000E+00j     2.2763E-03
      6     -2.861514E+00+0.000000E+00j     -1.110485E-06+0.000000E+00j     4.2153E-04
      7     -2.861514E+00+0.000000E+00j     -4.213369E-08+0.000000E+00j     8.0819E-05
      8     -2.861514E+00+0.000000E+00j     -1.628149E-09+0.000000E+00j     1.5725E-05
      9     -2.861514E+00+0.000000E+00j     -6.312995E-11+0.000000E+00j     3.0805E-06
     10     -2.861514E+00+0.000000E+00j     -2.453149E-12+0.000000E+00j     6.0535E-07
     11     -2.861514E+00+0.000000E+00j     -9.459100E-14+0.000000E+00j     1.1912E-07
     12     -2.861514E+00+0.000000E+00j     -3.108624E-15+0.000000E+00j     2.3456E-08
     13     -2.861514E+00+0.000000E+00j     -1.776357E-15+0.000000E+00j     4.6198E-09
     14     -2.861514E+00+0.000000E+00j      8.881784E-16+0.000000E+00j     9.1000E-10
   ------------------------------    STARTED DIIS   ------------------------------
     15     -2.861514E+00+0.000000E+00j      2.664535E-15+0.000000E+00j     1.7925E-10
     16     -2.861514E+00+0.000000E+00j     -1.776357E-15+0.000000E+00j     2.7512E-13
Convergence achieved after 16 iterations.
```

Where all the results are stored in the `He_results` class. 

#### RHF parameters
We have introduced the basic RHF calculation. The UHF and MP2 calculations are similar in structure, just changing the context and the function called. First we will take a look at the RHF parameters that can be modified in the context, that will lead to different calculation behaviors. Currently these attrubutes are:

| Name | Optional | Type | Description |
| :--- | :--- | :--- | :--- |
| `S` | No | `NDArray[np.float64]` | Overlap matrix. |
| `T` | No | `NDArray[np.float64]` | Kinetic energy matrix. |
| `V` | No | `NDArray[np.float64]` | Nuclear attraction matrix. |
| `eri` | No | `NDArray[np.float64]` | Electron repulsion integrals. |
| `n_electrons` | No | `int` | Total electron count (must be even). |
| `theta` | Yes | `float` | Complex-scaling angle in radians (default: `0.0`). |
| `occupation` | Yes | `int`, `NDArray[np.int32]` | Occupation vector. If `None` (default), it is built automatically. |
| `max_iter` | Yes | `int` | Maximum SCF iterations (default: `100`). |
| `threshold` | Yes | `float` | Convergence threshold (default: `1e-12`). |
| `p_guess` | Yes | `{'core', 'ones', 'INPORB'}` | Initial density guess type (default: `'core'`). |
| `initial_orbitals` | Yes | `NDArray[float/complex]` | Imported orbitals. Required if `p_guess='INPORB'` (default: `None`). |
| `verbose` | Yes | `bool` | If `True`, print progress (default: `False`). |
| `conv_type` | Yes | `{'DIIS', 'CROP', None}` | Convergence algorithm (default: `'DIIS'`). |
| `acc_hist_size` | Yes | `int` | History size for convergence acceleration (default: `10`). |
| `acc_iteration_start` | Yes | `int` | Iteration to start acceleration (default: `12`). |

In these atributes we can define the complex scaling angle `theta`, the occupation, the convergence algorithm (DIIS or CROP) and parameters, the initial guess type, the maximum number of iterations, the convergence threshold, and verbose. We will briefly discuss the different options of each parameter:
- `theta`: Complex scaling angle in radians. Default is `0.0` (no scaling).
- `occupation`: Occupation vector. If `None`, it is built automatically based on the number of electrons. By passing a custom occupation vector, one can define different electronic states.
- `conv_type`: Convergence algorithm. Options are `'DIIS'` (Direct Inversion in the Iterative Subspace), `'CROP'` (Convergence acceleration by optimal parameterization), or `None` (no acceleration). Default is `'DIIS'`. The internal parameters of convergence acceleration are:
    - `acc_hist_size`: History size for convergence acceleration. This is the number of guesses stored for the extrapolation. Default is `10`.
    - `acc_iteration_start`: Iteration to start acceleration. Default is `12`.
- `p_guess`: Initial density guess type. Options are:
    - `'core'`: Core Hamiltonian guess (default).
    - `'ones'`: All ones guess.
    - `'INPORB'`: Import initial orbitals from `initial_orbitals` attribute. If selected, `initial_orbitals` must be provided.
- `max_iter`: Maximum number of SCF iterations. Default is `100`.
- `threshold`: Convergence threshold for the SCF procedure. Default is `1e-12`.
- `verbose`: If `True`, print progress during the SCF iterations. Default is `False`.

### Complex-scaled RHF calculation of Helium atom

A complex-scaled RHF calculation is as simple as defining the `theta` attribute in the context. For example, for a scaling of `0.2` radians:

```python
He_cxt.theta = 0.2
He_results = CS_RHF(He_cxt)
```

Where we obtain something that looks like:

```
--------------------------------------------------------------------------------------------------------------------------------
|   Iter     |                   E_iter                      |                   Delta_e                   |      norm(e_i)      |
--------------------------------------------------------------------------------------------------------------------------------
      1      0.000000E+00+0.000000E+00j      0.000000E+00+0.000000E+00j     0.0000E+00
      2     -2.750138E+00+0.000000E+00j     -2.750138E+00+0.000000E+00j     1.6570E+00
      3     -2.860139E+00+0.000000E+00j     -1.100010E-01+0.000000E+00j     1.2436E-01
      4     -2.861481E+00+0.000000E+00j     -1.342423E-03+0.000000E+00j     1.3808E-02
      5     -2.861513E+00+0.000000E+00j     -3.210857E-05+0.000000E+00j     2.2763E-03
      6     -2.861514E+00+0.000000E+00j     -1.110485E-06+0.000000E+00j     4.2153E-04
      7     -2.861514E+00+0.000000E+00j     -4.213369E-08+0.000000E+00j     8.0819E-05
      8     -2.861514E+00+0.000000E+00j     -1.628149E-09+0.000000E+00j     1.5725E-05
      9     -2.861514E+00+0.000000E+00j     -6.312995E-11+0.000000E+00j     3.0805E-06
     10     -2.861514E+00+0.000000E+00j     -2.453149E-12+0.000000E+00j     6.0535E-07
     11     -2.861514E+00+0.000000E+00j     -9.459100E-14+0.000000E+00j     1.1912E-07
     12     -2.861514E+00+0.000000E+00j     -3.108624E-15+0.000000E+00j     2.3456E-08
     13     -2.861514E+00+0.000000E+00j     -1.776357E-15+0.000000E+00j     4.6198E-09
     14     -2.861514E+00+0.000000E+00j      8.881784E-16+0.000000E+00j     9.1000E-10
------------------------------    STARTED DIIS   ------------------------------
     15     -2.861514E+00+0.000000E+00j      2.664535E-15+0.000000E+00j     1.7925E-10
     16     -2.861514E+00+0.000000E+00j     -1.776357E-15+0.000000E+00j     2.7512E-13
Convergence achieved after 16 iterations.

Unscaled energy:  (-2.861514227228338+0j)

Converging scaled case from unscaled density as reference:
--------------------------------------------------------------------------------------------------------------------------------
|   Iter     |                   E_iter                      |                   Delta_e                   |      norm(e_i)      |
--------------------------------------------------------------------------------------------------------------------------------
      1     -2.973320E+00+2.266421E-02j     -1.118055E-01+2.266421E-02j     2.0854E+00
      2     -2.862389E+00+4.851886E-06j      1.109308E-01-2.265936E-02j     1.0552E-01
      3     -2.861639E+00+8.960810E-05j      7.496050E-04+8.475621E-05j     9.8133E-03
      4     -2.861625E+00+1.007538E-04j      1.466666E-05+1.114567E-05j     1.8835E-03
      5     -2.861624E+00+1.013744E-04j      3.822223E-07+6.206118E-07j     3.8482E-04
      6     -2.861624E+00+1.014044E-04j      6.724789E-09+3.002127E-08j     7.8721E-05
      7     -2.861624E+00+1.014057E-04j     -1.412892E-10+1.303846E-09j     1.6167E-05
      8     -2.861624E+00+1.014058E-04j     -2.365796E-11+5.075518E-11j     3.3304E-06
      9     -2.861624E+00+1.014058E-04j     -1.655120E-12+1.726286E-12j     6.8727E-07
...
------------------------------    STARTED DIIS   ------------------------------
     15     -2.861624E+00+1.014058E-04j      0.000000E+00+0.000000E+00j     5.3493E-11
     16     -2.861624E+00+1.014058E-04j      4.440892E-16+0.000000E+00j     3.8401E-14
Convergence achieved after 16 iterations.
```

Here the final energy has a small imaginary part, corresponding to the resonance width.

### Custom occupation RHF calculation of He atom
In order to define a custom occupation, we can pass an occupation vector to the context. For example, to define an excited singled state in Helium ($2s2$), we can do:

```python
import numpy as np
He_cxt.occupation = np.array([0, 2, 0, 0])

He_results = CS_RHF(He_cxt)
```

Where we obtain:

```
--------------------------------------------------------------------------------------------------------------------------------
|   Iter     |                   E_iter                      |                   Delta_e                   |      norm(e_i)      |
--------------------------------------------------------------------------------------------------------------------------------
      1      0.000000E+00+0.000000E+00j      0.000000E+00+0.000000E+00j     0.0000E+00
      2     -4.827788E-01+0.000000E+00j     -4.827788E-01+0.000000E+00j     1.6296E-01
      3     -4.855469E-01+0.000000E+00j     -2.768099E-03+0.000000E+00j     5.2221E-02
      4     -4.853572E-01+0.000000E+00j      1.896836E-04+0.000000E+00j     1.3635E-02
      5     -4.853378E-01+0.000000E+00j      1.946605E-05+0.000000E+00j     4.0252E-03
      6     -4.853361E-01+0.000000E+00j      1.641326E-06+0.000000E+00j     1.1427E-03
      7     -4.853360E-01+0.000000E+00j      1.354397E-07+0.000000E+00j     3.2904E-04
      8     -4.853360E-01+0.000000E+00j      1.115189E-08+0.000000E+00j     9.4271E-05
      9     -4.853360E-01+0.000000E+00j      9.178537E-10+0.000000E+00j     2.7058E-05
     10     -4.853360E-01+0.000000E+00j      7.554174E-11+0.000000E+00j     7.7611E-06
     11     -4.853360E-01+0.000000E+00j      6.218026E-12+0.000000E+00j     2.2267E-06
     12     -4.853360E-01+0.000000E+00j      5.111467E-13+0.000000E+00j     6.3878E-07
     13     -4.853360E-01+0.000000E+00j      4.285461E-14+0.000000E+00j     1.8326E-07
     14     -4.853360E-01+0.000000E+00j      4.163336E-15+0.000000E+00j     5.2574E-08
------------------------------    STARTED DIIS   ------------------------------
     15     -4.853360E-01+0.000000E+00j     -3.885781E-16+0.000000E+00j     1.5083E-08
     16     -4.853360E-01+0.000000E+00j      0.000000E+00+0.000000E+00j     2.0265E-13
Convergence achieved after 16 iterations.
```

With a higher energy than the one of the ground state ($-2.8615$).

### CS calculation of an excited state
Now, combining both previous examples, we can compute a complex-scaled RHF calculation of the excited $2s2$ state of Helium:

```python
He_cxt.theta = 0.01
He_cxt.occupation = np.array([0, 2, 0, 0])
He_results = CS_RHF(He_cxt)
```

Where we obtain:

```--------------------------------------------------------------------------------------------------------------------------------
|   Iter     |                   E_iter                      |                   Delta_e                   |      norm(e_i)      |
--------------------------------------------------------------------------------------------------------------------------------
      1      0.000000E+00+0.000000E+00j      0.000000E+00+0.000000E+00j     0.0000E+00
      2     -4.827788E-01+0.000000E+00j     -4.827788E-01+0.000000E+00j     1.6296E-01
      3     -4.855469E-01+0.000000E+00j     -2.768099E-03+0.000000E+00j     5.2221E-02
      4     -4.853572E-01+0.000000E+00j      1.896836E-04+0.000000E+00j     1.3635E-02
      5     -4.853378E-01+0.000000E+00j      1.946605E-05+0.000000E+00j     4.0252E-03
      6     -4.853361E-01+0.000000E+00j      1.641326E-06+0.000000E+00j     1.1427E-03
      7     -4.853360E-01+0.000000E+00j      1.354397E-07+0.000000E+00j     3.2904E-04
      8     -4.853360E-01+0.000000E+00j      1.115189E-08+0.000000E+00j     9.4271E-05
      9     -4.853360E-01+0.000000E+00j      9.178537E-10+0.000000E+00j     2.7058E-05
     10     -4.853360E-01+0.000000E+00j      7.554174E-11+0.000000E+00j     7.7611E-06
     11     -4.853360E-01+0.000000E+00j      6.218026E-12+0.000000E+00j     2.2267E-06
     12     -4.853360E-01+0.000000E+00j      5.111467E-13+0.000000E+00j     6.3878E-07
     13     -4.853360E-01+0.000000E+00j      4.285461E-14+0.000000E+00j     1.8326E-07
     14     -4.853360E-01+0.000000E+00j      4.163336E-15+0.000000E+00j     5.2574E-08
------------------------------    STARTED DIIS   ------------------------------
     15     -4.853360E-01+0.000000E+00j     -3.885781E-16+0.000000E+00j     1.5083E-08
     16     -4.853360E-01+0.000000E+00j      0.000000E+00+0.000000E+00j     2.0265E-13
Convergence achieved after 16 iterations.
 
Unscaled energy:  (-0.48533598438938774+0j)

Converging scaled case from unscaled density as reference:
--------------------------------------------------------------------------------------------------------------------------------
|   Iter     |                   E_iter                      |                   Delta_e                   |      norm(e_i)      |
--------------------------------------------------------------------------------------------------------------------------------
      1     -4.855691E-01-1.230342E-02j     -2.331032E-04-1.230342E-02j     4.7840E-02
      2     -4.855845E-01-1.230285E-02j     -1.541510E-05+5.651145E-07j     1.3629E-02
      3     -4.856003E-01-1.230225E-02j     -1.580367E-05+6.084994E-07j     3.6800E-03
      4     -4.856017E-01-1.230221E-02j     -1.420671E-06+3.753163E-08j     1.0772E-03
      5     -4.856018E-01-1.230221E-02j     -1.182489E-07+2.453019E-09j     3.0682E-04
      6     -4.856019E-01-1.230220E-02j     -9.751081E-09+1.583032E-10j     8.8279E-05
      7     -4.856019E-01-1.230220E-02j     -8.030990E-10+9.712890E-12j     2.5307E-05
      8     -4.856019E-01-1.230220E-02j     -6.612993E-11+5.275450E-13j     7.2644E-06
      9     -4.856019E-01-1.230220E-02j     -5.446921E-12+2.114801E-14j     2.0842E-06
     10     -4.856019E-01-1.230220E-02j     -4.488632E-13-1.162265E-16j     5.9810E-07
     11     -4.856019E-01-1.230220E-02j     -3.513856E-14-1.890849E-16j     1.7162E-07
     12     -4.856019E-01-1.230220E-02j     -2.498002E-15-2.602085E-17j     4.9247E-08
     13     -4.856019E-01-1.230220E-02j     -2.720046E-15-5.204170E-18j     1.4131E-08
     14     -4.856019E-01-1.230220E-02j      1.720846E-15+1.387779E-17j     4.0550E-09
------------------------------    STARTED DIIS   ------------------------------
     15     -4.856019E-01-1.230220E-02j     -1.221245E-15+1.040834E-17j     1.1636E-09
     16     -4.856019E-01-1.230220E-02j      1.110223E-15+1.734723E-17j     7.2768E-15
Convergence achieved after 16 iterations.
```

Where the scaled energy possesses now a small imaginary part, corresponding to the resonance width of the excited state.

### CSRHF theta trajectories
The complex scaling angle `theta` can be varied to create theta trajectories. This is done by running several calculations with different `theta` values. For example, to create a theta trajectory for Helium from `0.0` to `0.8` radians in $9$ steps we can use:

```python
from py_mods.src.SCF.CSRHF import rhf_theta_traj
import matplotlib.pyplot as plt 

pyscf_args = {"atom": "He 0 0 0", "spin": 0, "charge": 0, "basis": "cc-pvqz"}
He_cxt = RHF_context_from_pyscf(**pyscf_args)
traj_ener_2 = rhf_theta_traj(0.08, 9, He_cxt)

He_2s2_qz_gs = np.array(traj_ener_2[1])

plt.scatter(
    He_2s2_qz_gs.real,
    He_2s2_qz_gs.imag,
    label="Implemetation",
    c="RebeccaPurple",
    marker="x",
)
plt.title("CS-RHF Theta Trajectory for He 2s2 (cc-pVQZ basis)")
plt.legend()
plt.xlabel("Re(Ener)")
plt.ylabel("Im(Ener)")
plt.show()
```

Resulting in:

![alt text](Figures/theta_traj_example.png)

Where each point (from right to left) corresponds to an increase of $0.01$ in $\theta$. 

## MP2 examples 
The MP2 calculations are very similar to the RHF ones, just changing the context and the function called. The context for a MP2 calculation is the results of a previous SCF calculation. In particular, the context can be a `CSRHFResults` or a `CSUHFResults` object.

```python
from py_mods.src.MP2.CSMP2 import CS_MP2
pyscf_args = {"atom": "He 0 0 0", "spin": 0, "charge": 0, "basis": "cc-pvqz"}
He_cxt = RHF_context_from_pyscf(**pyscf_args)

He_results = CS_RHF(He_cxt)
He_MP2_results = CS_MP2(He_results)
```

Where we can extract the values from this object:

```python
print(f"Correlation energy (MP2): {He_MP2_results.E_corr.real:4e}")
print(f"Total energy (RHF + MP2): {He_MP2_results.E_MP2.real:4e}")
```

Resulting in:

```
Correlation energy (MP2): -3.547800e-02
Total energy (RHF + MP2): -2.896992e+00
```

### Complex-scaled MP2 calculation
Similarly to the RHF case, we can perform a complex-scaled MP2 calculation by providing a complex-scaled SCF results object as context. For example, using the previously calculated complex-scaled RHF results:

```python
He_cxt = RHF_context_from_pyscf(**pyscf_args)
He_cxt.theta = 0.2
He_results_cs = CS_RHF(He_cxt)
He_MP2_results_cs = CS_MP2(He_results_cs)
```

Unpacking:

```python
print(f"Correlation energy (MP2): {He_MP2_results_cs.E_corr:4e}")
print(f"Total energy (RHF + MP2): {He_MP2_results_cs.E_MP2:4e}")
```

Resulting in:

```
Correlation energy (MP2): -3.579815e-02+1.005355e-04j
Total energy (RHF + MP2): -2.897422e+00+2.019413e-04j
```

### MP2 calculation with excited states 
Similarly to the RHF case, we can define custom occupations in the SCF calculation before performing the MP2 calculation. For example, defining the $2s2$ excited state in Helium:

```python
pyscf_args = {"atom": "He 0 0 0", "spin": 0, "charge": 0, "basis": "cc-pvqz"}
He_cxt = RHF_context_from_pyscf(**pyscf_args)
He_cxt.occupation = np.array([0, 2, 0, 0])

He_results = CS_RHF(He_cxt)
He_MP2_results = CS_MP2(He_results)
```

We can get the results as previously:
```python
print(f"Correlation energy (MP2): {He_MP2_results.E_corr.real:4e}")
print(f"Total energy (RHF + MP2): {He_MP2_results.E_MP2.real:4e}")
``` 

That are: 

```
Correlation energy (MP2): -8.418459e-03
Total energy (RHF + MP2): -4.937544e-01
```

And both can be combined as in the SCF case to perform a complex-scaled MP2 calculation of an excited state:

```python
pyscf_args = {"atom": "He 0 0 0", "spin": 0, "charge": 0, "basis": "cc-pvqz"}
He_cxt = RHF_context_from_pyscf(**pyscf_args)

He_cxt.theta = 0.01
He_cxt.occupation = np.array([0, 2, 0, 0])
He_results_cs = CS_RHF(He_cxt)

He_MP2_results_cs = CS_MP2(He_results_cs)
```

## Results 
These calculations can be run interactively in a [Jupyter notebook](../notebooks/Examples/1_Performing_a_calculation.ipynb).

UHF examples are provided in the following [Notebook](../notebooks/CS-SCF%20results/3_UHF_examples.ipynb). However, the input parameters are similar with a few exceptions to the ones shown, in order to adapt to the unrestricted formalism. 

For examples of results obtained with this code, see [Results](../docs/Results.md) and for more use cases check the **Notebooks directory**.

# References
For a complete list of references, see [References](../docs/References.md).