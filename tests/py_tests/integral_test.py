from py_mods.src.integrals.primitive import S_3D, T_3D, V_3D, eri
from py_mods.src.integrals.primitive import Primitive, create_normalized_primitive
import numpy as np


def test_1s1s():
    He_1s = create_normalized_primitive(np.array([0, 0, 0]), 0.4, 0)
    H_1s = create_normalized_primitive(np.array([1, 0, 0]), 0.5, 0)

    S_1s1s = S_3D(
        He_1s,
        He_1s.l_projections[0],
        He_1s.normalization_constants[0],
        H_1s,
        H_1s.l_projections[0],
        H_1s.normalization_constants[0],
    )

    T_1s1s = T_3D(
        He_1s,
        He_1s.l_projections[0],
        He_1s.normalization_constants[0],
        H_1s,
        H_1s.l_projections[0],
        H_1s.normalization_constants[0],
    )

    V_1s1s = V_3D(
        He_1s,
        He_1s.l_projections[0],
        He_1s.normalization_constants[0],
        H_1s,
        H_1s.l_projections[0],
        H_1s.normalization_constants[0],
        2.0,
        np.array([0.0, 0.0, 0.0]),
    ) + V_3D(
        He_1s,
        He_1s.l_projections[0],
        He_1s.normalization_constants[0],
        H_1s,
        H_1s.l_projections[0],
        H_1s.normalization_constants[0],
        1.0,
        np.array([1.0, 0.0, 0.0]),
    )

    reference_eris = np.array(
        [
            [
                [[0.71364965, 0.55814139], [0.55814139, 0.65422141]],
                [[0.55814139, 0.47637504], [0.47637504, 0.59740119]],
            ],
            [
                [[0.55814139, 0.47637504], [0.47637504, 0.59740119]],
                [[0.65422141, 0.59740119], [0.59740119, 0.79788456]],
            ],
        ]
    )

    self_eris = np.zeros([2, 2, 2, 2])

    for i, prim_a in enumerate([He_1s, H_1s]):
        for j, prim_b in enumerate([He_1s, H_1s]):
            for k, prim_c in enumerate([He_1s, H_1s]):
                for l, prim_d in enumerate([He_1s, H_1s]):
                    self_eris[i, j, k, l] = eri(
                        prim_a,
                        prim_a.l_projections[0],
                        prim_a.normalization_constants[0],
                        prim_b,
                        prim_b.l_projections[0],
                        prim_b.normalization_constants[0],
                        prim_c,
                        prim_c.l_projections[0],
                        prim_c.normalization_constants[0],
                        prim_d,
                        prim_d.l_projections[0],
                        prim_d.normalization_constants[0],
                    )

    assert (
        np.abs(S_1s1s - 0.7933116667) < 1e-7
    ), f"S_1s1s test failed: {S_1s1s}, expected 0.7933116667"
    assert (
        np.abs(T_1s1s - 0.4505226749) < 1e-7
    ), f"T_1s1s test failed: {T_1s1s}, expected 0.4505226749"
    assert (
        np.abs(V_1s1s - (-2.3549300013)) < 1e-7
    ), f"V_He test failed: {V_1s1s}, expected -2.3549300013"

    assert np.all(
        np.abs(self_eris - reference_eris) < 1e-7
    ), f"ERI 1s1s test failed:\n{self_eris}\nexpected:\n{reference_eris}"
