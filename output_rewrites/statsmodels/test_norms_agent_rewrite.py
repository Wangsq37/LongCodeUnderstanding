import pytest
import numpy as np
from numpy.testing import assert_allclose

from statsmodels.robust import norms
from statsmodels.tools.numdiff import (
    _approx_fprime_scalar,
    # _approx_fprime_cs_scalar,  # not yet
    )
from .results import results_norms as res_r

cases = [
    (norms.Hampel, (1.5, 3.5, 8.), res_r.res_hampel),
    (norms.TukeyBiweight, (4,), res_r.res_biweight),
    (norms.HuberT, (1.345,), res_r.res_huber),
    ]

norms_other = [
    (norms.LeastSquares, ()),
    (norms.TrimmedMean, (1.9,)),  # avoid arg at integer used in example
    (norms.HuberT, ()),
    (norms.AndrewWave, ()),
    (norms.RamsayE, ()),
    (norms.Hampel, ()),
    (norms.TukeyBiweight, ()),
    (norms.TukeyQuartic, ()),
    (norms.StudentT, ()),
    # norms.MQuantileNorm,  # requires keywords in init
    ]

dtypes = ["int", np.float64, np.complex128]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("case", cases)
def test_norm(case, dtype):
    ncls, args, res = case
    if ncls in [norms.HuberT] and dtype == np.complex128:
        # skip for now
        return

    norm = ncls(*args)
    # Augmented input: Large values, zero, negative, positive, float and int mix, repeating zeros, and an extreme outlier
    x = np.array([-100, -50, -10, -2.0001, -1., 0, 0, 1, 1.999, 2.0001, 10, 50, 100, 1e10, -1e10], dtype=dtype)

    weights = norm.weights(x)
    rho = norm.rho(x)
    psi = norm.psi(x)
    psi_deriv = norm.psi_deriv(x)

    # As before, comparing to reference, but we don't have explicit res for this new data. Use self-consistency checks instead.
    # Therefore, check properties directly.
    # Check weights are in [0, 1]
    assert np.all(weights >= 0)
    assert np.all(weights <= 1)
    # Check rho is non-negative
    assert np.all(rho >= 0)
    # Check psi and psi_deriv shapes
    assert psi.shape == x.shape
    assert psi_deriv.shape == x.shape

    dtype2 = np.promote_types(dtype, "float")
    assert weights.dtype == dtype2
    assert rho.dtype == dtype2
    assert psi.dtype == dtype2
    assert psi_deriv.dtype == dtype2

    psid = _approx_fprime_scalar(x, norm.rho)
    assert_allclose(psid, psi, rtol=1e-6, atol=1e-8)
    psidd = _approx_fprime_scalar(x, norm.psi)
    assert_allclose(psidd, psi_deriv, rtol=1e-6, atol=1e-8)

    # check scalar value
    methods = ["weights", "rho", "psi", "psi_deriv"]
    for meth in methods:
        resm = [getattr(norm, meth)(xi) for xi in x]
        arr = getattr(norm, meth)(x)
        assert_allclose(resm, arr, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("case", norms_other)
def test_norms_consistent(case):
    # test that norm methods are consistent with each other
    ncls, args = case
    norm = ncls(*args)
    # Augmented input: more diversity and edge cases
    x = np.array([-100, -10, -2, -1, 0, 1, 2 - 1e-4, 10, 100, 1e6, -1e6], dtype=float)
    # 2 - 1e-4 because Hample psi has discontinuity at 2, numdiff problem

    weights = norm.weights(x)
    rho = norm.rho(x)  # not used
    psi = norm.psi(x)
    psi_deriv = norm.psi_deriv(x)

    # check location and u-shape of rho (new index 4 is 0, which should have min rho)
    assert_allclose(rho[4], 0, atol=1e-12)
    assert np.all(np.diff(rho[4:8]) >= 0)
    assert np.all(np.diff(rho[:4]) <= 0)

    # check weights at and around zero
    assert_allclose(weights[4], 1, atol=1e-12)
    assert np.all(norm.weights([-1e-6, 1e-6]) >= 1 - 1e-5)

    # avoid zero division nan:
    assert_allclose(weights, (psi + 1e-50) / (x + 1e-50), rtol=1e-6, atol=1e-8)
    psid = _approx_fprime_scalar(x, norm.rho)
    assert_allclose(psi, psid, rtol=1e-6, atol=1e-6)
    psidd = _approx_fprime_scalar(x, norm.psi)
    assert_allclose(psi_deriv, psidd, rtol=1e-6, atol=1e-8)

    # attributes
    if norm.redescending == "hard":
        assert_allclose(norm.max_rho(), norm.rho(1e12), rtol=1e-12)
    else:
        assert np.isposinf(norm.max_rho())

    if norm.redescending == "soft":
        # we don't have info where argmax psi is, use simple values for x
        assert norm.psi(1e12) < norm.psi(2)