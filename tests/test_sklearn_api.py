"""Test sklearn API compatibility for SphericalHarmonicsBasis."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import sph_harm_y
from sklearn.base import clone

from spatial_basis import PolynomialBasis, SphericalHarmonicsBasis


# =============================================================================
# Basic API Tests
# =============================================================================

def test_basic_fit_transform():
    """Test that fit_transform works correctly."""
    X = np.array([[116.4, 39.9], [121.5, 31.2], [114.1, 22.6]])
    
    basis = SphericalHarmonicsBasis(degree=2, cup=True)
    X_transformed = basis.fit_transform(X)
    
    assert X_transformed.shape[0] == 3
    assert X_transformed.shape[1] == basis.n_output_features_


def test_normalize_uses_fitted_column_norms():
    """Test normalize reuses column norms learned during fit."""
    X_train = np.array(
        [
            [116.4, 39.9],
            [121.5, 31.2],
            [114.1, 22.6],
            [113.3, 23.1],
        ]
    )
    X_test = np.array([[118.8, 32.1], [110.2, 19.6]])

    basis = SphericalHarmonicsBasis(degree=2, cup=True, normalize=True)
    Xt_fit_transform = basis.fit_transform(X_train)
    Xt_transform_train = basis.transform(X_train)

    raw_basis = SphericalHarmonicsBasis(degree=2, cup=True, normalize=False)
    raw_basis.fit(X_train)
    Xt_raw_test = raw_basis.transform(X_test)
    Xt_transform_test = basis.transform(X_test)

    assert np.allclose(Xt_fit_transform, Xt_transform_train)
    assert np.allclose(
        Xt_transform_test,
        Xt_raw_test / basis.column_norms_,
    )


def test_transform_checks_feature_count():
    """Test that transform validates feature count."""
    rng = np.random.default_rng(0)
    X = rng.random((5, 2))
    basis = SphericalHarmonicsBasis()
    basis.fit(X)
    with pytest.raises(ValueError):
        basis.transform(rng.random((2, 3)))


def test_transform_checks_feature_names():
    """Test that transform validates feature names from pandas."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((5, 2)), columns=["lon", "lat"])
    basis = SphericalHarmonicsBasis()
    basis.fit(X)
    X_bad = pd.DataFrame(rng.random((2, 2)), columns=["x", "y"])
    with pytest.raises(ValueError):
        basis.transform(X_bad)


def test_get_feature_names_out():
    """Test get_feature_names_out returns correct names."""
    X = np.array([[116.4, 39.9], [121.5, 31.2]])
    
    basis = SphericalHarmonicsBasis(degree=2, cup=True)
    basis.fit(X)
    
    names = basis.get_feature_names_out()
    assert len(names) == basis.n_output_features_
    assert names.tolist() == ["Y:0,0", "Y:1,-1", "Y:1,1", "Y:2,-2", "Y:2,0", "Y:2,2"]


def test_get_feature_names_out_validates_features():
    """Test that get_feature_names_out validates input features."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.random((5, 2)), columns=["lon", "lat"])
    basis = SphericalHarmonicsBasis()
    basis.fit(X)
    with pytest.raises(ValueError):
        basis.get_feature_names_out(["x", "y"])


# =============================================================================
# sklearn Convention Tests
# =============================================================================

def test_fitted_attributes_naming():
    """Test that fitted attributes end with underscore (sklearn convention)."""
    X = np.array([[116.4, 39.9], [121.5, 31.2], [114.1, 22.6]])
    
    basis = SphericalHarmonicsBasis(degree=2, cup=True)
    basis.fit(X)
    
    # Check fitted attributes end with underscore
    assert hasattr(basis, 'min_vals_'), "min_vals_ should exist"
    assert hasattr(basis, 'max_vals_'), "max_vals_ should exist"
    assert hasattr(basis, 'coords_converter_'), "coords_converter_ should exist"
    assert hasattr(basis, 'terms_'), "terms_ should exist"
    assert hasattr(basis, 'n_output_features_'), "n_output_features_ should exist"
    
    # Check old attribute names don't exist
    assert not hasattr(basis, 'min_vals'), "min_vals should not exist"
    assert not hasattr(basis, 'max_vals'), "max_vals should not exist"
    assert not hasattr(basis, 'coords_convert'), "coords_convert should not exist"
    assert not hasattr(basis, 'terms'), "terms should not exist"


def test_transform_is_stateless():
    """Test that transform doesn't modify fitted state."""
    X = np.array([[116.4, 39.9], [121.5, 31.2], [114.1, 22.6]])
    
    basis = SphericalHarmonicsBasis(degree=2, cup=True)
    basis.fit(X)
    
    # Get state after fit
    terms_after_fit = basis.terms_.copy()
    n_output_after_fit = basis.n_output_features_
    
    # Transform
    basis.transform(X)
    
    # Verify state unchanged
    assert basis.terms_ == terms_after_fit, "terms_ should not change after transform"
    assert basis.n_output_features_ == n_output_after_fit, "n_output_features_ should not change"


def test_polynomial_fit_exposes_output_metadata():
    """Test PolynomialBasis exposes fitted output metadata."""
    X = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])

    basis = PolynomialBasis(degree=2, include_bias=True, basis="polynomial")
    basis.fit(X)

    expected_powers = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0]],
        dtype=int,
    )
    assert np.array_equal(basis.powers_, expected_powers)
    assert np.array_equal(basis.feature_degrees_, expected_powers.sum(axis=1))
    assert basis.n_output_features_ == expected_powers.shape[0]


def test_polynomial_output_metadata_aligns_with_transform_and_names():
    """Test PolynomialBasis uses powers_ as single output schema source."""
    X = pd.DataFrame(
        [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]],
        columns=["lon", "lat"],
    )

    basis = PolynomialBasis(degree=2, include_bias=True, basis="polynomial")
    Xt = basis.fit_transform(X)
    names = basis.get_feature_names_out()
    X_raw = X.to_numpy()
    X_scaled = 2 * (X_raw - basis.min_vals_) / basis.range_ - 1
    expected = np.column_stack(
        [
            (X_scaled[:, 0] ** i) * (X_scaled[:, 1] ** j)
            for i, j in basis.powers_
        ]
    )

    assert Xt.shape[1] == len(names) == len(basis.feature_degrees_) == len(basis.powers_)
    assert np.allclose(Xt, expected)
    assert names.tolist() == ["1", "lat", "lat^2", "lon", "lon lat", "lon^2"]


def test_spherical_fit_exposes_output_metadata():
    """Test SphericalHarmonicsBasis exposes fitted numeric output metadata."""
    X = np.array([[116.4, 39.9], [121.5, 31.2], [114.1, 22.6]])

    basis = SphericalHarmonicsBasis(degree=2, cup=True)
    basis.fit(X)

    expected_indices = np.array(
        [[0, 0], [1, -1], [1, 1], [2, -2], [2, 0], [2, 2]],
        dtype=int,
    )
    assert np.array_equal(basis.harmonic_indices_, expected_indices)
    assert np.array_equal(basis.feature_degrees_, expected_indices[:, 0])
    assert basis.n_output_features_ == expected_indices.shape[0]


def test_spherical_output_metadata_aligns_with_transform_and_names():
    """Test SphericalHarmonicsBasis keeps names, degrees and columns aligned."""
    X = np.array([[116.4, 39.9], [121.5, 31.2], [114.1, 22.6]])

    basis = SphericalHarmonicsBasis(degree=2, cup=True)
    Xt = basis.fit_transform(X)
    names = basis.get_feature_names_out()
    theta, phi = basis.coords_converter_.transform(X[:, 0], X[:, 1])

    expected = []
    for order, m in basis.harmonic_indices_:
        order = int(order)
        m = int(m)
        y_l_m_abs = sph_harm_y(order, abs(m), theta, phi)
        if m < 0:
            term = np.sqrt(2) * (-1) ** m * y_l_m_abs.imag
        elif m == 0:
            term = y_l_m_abs.real
        else:
            term = np.sqrt(2) * (-1) ** m * y_l_m_abs.real
        expected.append(term)
    expected = np.column_stack(expected) * np.sqrt(2)

    assert Xt.shape[1] == len(names) == len(basis.feature_degrees_)
    assert len(basis.harmonic_indices_) == len(names)
    assert np.allclose(Xt, expected)
    assert names.tolist() == ["Y:0,0", "Y:1,-1", "Y:1,1", "Y:2,-2", "Y:2,0", "Y:2,2"]


def test_get_params_set_params():
    """Test get_params and set_params (sklearn requirement)."""
    basis = SphericalHarmonicsBasis(degree=3, cup=False)
    
    params = basis.get_params()
    assert params['degree'] == 3
    assert not params['cup']
    
    basis.set_params(degree=5)
    assert basis.degree == 5


def test_clone():
    """Test that estimator can be cloned."""
    basis = SphericalHarmonicsBasis(degree=3, cup=True)
    basis_clone = clone(basis)
    
    assert basis_clone.degree == basis.degree
    assert basis_clone.cup == basis.cup


def test_pole_tuple():
    """Test that pole accepts tuple."""
    X = np.array([[116.4, 39.9], [121.5, 31.2]])
    
    basis = SphericalHarmonicsBasis(degree=2, pole=(30.0, 120.0))
    basis.fit(X)
    
    assert hasattr(basis, 'coords_converter_')
