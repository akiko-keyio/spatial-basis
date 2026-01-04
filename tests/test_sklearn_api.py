"""Test sklearn API compatibility for SphericalHarmonicsBasis."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from spatial_basis import SphericalHarmonicsBasis


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


def test_get_params_set_params():
    """Test get_params and set_params (sklearn requirement)."""
    basis = SphericalHarmonicsBasis(degree=3, cup=False)
    
    params = basis.get_params()
    assert params['degree'] == 3
    assert params['cup'] == False
    
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
