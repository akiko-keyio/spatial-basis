import numpy as np
import pytest

from geospectra.basis import PolynomialBasis, SphericalHarmonicsBasis
from geospectra.linear_model import LinearRegressionCond
from joblib import Memory
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import OrthogonalMatchingPursuit


def _make_polynomial_data() -> tuple[np.ndarray, np.ndarray, PolynomialBasis]:
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(20, 2))
    basis = PolynomialBasis(degree=2, include_bias=True)
    design = basis.fit_transform(X)
    coef = rng.normal(size=(design.shape[1], 2))
    intercept = rng.normal(size=2)
    y = design @ coef + intercept
    return X, y, basis


class CallCountingBasis(PolynomialBasis):
    n_calls = 0

    def transform(self, X):
        CallCountingBasis.n_calls += 1
        return super().transform(X)


def test_polynomial_fit_predict_multi_target() -> None:
    X, y, basis = _make_polynomial_data()
    pipe = Pipeline(
        [
            ("basis", basis),
            ("reg", LinearRegressionCond(fit_intercept=True)),
        ]
    )
    pipe.fit(X, y)
    reg = pipe.named_steps["reg"]
    assert reg.coef_.shape == (2, basis.n_output_features_)
    pred = pipe.predict(X)
    assert np.allclose(pred, y)


def _make_spherical_data() -> tuple[np.ndarray, np.ndarray, SphericalHarmonicsBasis]:
    rng = np.random.default_rng(1)
    degree = 40
    lon = rng.uniform(-np.pi, np.pi, size=degree * 2)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, size=degree * 2)
    X = np.column_stack([lon, lat])
    basis = SphericalHarmonicsBasis(degree=degree, cup=False, include_bias=False)
    design = basis.fit_transform(X)
    coef = rng.normal(size=(design.shape[1], 1))
    intercept = rng.normal(size=1)
    y = design @ coef + intercept
    return X, y, basis


def test_spherical_fit_predict_single_target() -> None:
    X, y, basis = _make_spherical_data()
    pipe = Pipeline(
        [
            ("basis", basis),
            ("reg", LinearRegressionCond(fit_intercept=True)),
        ]
    )
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert np.allclose(pred, y)


def test_incompatible_feature_space() -> None:
    X, y, basis = _make_polynomial_data()
    pipe = Pipeline(
        [
            ("basis", basis),
            ("reg", LinearRegressionCond(fit_intercept=True)),
        ]
    )
    pipe.fit(X, y)
    with pytest.raises(ValueError):
        pipe.predict(np.ones((3, 3)))


def test_pipeline_compatibility() -> None:
    X, y, _ = _make_polynomial_data()
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("basis", PolynomialBasis(degree=1)),
            ("reg", LinearRegressionCond(fit_intercept=True)),
        ]
    )
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == y.shape


def test_grid_search_compatibility() -> None:
    X, y, _ = _make_polynomial_data()
    pipe = Pipeline(
        [
            ("basis", PolynomialBasis(degree=1)),
            ("reg", LinearRegressionCond()),
        ]
    )
    grid = {"basis__degree": [1, 2]}
    search = GridSearchCV(pipe, grid, cv=2)
    search.fit(X, y)
    assert hasattr(search, "best_estimator_")


def test_pipeline_caches_basis(tmp_path) -> None:
    X, y, _ = _make_polynomial_data()
    mem = Memory(location=tmp_path)
    basis = CallCountingBasis(degree=2)
    pipe = Pipeline(
        [
            ("basis", basis),
            ("reg", LinearRegressionCond(fit_intercept=True)),
        ],
        memory=mem,
    )
    CallCountingBasis.n_calls = 0
    pipe.fit(X, y)
    assert CallCountingBasis.n_calls == 1
    pipe.fit(X, y)
    assert CallCountingBasis.n_calls == 1


def test_sparse_coding_with_omp() -> None:
    rng = np.random.default_rng(2)
    lon = rng.uniform(-np.pi, np.pi, size=30)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, size=30)
    X = np.column_stack([lon, lat])
    basis_gen = SphericalHarmonicsBasis(degree=3, cup=False, include_bias=False)
    design = basis_gen.fit_transform(X)
    coef_true = np.zeros(design.shape[1])
    idx = rng.choice(design.shape[1], size=5, replace=False)
    coef_true[idx] = rng.normal(size=5)
    y = design @ coef_true

    pipe = Pipeline(
        [
            ("basis", SphericalHarmonicsBasis(degree=3, cup=False, include_bias=False)),
            (
                "reg",
                OrthogonalMatchingPursuit(n_nonzero_coefs=5, fit_intercept=False),
            ),
        ]
    )
    pipe.fit(X, y)
    omp = pipe.named_steps["reg"]
    assert np.allclose(omp.coef_, coef_true)


def test_polynomial_constant_feature() -> None:
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    basis = PolynomialBasis(degree=2)
    basis.fit(X)
    Xt = basis.transform(X)
    assert np.all(np.isfinite(Xt))
