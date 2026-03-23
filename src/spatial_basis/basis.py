from numbers import Integral
from numpy.polynomial import Legendre, Chebyshev
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions, RealNotInt
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted
import numpy as np
from scipy.special import sph_harm_y
from sklearn.utils.validation import validate_data


def _enumerate_polynomial_powers(degree: int, include_bias: bool) -> np.ndarray:
    powers: list[tuple[int, int]] = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            if i == 0 and j == 0 and not include_bias:
                continue
            powers.append((i, j))
    return np.asarray(powers, dtype=int).reshape(-1, 2)


def _format_polynomial_term(
    powers: np.ndarray, input_features: np.ndarray, basis: str
) -> str:
    i, j = map(int, powers)
    if i == 0 and j == 0:
        return "1"

    if basis == "polynomial":
        term1 = input_features[0] if i != 0 else ""
        term1 += f"^{i}" if i > 1 else ""
        term2 = input_features[1] if j != 0 else ""
        term2 += f"^{j}" if j > 1 else ""
        return f"{term1} {term2}".strip()

    prefix = "T" if basis == "chebyshev" else "L"
    term1 = (
        f"{prefix}{i}({input_features[0]})"
        if i > 1
        else f"{input_features[0]}"
        if i == 1
        else ""
    )
    term2 = (
        f"{prefix}{j}({input_features[1]})"
        if j > 1
        else f"{input_features[1]}"
        if j == 1
        else ""
    )
    return f"{term1} {term2}".strip()


def _evaluate_polynomial_term(
    powers: np.ndarray, basis: str, x0: np.ndarray, x1: np.ndarray
) -> np.ndarray:
    i, j = map(int, powers)
    if basis == "polynomial":
        return (x0**i) * (x1**j)

    basis_cls = Legendre if basis == "legendre" else Chebyshev
    return basis_cls.basis(i)(x0) * basis_cls.basis(j)(x1)


def _enumerate_harmonic_indices(
    degree: int, include_bias: bool, cup: bool
) -> np.ndarray:
    indices: list[tuple[int, int]] = []
    for order in range(degree + 1):
        if order == 0 and not include_bias:
            continue
        for m in range(-order, order + 1):
            if cup and (m - order) % 2 == 1:
                continue
            indices.append((order, m))
    return np.asarray(indices, dtype=int).reshape(-1, 2)


def _format_harmonic_term(indices: np.ndarray) -> str:
    order, m = map(int, indices)
    return f"Y:{order},{m}"


def _evaluate_real_spherical_harmonic(
    indices: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    order, m = map(int, indices)
    y_l_m_abs = sph_harm_y(order, abs(m), theta, phi)
    if m < 0:
        return np.sqrt(2) * (-1) ** m * y_l_m_abs.imag
    if m == 0:
        return y_l_m_abs.real
    return np.sqrt(2) * (-1) ** m * y_l_m_abs.real


def _precompute_alf_coefficients(
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute recursion coefficients for fully normalized ALFs."""
    alpha = np.zeros((degree + 1, degree + 1), dtype=float)
    beta = np.zeros((degree + 1, degree + 1), dtype=float)

    for order in range(2, degree + 1):
        m = np.arange(order, dtype=float)
        order_sq_minus_m_sq = order * order - m * m
        alpha[order, :order] = np.sqrt(
            (4.0 * order * order - 1.0) / order_sq_minus_m_sq
        )
        beta[order, :order] = np.sqrt(
            (2.0 * order + 1.0) * (((order - 1.0) ** 2) - m * m)
            / ((2.0 * order - 3.0) * order_sq_minus_m_sq)
        )

    m_vals = np.arange(1, degree + 1, dtype=float)
    sectoral = np.empty(degree + 1, dtype=float)
    sectoral[0] = 0.0
    sectoral[1:] = np.sqrt((2.0 * m_vals + 1.0) / (2.0 * m_vals))

    sub_sectoral = np.sqrt(2.0 * np.arange(degree, dtype=float) + 3.0)
    return alpha, beta, sectoral, sub_sectoral


def _build_harmonic_column_map(
    harmonic_indices: np.ndarray,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Map ``(order, |m|)`` to output columns and signed orders."""
    column_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index, (order, m) in enumerate(harmonic_indices):
        key = (int(order), int(abs(m)))
        column_map.setdefault(key, []).append((index, int(m)))
    return column_map


def _assign_real_harmonic_columns(
    X: np.ndarray,
    column_map: dict[tuple[int, int], list[tuple[int, int]]],
    order: int,
    abs_m: int,
    plm: np.ndarray,
    cos_mphi: dict[int, np.ndarray],
    sin_mphi: dict[int, np.ndarray],
) -> None:
    """Write all real harmonics for a given ``(order, |m|)``."""
    entries = column_map.get((order, abs_m))
    if entries is None:
        return

    sqrt2 = np.sqrt(2.0)
    for column_index, signed_m in entries:
        if signed_m == 0:
            X[:, column_index] = plm
        elif signed_m > 0:
            X[:, column_index] = sqrt2 * plm * cos_mphi[abs_m]
        else:
            X[:, column_index] = sqrt2 * plm * sin_mphi[abs_m]


def _build_design_matrix_via_alf_recursion(
    harmonic_indices: np.ndarray,
    degree: int,
    theta: np.ndarray,
    phi: np.ndarray,
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    column_map: dict[tuple[int, int], list[tuple[int, int]]],
    abs_orders: tuple[int, ...],
) -> np.ndarray:
    """Build the real spherical harmonics design matrix via stable ALF recursion."""
    alpha, beta, sectoral, sub_sectoral = coefficients
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_mphi = {abs_m: np.cos(abs_m * phi) for abs_m in abs_orders}
    sin_mphi = {abs_m: np.sin(abs_m * phi) for abs_m in abs_orders}

    X = np.empty((len(theta), len(harmonic_indices)), dtype=float)
    p_mm = np.full(len(theta), 1.0 / np.sqrt(4.0 * np.pi), dtype=float)

    for abs_m in range(degree + 1):
        if abs_m > 0:
            p_mm = sectoral[abs_m] * sin_theta * p_mm

        p_prev2 = p_mm
        _assign_real_harmonic_columns(
            X, column_map, abs_m, abs_m, p_prev2, cos_mphi, sin_mphi
        )
        if abs_m == degree:
            continue

        p_prev1 = sub_sectoral[abs_m] * cos_theta * p_prev2
        _assign_real_harmonic_columns(
            X, column_map, abs_m + 1, abs_m, p_prev1, cos_mphi, sin_mphi
        )

        for order in range(abs_m + 2, degree + 1):
            p_current = (
                alpha[order, abs_m] * cos_theta * p_prev1
                - beta[order, abs_m] * p_prev2
            )
            _assign_real_harmonic_columns(
                X, column_map, order, abs_m, p_current, cos_mphi, sin_mphi
            )
            p_prev2 = p_prev1
            p_prev1 = p_current

    return X


class PolynomialBasis(TransformerMixin, BaseEstimator):
    """An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    degree : int, default=2
        Specifies the maximal degree of the polynomial features.

    include_bias : bool, default=True
        If `True` (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).

    basis : {"normal", "chebyshev", "legendre"}, default="normal


    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_output_features_ : int
        The number of output features
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "degree": [Interval(Integral, 0, None, closed="left")],
        "include_bias": ["boolean"],
        "basis": [StrOptions({"polynomial", "chebyshev", "legendre"})],
        "normalize": ["boolean"],
    }

    def __init__(
        self,
        *,
        degree=2,
        include_bias=True,
        basis="polynomial",
        normalize=False,
    ) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.basis = basis
        self.normalize = normalize

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = validate_data(self, X, accept_sparse=True)
        _, n_features = X.shape

        if n_features != 2:
            raise ValueError(
                f"This transformer only works with 2 features, but got {n_features}."
            )

        if isinstance(self.degree, Integral):
            if self.degree == 0 and not self.include_bias:
                raise ValueError(
                    "Setting degree to zero and include_bias to False would result in"
                    "an empty output array."
                )
        else:
            raise ValueError(f"degree must be a non-negative int, got {self.degree}.")

        self.min_vals_ = X.min(axis=0)
        self.max_vals_ = X.max(axis=0)
        self.range_ = self.max_vals_ - self.min_vals_
        self.range_[self.range_ == 0] = 1  # Avoid division by zero
        self.powers_ = _enumerate_polynomial_powers(self.degree, self.include_bias)
        self.feature_degrees_ = self.powers_.sum(axis=1)
        self.n_output_features_ = self.powers_.shape[0]

        if self.normalize:
            X_scaled = 2 * (X - self.min_vals_) / self.range_ - 1
            X_basis = self._build_design_matrix(X_scaled)
            self.column_norms_ = self._compute_column_norms(X_basis)
            self.column_normalizers_ = self._compute_column_normalizers(
                self.column_norms_,
                X_basis.shape[0],
            )
        elif hasattr(self, "column_norms_"):
            delattr(self, "column_norms_")
            if hasattr(self, "column_normalizers_"):
                delattr(self, "column_normalizers_")

        return self

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "powers_")
        input_features = _check_feature_names_in(self, input_features)
        return np.asarray(
            [
                _format_polynomial_term(powers, input_features, self.basis)
                for powers in self.powers_
            ],
            dtype=object,
        )

    def transform(self, X):
        """A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        check_is_fitted(self)

        X = validate_data(self, X, accept_sparse=True, reset=False)

        X_scaled = 2 * (X - self.min_vals_) / self.range_ - 1
        X_transform = self._build_design_matrix(X_scaled)

        if self.normalize:
            X_transform = X_transform / self.column_normalizers_

        return X_transform

    def _build_design_matrix(self, X_scaled):
        """Evaluate the unnormalized polynomial design matrix."""
        X1, X2 = X_scaled[:, 0], X_scaled[:, 1]
        X_transform = np.empty((len(X_scaled), self.n_output_features_))
        for index, powers in enumerate(self.powers_):
            X_transform[:, index] = _evaluate_polynomial_term(
                powers, self.basis, X1, X2
            )
        return X_transform

    @staticmethod
    def _compute_column_norms(X):
        """Compute per-column L2 norms from the training design matrix."""
        norms = np.linalg.norm(X, axis=0)
        norms[norms == 0] = 1.0
        return norms

    @staticmethod
    def _compute_column_normalizers(column_norms, n_samples):
        """Compute normalizers so that diag((1/m) * Y^T Y) equals one."""
        return column_norms / np.sqrt(float(n_samples))


class SphericalHarmonicsBasis(TransformerMixin, BaseEstimator):
    """Transformer for generating spherical harmonics features.

    Parameters
    ----------
    pole : str, default='xyzmean'
        Specifies the pole for coordinate conversion.

    cup : bool, default=True
        If `True`, use the cup design matrix; otherwise, use the full design matrix.

    implementation : {"recursion", "scipy"}, default="recursion"
        Backend used to evaluate the design matrix. ``"recursion"`` uses a
        stable associated-Legendre recursion and is faster for moderate to high
        degrees, while ``"scipy"`` preserves the direct ``scipy.special``
        evaluation path.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_output_features_ : int
        The number of output features

    """

    _parameter_constraints = {
        "degree": [Interval(Integral, 0, None, closed="left")],
        "pole": [StrOptions({"haversine", "xyzmean"}), tuple],
        "cup": ["boolean"],
        "include_bias": ["boolean"],
        "hemisphere_scale": [
            StrOptions({"auto"}),
            Interval(RealNotInt, 0, 1, closed="right"),
        ],
        "normalize": ["boolean"],
        "coords_convert_method": [
            StrOptions({"central_scale", "central", "basic", "non"})
        ],
        "implementation": [StrOptions({"recursion", "scipy"})],
    }

    def __init__(
        self,
        *,
        degree=2,
        pole="xyzmean",
        cup=True,
        include_bias=True,
        hemisphere_scale="auto",
        normalize=False,
        coords_convert_method="central_scale",
        implementation="recursion",
    ) -> None:
        self.degree = degree
        self.pole = pole
        self.cup = cup
        self.include_bias = include_bias
        self.hemisphere_scale = hemisphere_scale
        self.normalize = normalize
        self.coords_convert_method = coords_convert_method
        self.implementation = implementation

    def __sklearn_tags__(self):
        """Override sklearn tags to indicate this transformer requires 2D input with exactly 2 features."""
        tags = super().__sklearn_tags__()
        # This transformer requires exactly 2 features (lon, lat)
        tags.input_tags.two_d_array = True
        tags.input_tags.allow_nan = False
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the coordinate converter based on input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            The input samples with longitude and latitude.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """

        X = validate_data(self, X, accept_sparse=True)
        _, n_features = X.shape

        if n_features != 2:
            raise ValueError(
                f"This transformer only works with 2 features, but got {n_features}."
            )

        if isinstance(self.degree, Integral):
            if self.degree == 0 and not self.include_bias:
                raise ValueError(
                    "Setting degree to zero and include_bias to False would result in"
                    "an empty output array."
                )
        else:
            raise ValueError(f"degree must be a non-negative int, got {self.degree}.")

        self.min_vals_ = X.min(axis=0)
        self.max_vals_ = X.max(axis=0)
        self.harmonic_indices_ = _enumerate_harmonic_indices(
            self.degree, self.include_bias, self.cup
        )
        self.feature_degrees_ = self.harmonic_indices_[:, 0]
        self.n_output_features_ = self.harmonic_indices_.shape[0]
        empirical_n_output_features_ = (
            (self.degree + 1) * (self.degree + 2) // 2
            if self.cup
            else (self.degree + 1) ** 2
        )
        if not self.include_bias:
            empirical_n_output_features_ -= 1

        if self.n_output_features_ != empirical_n_output_features_:
            raise ValueError(
                f"Mismatch in output features: expected {empirical_n_output_features_}, "
                f"got {self.n_output_features_}"
            )

        if self.hemisphere_scale == "auto":
            hemisphere_scale = 0.5 if self.cup else 1
        else:
            hemisphere_scale = self.hemisphere_scale
        self.coords_converter_ = CoordsConverter(
            pole=self.pole,
            hemisphere_scale=hemisphere_scale,
            method=self.coords_convert_method,
        )

        lon, lat = X[:, 0], X[:, 1]
        self.coords_converter_.fit(lon, lat)

        if self.implementation == "recursion":
            self.alf_coefficients_ = _precompute_alf_coefficients(self.degree)
            self.harmonic_column_map_ = _build_harmonic_column_map(
                self.harmonic_indices_
            )
            self.harmonic_abs_orders_ = tuple(
                sorted({int(abs(m)) for _, m in self.harmonic_indices_ if m != 0})
            )
        elif hasattr(self, "alf_coefficients_"):
            delattr(self, "alf_coefficients_")
            delattr(self, "harmonic_column_map_")
            delattr(self, "harmonic_abs_orders_")

        if self.normalize:
            theta, phi = self.coords_converter_.transform(lon, lat)
            X_basis = self._build_design_matrix(theta, phi)
            self.column_norms_ = self._compute_column_norms(X_basis)
            self.column_normalizers_ = self._compute_column_normalizers(
                self.column_norms_,
                X_basis.shape[0],
            )
        elif hasattr(self, "column_norms_"):
            delattr(self, "column_norms_")
            if hasattr(self, "column_normalizers_"):
                delattr(self, "column_normalizers_")

        # Expose key fitted parameters as top-level attributes for sklearn compliance
        self.pole_ = (
            90 - np.degrees(self.coords_converter_.theta0),  # lat
            np.degrees(self.coords_converter_.phi0)  # lon
        )
        self.scale_ = self.coords_converter_.scale

        # Keep legacy string metadata, but derive it from numeric output indices.
        self.terms_ = [
            f"Y{order}{m}" for order, m in self.harmonic_indices_.tolist()
        ]

        return self

    def _build_design_matrix(self, theta, phi):
        """Evaluate the unnormalized spherical harmonics design matrix."""
        if self.implementation == "recursion":
            X = _build_design_matrix_via_alf_recursion(
                self.harmonic_indices_,
                self.degree,
                theta,
                phi,
                self.alf_coefficients_,
                self.harmonic_column_map_,
                self.harmonic_abs_orders_,
            )
        else:
            n_samples = len(phi)
            X = np.empty((n_samples, self.n_output_features_), dtype=float)

            for index, harmonic_index in enumerate(self.harmonic_indices_):
                X[:, index] = _evaluate_real_spherical_harmonic(
                    harmonic_index, theta, phi
                )

        if self.cup:
            X = X * np.sqrt(2)

        return X

    @staticmethod
    def _compute_column_norms(X):
        """Compute per-column L2 norms from the training design matrix."""
        norms = np.linalg.norm(X, axis=0)
        norms[norms == 0] = 1.0
        return norms

    @staticmethod
    def _compute_column_normalizers(column_norms, n_samples):
        """Compute normalizers so that diag((1/m) * Y^T Y) equals one."""
        return column_norms / np.sqrt(float(n_samples))

    def transform(self, X):
        """Transform the input data to spherical harmonics features.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            The input samples with longitude and latitude.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_output_features_)
            The transformed feature matrix.
        """
        check_is_fitted(self, "coords_converter_")

        X = validate_data(self, X, accept_sparse=True, reset=False)
        lon, lat = X[:, 0], X[:, 1]
        theta, phi = self.coords_converter_.transform(lon, lat)
        X = self._build_design_matrix(theta, phi)

        if self.normalize:
            X = X / self.column_normalizers_

        return X

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str, optional
            Input feature names.

        Returns
        -------
        feature_names : ndarray of str
            Transformed feature names.
        """
        check_is_fitted(self, "harmonic_indices_")
        _check_feature_names_in(self, input_features)
        return np.asarray(
            [_format_harmonic_term(indices) for indices in self.harmonic_indices_],
            dtype=object,
        )


class CoordsConverter:
    def __init__(self, pole="haversine", method="central_scale", hemisphere_scale=1.0):
        """
        Initialize the CoordsConverter class.

        :param pole: Tuple of (latitude, longitude) or 'calculate' to determine the pole automatically.
        :param method: The method to use for conversion ('non', 'basic', 'central', 'central_scale').
        :param hemisphere_scale: Scale factor for the hemisphere.
        """
        self.pole = pole
        self.method = method
        self.hemisphere_scale = hemisphere_scale
        self.scale = None
        self.theta0 = None
        self.phi0 = None

    def fit(self, lon, lat):
        if self.pole == "haversine":
            lon0, lat0 = self._calculate_central_haversine(lon, lat)
            self.theta0 = np.radians(90 - lat0) + 1.0e-6
            self.phi0 = np.radians(lon0)
        elif self.pole == "xyzmean":
            lon0, lat0 = self._calculate_central_xyzmean(lon, lat)
            self.theta0 = np.radians(90 - lat0)
            self.phi0 = np.radians(lon0)
        elif (
            isinstance(self.pole, tuple)
            and len(self.pole) == 2
            and all(isinstance(item, float) for item in self.pole)
        ):
            self.theta0 = np.radians(90 - self.pole[0])
            self.phi0 = np.radians(self.pole[1])
        else:
            raise ValueError(
                "Pole must be a tuple of two floats or ['haversine','xyzmean']."
            )

        if self.method == "central_scale":
            theta1, _ = self._central(lon, lat)
            self.scale = np.pi * self.hemisphere_scale / theta1.max()

        return self

    def transform(self, lon, lat):
        if self.theta0 is None or self.phi0 is None:
            raise AttributeError(
                "theta0 or phi0 is None, please call the fit method first"
            )
        if self.method == "non":
            return self._non(lon, lat)
        elif self.method == "basic":
            return self._basic(lon, lat)
        elif self.method == "central":
            return self._central(lon, lat)
        elif self.method == "central_scale":
            return self._central_scale(lon, lat)
        else:
            raise ValueError(
                'Method must be one of "non", "basic", "central", or "central_scale".'
            )

    def fit_transform(self, lon, lat):
        """
        Convert the given longitude and latitude using the specified method.

        :param lon: Array of longitudes.
        :param lat: Array of latitudes.
        :return: Converted coordinates.
        """
        self.fit(lon, lat)
        return self.transform(lon, lat)

    def plot_convert(self, lon, lat, color=None, cmid=0, **kwargs):
        """
        Plot the converted coordinates using Plotly.

        :param lon: Array of longitudes.
        :param lat: Array of latitudes.
        :param color: Color for the plot.
        :param cmid: Center of the color scale.
        :param kwargs: Additional keyword arguments for layout.
        :return: Plotly Figure object.
        """
        import plotly.graph_objs as go
        import plotly.express as px

        phi, theta = self.transform(np.asarray(lon), np.asarray(lat))
        phi = np.degrees(phi)
        theta = np.degrees(theta)
        return go.Figure(
            go.Scatterpolar(
                r=phi,
                theta=theta,
                mode="markers",
                hovertext=color,
                marker=dict(
                    size=4,
                    color=color,
                    cmid=cmid,
                    colorscale=px.colors.sequential.RdBu,
                    showscale=True,
                ),
            )
        ).update_layout(**kwargs)

    def _calculate_central_haversine(self, lon, lat):
        """
        Calculate the central point using the haversine formula.

        :param lon: Array of longitudes.
        :param lat: Array of latitudes.
        :return: Tuple of (central longitude, central latitude).
        """

        def haversine_np(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = (
                np.sin(dlat / 2.0) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
            )
            c = 2 * np.arcsin(np.sqrt(a))
            return c

        dist_matrix = haversine_np(lon[:, None], lat[:, None], lon, lat)
        max_dists = dist_matrix.max(axis=1)
        centroid_idx = np.argmin(max_dists)
        return lon[centroid_idx], lat[centroid_idx]

    def _calculate_central_xyzmean(self, lon, lat):
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)

        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        z_mean = np.mean(z)

        central_lon = np.arctan2(y_mean, x_mean)
        central_lat = np.arctan2(z_mean, np.sqrt(x_mean**2 + y_mean**2))

        central_lon = np.degrees(central_lon)
        central_lat = np.degrees(central_lat)

        return central_lon, central_lat

    def _non(self, lon, lat):
        phi = np.radians(lon)  # Longitude
        theta = np.radians(lat)  # Latitude to colatitude
        return theta, phi

    def _basic(self, lon, lat):
        phi = np.radians(lon)  # Longitude
        theta = np.radians(90 - lat)  # Latitude to colatitude
        return theta, phi

    def _central(self, lon, lat):
        # Reference：
        # GNSS_MODIS_ER...据融合的球冠谐水汽模型构建_樊鉴庆.pdf p55
        # 球冠谐方法精化区域大地水准面_储王宁.pdf p46
        # 基于球冠谐分析的区域精密对流层建模
        theta = np.radians(90 - lat)  # Colatitude
        phi = np.radians(lon)  # Longitude

        # Convert to colatitude with the pole as the center
        theta1 = np.arccos(
            np.cos(self.theta0) * np.cos(theta)
            + np.sin(self.theta0) * np.sin(theta) * np.cos(self.phi0 - phi)
        )

        # Convert to longitude with the pole as the center. When ``theta1`` is
        # zero, the longitude becomes undefined. In this degenerate case we
        # explicitly set ``phi1`` to zero to avoid NaNs.
        sin_theta1 = np.sin(theta1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sin_phi1 = np.where(
                np.isclose(sin_theta1, 0),
                0.0,
                np.sin(theta) * np.sin(phi - self.phi0) / sin_theta1,
            )
            cos_phi1 = np.where(
                np.isclose(sin_theta1, 0),
                1.0,
                (
                    np.sin(self.theta0) * np.cos(theta)
                    - np.cos(self.theta0) * np.sin(theta) * np.cos(phi - self.phi0)
                )
                / sin_theta1,
            )
        phi1 = np.arctan2(sin_phi1, cos_phi1)

        return theta1, phi1

    def _central_scale(self, lon, lat):
        if self.scale is None:
            raise AttributeError("scale is None, please call the fit method first")
        theta1, phi1 = self._central(lon, lat)
        theta_scale = theta1 * self.scale
        return theta_scale, phi1
