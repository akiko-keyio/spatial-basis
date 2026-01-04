from numbers import Integral
from numpy.polynomial import Legendre, Chebyshev
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions, RealNotInt
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted
import numpy as np
from scipy.special import sph_harm_y
from sklearn.utils.validation import validate_data


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
    }

    def __init__(
        self,
        *,
        degree=2,
        include_bias=True,
        basis="polynomial",
    ) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.basis = basis

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

<<<<<<< HEAD:src/spatial_basis/basis.py
        self.min_vals_ = X.min(axis=0)
        self.max_vals_ = X.max(axis=0)
=======
        self.min_vals = X.min(axis=0)
        self.max_vals = X.max(axis=0)
        self.range_ = self.max_vals - self.min_vals
        self.range_[self.range_ == 0] = 1
>>>>>>> 6fc7b44f2e3820317a57cfc10058b498af663e10:src/geospectra/basis.py
        self.n_output_features_ = len(self.get_feature_names_out())

        return self

    def get_feature_names_out(self, input_features=None):
        input_features = _check_feature_names_in(self, input_features)
        feature_names = []
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                if i == 0 and j == 0:
                    if self.include_bias:
                        feature_names.append("1")
                    continue

                elif self.basis == "polynomial":
                    term1 = input_features[0] if i != 0 else ""
                    term1 += f"^{i}" if i > 1 else ""
                    term2 = input_features[1] if j != 0 else ""
                    term2 += f"^{j}" if j > 1 else ""
                    feature_names.append(f"{term1} {term2}".strip())

                else:
                    f = (
                        "T" if self.basis == "chebyshev" else "L"
                    )  # if self.basis == 'legendre'

                    term1 = (
                        f"{f}{i}({input_features[0]})"
                        if i > 1
                        else f"{input_features[0]}"
                        if i == 1
                        else ""
                    )
                    term2 = (
                        f"{f}{j}({input_features[1]})"
                        if j > 1
                        else f"{input_features[1]}"
                        if j == 1
                        else ""
                    )
                    feature_names.append(f"{term1} {term2}".strip())

        return np.asarray(feature_names, dtype=object)

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

<<<<<<< HEAD:src/spatial_basis/basis.py
        X = 2 * (X - self.min_vals_) / (self.max_vals_ - self.min_vals_) - 1
=======
        X = 2 * (X - self.min_vals) / self.range_ - 1
>>>>>>> 6fc7b44f2e3820317a57cfc10058b498af663e10:src/geospectra/basis.py

        X1, X2 = X[:, 0], X[:, 1]

        X_transform = np.ones((len(X), self.n_output_features_))
        index = 0
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                if i == 0 and j == 0:
                    if self.include_bias:
                        X_transform[:, index] = 1
                        index += 1
                    continue
                if self.basis == "polynomial":
                    X_transform[:, index] = (X1**i) * (X2**j)
                elif self.basis == "legendre":
                    X_transform[:, index] = Legendre.basis(i)(X1) * Legendre.basis(j)(
                        X2
                    )
                else:  # self.basis=='chebyshev':
                    X_transform[:, index] = Chebyshev.basis(i)(X1) * Chebyshev.basis(j)(
                        X2
                    )

                index += 1
        
        if index != self.n_output_features_:
            raise ValueError(
                f"Mismatch in output features: expected {self.n_output_features_}, got {index}"
            )

        return X_transform


class SphericalHarmonicsBasis(TransformerMixin, BaseEstimator):
    """Transformer for generating spherical harmonics features.

    Parameters
    ----------
    pole : str, default='xyzmean'
        Specifies the pole for coordinate conversion.

    cup : bool, default=True
        If `True`, use the cup design matrix; otherwise, use the full design matrix.

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
        "force_norm": ["boolean"],
        "coords_convert_method": [
            StrOptions({"central_scale", "central", "basic", "non"})
        ],
    }

    def __init__(
        self,
        *,
        degree=2,
        pole="xyzmean",
        cup=True,
        include_bias=True,
        hemisphere_scale="auto",
        force_norm=False,
        coords_convert_method="central_scale",
    ) -> None:
        self.degree = degree
        self.pole = pole
        self.cup = cup
        self.include_bias = include_bias
        self.hemisphere_scale = hemisphere_scale
        self.force_norm = force_norm
        self.coords_convert_method = coords_convert_method

    def __sklearn_tags__(self):
        """Override sklearn tags to indicate this transformer requires 2D input with exactly 2 features."""
        tags = super().__sklearn_tags__()
        # Mark that we need special input validation (lon, lat only)
        tags.input_tags.pairwise = True  # Input is pairwise (lon, lat coordinates)
        tags._skip_test = True  # Skip standard check_estimator tests due to 2-feature requirement
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
        self.n_output_features_ = len(self.get_feature_names_out())
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
        
        # Pre-compute terms_ during fit for sklearn compliance
        self.terms_ = []
        for order in range(self.degree + 1):
            if order == 0 and not self.include_bias:
                continue
            for m in range(-order, order + 1):
                if self.cup and (m - order) % 2 == 1:
                    continue
                self.terms_.append(f"Y{order}{m}")

        return self

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

        n_samples = len(phi)
        X = np.zeros((n_samples, self.n_output_features_))
        index = 0

        # https://www.wikiwand.com/en/articles/Spherical_harmonics
        # https://www.wikiwand.com/en/articles/Spherical_harmonics#Real_form
        for order in range(self.degree + 1):
            if order == 0 and not self.include_bias:
                continue
            for m in range(-order, order + 1):
                if self.cup and (m - order) % 2 == 1:
                    continue
                Y_l_m_abs = sph_harm_y(order, abs(m), theta, phi)
                if m < 0:
                    X[:, index] = np.sqrt(2) * (-1) ** m * Y_l_m_abs.imag
                elif m == 0:
                    X[:, index] = Y_l_m_abs.real
                else:
                    X[:, index] = np.sqrt(2) * (-1) ** m * Y_l_m_abs.real
                index = index + 1

        #
        if self.cup:
            X = X * np.sqrt(2)

        if self.force_norm:
            X = X * X.shape[0] / np.linalg.norm(X, axis=0)

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
        input_features = _check_feature_names_in(self, input_features)
        feature_names = []
        for order in range(self.degree + 1):
            if order == 0 and not self.include_bias:
                continue
            for m in range(-order, order + 1):
                if self.cup and (m - order) % 2 == 1:
                    continue
                feature_names.append(f"Y:{order},{m}")

        return np.asarray(feature_names, dtype=object)


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
