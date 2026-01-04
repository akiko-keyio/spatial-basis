import numpy as np
from spatial_basis.basis import CoordsConverter


def test_transform_handles_pole_point() -> None:
    lon = np.array([10.0, 20.0, 30.0])
    lat = np.array([40.0, 50.0, 60.0])
    conv = CoordsConverter(pole="xyzmean", method="central")
    conv.fit(lon, lat)
    lon0 = np.degrees(conv.phi0)
    lat0 = 90 - np.degrees(conv.theta0)
    theta, phi = conv.transform(np.array([lon0]), np.array([lat0]))
    assert not np.isnan(phi).any()
    assert np.allclose(phi, 0.0)


def test_central_scale_consistent() -> None:
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, size=10)
    lat = rng.uniform(-90, 90, size=10)

    conv = CoordsConverter(pole="xyzmean", method="central_scale")
    conv.fit(lon, lat)
    scale_after_fit = conv.scale

    conv.transform(lon, lat)
    scale_after_first = conv.scale

    lon2 = rng.uniform(-180, 180, size=5)
    lat2 = rng.uniform(-90, 90, size=5)
    conv.transform(lon2, lat2)
    scale_after_second = conv.scale

    assert np.allclose(scale_after_fit, scale_after_first)
    assert np.allclose(scale_after_first, scale_after_second)

