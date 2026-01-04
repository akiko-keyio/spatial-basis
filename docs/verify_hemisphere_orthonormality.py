
import numpy as np
from spatial_basis.basis import SphericalHarmonicsBasis

def verify_orthonormality():
    # 1. Define a dense grid on the Northern Hemisphere
    # Latitude: 0 to 90 degrees (Equator to North Pole)
    # Longitude: -180 to 180
    n_lat = 200
    n_lon = 400
    
    # Use midpoint rule to avoid singularity at pole and boundary issues
    dlat = 90.0 / n_lat
    dlon = 360.0 / n_lon
    
    lat_centers = np.linspace(0 + dlat/2, 90 - dlat/2, n_lat)
    lon_centers = np.linspace(-180 + dlon/2, 180 - dlon/2, n_lon)
    
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    
    X = np.column_stack([lon_flat, lat_flat])
    
    # 2. Initialize Basis (Cup=True for Hemisphere)
    # We use coords_convert_method='non' or 'basic' because we are providing exactly the hemisphere
    # and we want to verify the theoretical orthogonality of the basis functions themselves
    # on this domain.
    # 'xyzmean' pole might default to centering on the data, which is fine if data is uniform hemisphere.
    # But let's be explicit: pole at North Pole (90, 0)
    
    # Note: explicit pole setup
    # pole=(90, 0) means theta is measured from North Pole.
    # coords_convert_method='basic' converts lon->phi, lat->theta (90-lat).
    
    basis = SphericalHarmonicsBasis(
        degree=5, 
        cup=True, 
        include_bias=True, 
        pole="xyzmean", # Use xyzmean which should find North Pole for uniform hemisphere
        coords_convert_method='basic',
        force_norm=False # We want to check theoretical normalization, not forced discrete normalization
    )
    
    # 3. Transform
    # We fit on the data (which is the full hemisphere)
    basis.fit(X)
    features = basis.transform(X)
    
    print(f"Number of features: {features.shape[1]}")
    print(f"Feature names: {basis.get_feature_names_out()}")

    # 4. Compute Integrals (Inner Products)
    # Integration element dA = cos(lat) * dlat * dlon (in radians)
    # R=1 assumed for normalization check.
    
    weights = np.cos(np.radians(lat_flat)) * np.radians(dlat) * np.radians(dlon)
    
    # weighted inner product: F.T @ W @ F
    # We can multiply each row of features by sqrt(weights) then compute dot product
    
    features_weighted = features * np.sqrt(weights)[:, np.newaxis]
    gram_matrix = features_weighted.T @ features_weighted
    
    # 5. Check Diagonal (Norms) and Off-Diagonal (Orthogonality)
    print("\nGram Matrix Diagonal (Norms^2):")
    print(np.diag(gram_matrix))
    
    # Expected normalization:
    # Standard spherical harmonics are normalized to 1 on the *sphere* (4pi).
    # Since we are on hemisphere (2pi) and keeping only symmetric terms,
    # the integral should be 0.5 if they are normalized on sphere.
    # However, the code does: if self.cup: X = X * np.sqrt(2) (line 358)
    # So they should be normalized to 1 on the hemisphere!
    
    is_orthonormal = np.allclose(gram_matrix, np.eye(len(gram_matrix)), atol=1e-2)
    print(f"\nIs Orthonormal? {is_orthonormal}")
    
    # Print max off-diagonal error
    off_diag = gram_matrix.copy()
    np.fill_diagonal(off_diag, 0)
    print(f"Max off-diagonal: {np.max(np.abs(off_diag))}")
    print(f"Max diagonal error: {np.max(np.abs(np.diag(gram_matrix) - 1.0))}")

    if not is_orthonormal:
        print("WARNING: Not orthonormal within tolerance (1e-2).")
        # detailed debug
        # print(gram_matrix)
    else:
        print("SUCCESS: Features are orthonormal on the hemisphere.")

if __name__ == "__main__":
    verify_orthonormality()
