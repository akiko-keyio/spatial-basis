"""
Generate visualizations for Spatial Basis library:
1. Spherical Harmonics (SH) pyramid with radial modulation (r = |Y|)
2. Hemispherical Harmonics (HSH) pyramid - surface view
3. HSH orthogonality verification (3D bar plot)
4. Coordinate conversion methods comparison

Output: docs/assets/

Uses SphericalHarmonicsBasis from basis.py for consistency.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

from spatial_basis import SphericalHarmonicsBasis

# Output directory (same folder as this script)
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure matplotlib
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9


def get_basis_values_on_grid(degree, cup, lon_grid, lat_grid):
    """
    Get all basis function values on a lon/lat grid using SphericalHarmonicsBasis.
    
    Returns:
        features: array of shape (n_points, n_features) 
        feature_names: list of feature names like 'Y:0,0', 'Y:1,-1', etc.
    """
    X = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    
    basis = SphericalHarmonicsBasis(
        degree=degree,
        cup=cup,
        include_bias=True,
        coords_convert_method='basic'  # Standard mapping for visualization
    )
    basis.fit(X)
    features = basis.transform(X)
    feature_names = basis.get_feature_names_out()
    
    return features, feature_names, lon_grid.shape


def parse_feature_name(name):
    """Parse 'Y:l,m' to (l, m) tuple."""
    parts = name.split(':')[1].split(',')
    return int(parts[0]), int(parts[1])


def plot_sh_pyramid(max_degree=3, savepath=None):
    """Plot Spherical Harmonics (SH) in pyramid with radial modulation."""
    if savepath is None:
        savepath = OUTPUT_DIR / 'sh_pyramid.png'
    
    n_theta, n_phi = 80, 160
    lon = np.linspace(-180, 180, n_phi)
    lat = np.linspace(-90, 90, n_theta)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Get basis values using SphericalHarmonicsBasis
    features, names, grid_shape = get_basis_values_on_grid(
        max_degree, cup=False, lon_grid=lon_grid, lat_grid=lat_grid
    )
    
    # Spherical coordinates for 3D plot
    phi_rad = np.radians(lon_grid)
    theta_rad = np.radians(90 - lat_grid)
    
    n_rows = max_degree + 1
    max_cols = 2 * max_degree + 1
    
    # First pass: compute global max range across all terms
    global_max = 0
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            target_name = f'Y:{l},{m}'
            try:
                idx = np.where(names == target_name)[0][0]
            except IndexError:
                continue
            Y = features[:, idx].reshape(grid_shape)
            r = np.abs(Y)
            x = r * np.sin(theta_rad) * np.cos(phi_rad)
            y = r * np.sin(theta_rad) * np.sin(phi_rad)
            z = r * np.cos(theta_rad)
            local_max = max(np.abs(x).max(), np.abs(y).max(), np.abs(z).max())
            global_max = max(global_max, local_max)
    
    global_max *= 1.05  # Add 5% margin
    
    cell_size = 1.8
    fig = plt.figure(figsize=(max_cols * cell_size, n_rows * cell_size))
    fig.suptitle('Spherical Harmonics (SH)', fontsize=12, fontweight='bold', y=0.98)
    
    for l in range(max_degree + 1):
        start_col = max_degree - l
        
        for i, m in enumerate(range(-l, l + 1)):
            col = start_col + i
            subplot_idx = l * max_cols + col + 1
            
            target_name = f'Y:{l},{m}'
            try:
                idx = np.where(names == target_name)[0][0]
            except IndexError:
                continue
            
            Y = features[:, idx].reshape(grid_shape)
            
            r = np.abs(Y)
            x = r * np.sin(theta_rad) * np.cos(phi_rad)
            y = r * np.sin(theta_rad) * np.sin(phi_rad)
            z = r * np.cos(theta_rad)
            
            ax = fig.add_subplot(n_rows, max_cols, subplot_idx, projection='3d')
            
            vmax = np.max(np.abs(Y))
            norm = plt.Normalize(-vmax, vmax) if vmax > 0 else plt.Normalize(-1, 1)
            
            ax.plot_surface(x, y, z, facecolors=cm.RdBu(norm(Y)),
                           rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
            
            ax.set_title(f'$Y_{{{l}}}^{{{m}}}$', fontsize=9, pad=1)
            ax.set_axis_off()
            
            # Use global max range for uniform scale
            ax.set_xlim([-global_max, global_max])
            ax.set_ylim([-global_max, global_max])
            ax.set_zlim([-global_max, global_max])
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=20, azim=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.5, w_pad=0.3)
    plt.savefig(savepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {savepath}")
    plt.close()


def get_hsh_terms(max_degree):
    """Get HSH terms where (l + m) is even."""
    all_terms = []
    for l in range(max_degree + 1):
        row_terms = []
        for m in range(-l, l + 1):
            if (l + m) % 2 == 0:
                row_terms.append((l, m))
        all_terms.append(row_terms)
    return all_terms


def plot_hsh_pyramid(max_degree=4, savepath=None):
    """
    Plot HSH in same pyramid layout as SH.
    Uses 2l+1 columns per row, only plots terms where l+m is even.
    """
    if savepath is None:
        savepath = OUTPUT_DIR / 'hsh_pyramid.png'
    
    n_theta, n_phi = 60, 120
    lon = np.linspace(-180, 180, n_phi)
    lat = np.linspace(0, 90, n_theta)  # Hemisphere
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Get basis values using SphericalHarmonicsBasis
    features, names, grid_shape = get_basis_values_on_grid(
        max_degree, cup=True, lon_grid=lon_grid, lat_grid=lat_grid
    )
    
    # Hemisphere surface coordinates - enlarged to fill subplot
    phi_rad = np.radians(lon_grid)
    theta_rad = np.radians(90 - lat_grid)
    scale = 2.0  # Larger scale for bigger hemispheres
    x_sphere = scale * np.sin(theta_rad) * np.cos(phi_rad)
    y_sphere = scale * np.sin(theta_rad) * np.sin(phi_rad)
    z_sphere = scale * np.cos(theta_rad)
    
    n_rows = max_degree + 1
    max_cols = 2 * max_degree + 1  # Same as SH: 2l+1 columns
    
    # Same layout as SH pyramid
    cell_size = 1.8
    fig = plt.figure(figsize=(max_cols * cell_size, n_rows * cell_size))
    fig.suptitle('Hemispherical Harmonics (HSH)', fontsize=12, fontweight='bold', y=0.98)
    
    for l in range(max_degree + 1):
        start_col = max_degree - l  # Center alignment
        
        for i, m in enumerate(range(-l, l + 1)):
            # Only plot if (l + m) is even
            if (l + m) % 2 != 0:
                continue
            
            col = start_col + i
            subplot_idx = l * max_cols + col + 1
            
            ax = fig.add_subplot(n_rows, max_cols, subplot_idx, projection='3d')
            
            # Find the feature index
            target_name = f'Y:{l},{m}'
            try:
                idx = np.where(names == target_name)[0][0]
            except IndexError:
                ax.set_axis_off()
                continue
            
            Y = features[:, idx].reshape(grid_shape)
            
            vmax = np.max(np.abs(Y))
            norm = plt.Normalize(-vmax, vmax) if vmax > 0 else plt.Normalize(-1, 1)
            
            ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                           facecolors=cm.RdBu(norm(Y)),
                           rstride=1, cstride=1, linewidth=0, 
                           antialiased=True, shade=False)
            
            ax.set_title(f'$H_{{{l}}}^{{{m}}}$', fontsize=9, pad=1)
            ax.set_axis_off()
            ax.set_box_aspect([1, 1, 0.5])
            ax.view_init(elev=30, azim=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.5, w_pad=0.3)
    plt.savefig(savepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {savepath}")
    plt.close()


def compute_orthogonality_matrix(max_degree=4, n_lat=200, n_lon=400):
    """
    Compute Gram matrix for HSH on hemisphere using SphericalHarmonicsBasis.
    
    Uses correct integration weight: cos(lat) * dlat * dlon (in radians)
    """
    dlat = 90.0 / n_lat
    dlon = 360.0 / n_lon
    
    lat_centers = np.linspace(0 + dlat/2, 90 - dlat/2, n_lat)
    lon_centers = np.linspace(-180 + dlon/2, 180 - dlon/2, n_lon)
    
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    
    # Get basis values
    features, names, _ = get_basis_values_on_grid(
        max_degree, cup=True, lon_grid=lon_grid, lat_grid=lat_grid
    )
    
    # Integration weights
    weights = np.cos(np.radians(lat_grid.ravel())) * np.radians(dlat) * np.radians(dlon)
    
    # Compute Gram matrix
    features_weighted = features * np.sqrt(weights)[:, np.newaxis]
    gram_matrix = features_weighted.T @ features_weighted
    
    # Parse feature names to get (l, m) tuples
    hsh_terms = [parse_feature_name(n) for n in names]
    
    return gram_matrix, hsh_terms


def compute_sh_orthogonality_matrix(max_degree, hemisphere_only=False, n_lat=200, n_lon=400):
    """
    Compute Gram matrix for SH (full sphere or hemisphere).
    """
    dlat = (90.0 if hemisphere_only else 180.0) / n_lat
    dlon = 360.0 / n_lon
    
    if hemisphere_only:
        lat_centers = np.linspace(0 + dlat/2, 90 - dlat/2, n_lat)
    else:
        lat_centers = np.linspace(-90 + dlat/2, 90 - dlat/2, n_lat)
    lon_centers = np.linspace(-180 + dlon/2, 180 - dlon/2, n_lon)
    
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    
    # Get SH basis values
    features, names, _ = get_basis_values_on_grid(
        max_degree, cup=False, lon_grid=lon_grid, lat_grid=lat_grid
    )
    
    # Integration weights
    weights = np.cos(np.radians(lat_grid.ravel())) * np.radians(dlat) * np.radians(dlon)
    
    # Compute Gram matrix
    features_weighted = features * np.sqrt(weights)[:, np.newaxis]
    gram_matrix = features_weighted.T @ features_weighted
    
    # Parse feature names
    terms = [parse_feature_name(n) for n in names]
    
    return gram_matrix, terms


def plot_single_orthogonality(ax, gram_matrix, title, max_val=None):
    """Plot a single 2D heatmap on given axes."""
    
    # Use RdBu_r so that Red is positive, Blue is negative (common for correlation)
    # or RdBu (Red is negative, Blue is positive). 
    # Usually standard is: Red=Positive, Blue=Negative (RdBu_r) or 
    # Red=High (Positive), Blue=Low (Negative).
    # Matplotlib 'RdBu': Red=-ve, Blue=+ve
    # Matplotlib 'RdBu_r': Blue=-ve, Red=+ve. 
    # Let's use 'RdBu' as requested, ensuring 0 is white/center.
    
    if max_val is None:
        max_val = max(np.abs(gram_matrix).max(), 1.0)
    
    # Force symmetric colorbar centered at 0
    im = ax.imshow(gram_matrix, cmap='RdBu', interpolation='nearest',
                   vmin=-max_val, vmax=max_val)
    
    # Styling
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel('Function sequence', fontsize=8)
    ax.set_ylabel('Function sequence', fontsize=8)
    
    # Simplify ticks
    n_terms = gram_matrix.shape[0]
    step = 10 if n_terms > 20 else 5
    ticks = np.arange(0, n_terms + 1, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis='both', labelsize=7)
    
    # Remove grid for heatmap
    ax.grid(False)
    
    return im

def plot_orthogonality_comparison(max_degree=5, savepath=None):
    """Plot 3 orthogonality comparisons as heatmaps."""
    if savepath is None:
        savepath = OUTPUT_DIR / 'orthogonality_comparison.png'
    
    print("  Computing HSH on hemisphere...")
    gram_hsh_hemi, _ = compute_orthogonality_matrix(max_degree, n_lat=200, n_lon=400)
    
    print("  Computing SH on full sphere...")
    gram_sh_sphere, _ = compute_sh_orthogonality_matrix(max_degree, hemisphere_only=False, n_lat=200, n_lon=400)
    
    print("  Computing SH on hemisphere...")
    gram_sh_hemi, _ = compute_sh_orthogonality_matrix(max_degree, hemisphere_only=True, n_lat=200, n_lon=400)
    
    # Find global max for consistent coloring
    global_max = max(
        np.abs(gram_hsh_hemi).max(),
        np.abs(gram_sh_sphere).max(),
        np.abs(gram_sh_hemi).max()
    )
    
    # Setup figure with 3 subplots + 1 colorbar
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    im1 = plot_single_orthogonality(axes[0], gram_sh_sphere, 
                                    '(a) SH on full sphere', global_max)
    
    im2 = plot_single_orthogonality(axes[1], gram_sh_hemi, 
                                    '(b) SH on hemisphere', global_max)
    
    im3 = plot_single_orthogonality(axes[2], gram_hsh_hemi, 
                                    '(c) HSH on hemisphere', global_max)
    
    # Common Colorbar
    cbar = fig.colorbar(im3, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Inner Product', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    plt.savefig(savepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {savepath}")
    plt.close()
    
    # Print statistics
    print("  Statistics:")
    for name, gram in [("SH on sphere", gram_sh_sphere), 
                       ("SH on hemisphere", gram_sh_hemi),
                       ("HSH on hemisphere", gram_hsh_hemi)]:
        off_diag = gram.copy()
        np.fill_diagonal(off_diag, 0)
        print(f"    {name}: max_off_diag={np.max(np.abs(off_diag)):.4f}, "
              f"diag_error={np.max(np.abs(np.diag(gram) - 1.0)):.4f}")


def plot_coords_convert_demo(savepath=None):
    """
    Visualize different coords_convert_method effects.
    """
    if savepath is None:
        savepath = OUTPUT_DIR / 'coords_convert_methods.png'
    
    np.random.seed(42)
    n_points = 500
    lon = np.random.uniform(100, 125, n_points)
    lat = np.random.uniform(20, 45, n_points)
    X = np.column_stack([lon, lat])
    
    methods = ['basic', 'central', 'central_scale']
    titles = [
        'basic: θ = 90° - lat',
        'central: Rotated to pole',
        'central_scale: Rotated + scaled'
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original data
    ax0 = axes[0]
    ax0.scatter(lon, lat, s=5, alpha=0.6, c='steelblue')
    ax0.set_xlabel('Longitude (°)')
    ax0.set_ylabel('Latitude (°)')
    ax0.set_title('Original Data (lon, lat)')
    ax0.set_aspect('equal')
    ax0.grid(True, alpha=0.3)
    
    # Polar plots for each method
    for i, (method, title) in enumerate(zip(methods, titles)):
        ax = fig.add_subplot(1, 4, i + 2, projection='polar')
        axes[i + 1].remove()
        
        basis = SphericalHarmonicsBasis(
            degree=3,
            cup=True,
            pole='xyzmean',
            coords_convert_method=method,
            hemisphere_scale=0.5 if method == 'central_scale' else 1.0
        )
        basis.fit(X)
        theta, phi = basis.coords_converter_.transform(lon, lat)
        
        theta_deg = np.degrees(theta)
        
        ax.scatter(phi, theta_deg, s=5, alpha=0.6, c='steelblue')
        ax.set_title(title, fontsize=10, pad=10)
        ax.set_ylim(0, max(theta_deg.max() * 1.1, 90))
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        if method == 'central_scale':
            ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.suptitle('Coordinate Conversion Methods Comparison', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {savepath}")
    plt.close()


# =============================================================================
# Polynomial Basis Visualization
# =============================================================================

def plot_poly_basis_single(basis_type='polynomial', max_degree=3, savepath=None):
    """
    Plot a single polynomial basis type with 1D curves and 2D heatmaps.
    
    Upper part: 1D curves for each degree (0 to max_degree)
    Lower part: 2D heatmap pyramid for all terms i+j <= max_degree
    """
    from numpy.polynomial import Legendre, Chebyshev
    
    if savepath is None:
        savepath = OUTPUT_DIR / f'poly_{basis_type}.png'
    
    # 1D setup
    x_1d = np.linspace(-1, 1, 200)
    
    # 2D setup
    n_grid = 100
    x1 = np.linspace(-1, 1, n_grid)
    x2 = np.linspace(-1, 1, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Get basis functions
    def get_1d_basis(n, x):
        if basis_type == 'polynomial':
            return x ** n
        elif basis_type == 'legendre':
            return Legendre.basis(n)(x)
        else:  # chebyshev
            return Chebyshev.basis(n)(x)
    
    def get_2d_basis(i, j, X1, X2):
        return get_1d_basis(i, X1) * get_1d_basis(j, X2)
    
    # Figure layout
    # Upper: 1 row of 1D curves (max_degree+1 subplots)
    # Lower: pyramid of 2D heatmaps
    n_1d_cols = max_degree + 1
    n_2d_rows = max_degree + 1
    max_2d_cols = max_degree + 1  # Maximum columns in pyramid
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(12, 10))
    
    # Title
    title_map = {
        'polynomial': 'Standard Polynomial Basis ($x^n$)',
        'legendre': 'Legendre Polynomial Basis ($L_n$)',
        'chebyshev': 'Chebyshev Polynomial Basis ($T_n$)'
    }
    fig.suptitle(title_map.get(basis_type, basis_type), fontsize=14, fontweight='bold', y=0.98)
    
    # Upper part: 1D curves
    gs_upper = fig.add_gridspec(1, n_1d_cols, top=0.92, bottom=0.72, left=0.05, right=0.95, wspace=0.3)
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, max_degree + 1))
    for n in range(max_degree + 1):
        ax = fig.add_subplot(gs_upper[0, n])
        y = get_1d_basis(n, x_1d)
        ax.plot(x_1d, y, color=colors[n], linewidth=2)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.5, 1.5)
        
        if basis_type == 'polynomial':
            ax.set_title(f'$x^{n}$', fontsize=11)
        elif basis_type == 'legendre':
            ax.set_title(f'$L_{n}(x)$', fontsize=11)
        else:
            ax.set_title(f'$T_{n}(x)$', fontsize=11)
        
        ax.set_xlabel('$x$', fontsize=9)
        if n == 0:
            ax.set_ylabel('$P_n(x)$', fontsize=9)
        ax.tick_params(labelsize=8)
    
    # Lower part: 2D pyramid heatmaps
    # Calculate total terms and positions
    gs_lower = fig.add_gridspec(n_2d_rows, max_2d_cols * 2 - 1, 
                                 top=0.65, bottom=0.02, left=0.05, right=0.95, 
                                 wspace=0.1, hspace=0.15)
    
    for total_deg in range(max_degree + 1):
        n_terms = total_deg + 1
        start_col = max_degree - total_deg  # Center alignment
        
        for k, (i, j) in enumerate([(total_deg - t, t) for t in range(n_terms)]):
            col = start_col + k
            ax = fig.add_subplot(gs_lower[total_deg, col * 2])
            
            Z = get_2d_basis(i, j, X1, X2)
            vmax = np.max(np.abs(Z)) if np.max(np.abs(Z)) > 0 else 1
            
            im = ax.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', 
                          cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='equal')
            
            if basis_type == 'polynomial':
                label = f'$x_1^{i} x_2^{j}$' if i > 0 and j > 0 else (f'$x_1^{i}$' if i > 0 else (f'$x_2^{j}$' if j > 0 else '$1$'))
            elif basis_type == 'legendre':
                label = f'$L_{i} L_{j}$'
            else:
                label = f'$T_{i} T_{j}$'
            
            ax.set_title(label, fontsize=9, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.savefig(savepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {savepath}")
    plt.close()


def compute_poly_orthogonality_2d(basis_type='legendre', max_degree=3, n_grid=100):
    """
    Compute Gram matrix for 2D polynomial basis on [-1,1]^2.
    
    For Legendre: uniform weight
    For Chebyshev: Chebyshev weight 1/sqrt(1-x^2)
    """
    from numpy.polynomial import Legendre, Chebyshev
    from spatial_basis import PolynomialBasis
    
    # Create grid
    x1 = np.linspace(-1 + 1e-6, 1 - 1e-6, n_grid)  # Avoid endpoints for Chebyshev weight
    x2 = np.linspace(-1 + 1e-6, 1 - 1e-6, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    dx = 2.0 / n_grid
    
    # Create basis and get features
    # Note: PolynomialBasis normalizes input to [-1,1], but we're already in [-1,1]
    # So we need to work around this by providing data that maps correctly
    X = np.column_stack([X1.ravel(), X2.ravel()])
    
    basis = PolynomialBasis(degree=max_degree, basis=basis_type, include_bias=True)
    # Fit on [-1,1] range
    basis.fit(np.array([[-1, -1], [1, 1]]))
    features = basis.transform(X)
    
    # Compute weights
    if basis_type == 'chebyshev':
        # Chebyshev weight: 1/sqrt(1-x^2)
        w1 = 1.0 / np.sqrt(1 - x1**2)
        w2 = 1.0 / np.sqrt(1 - x2**2)
        W1, W2 = np.meshgrid(w1, w2)
        weights = (W1 * W2 * dx * dx).ravel()
    else:
        # Uniform weight for polynomial and Legendre
        weights = np.ones(len(X)) * dx * dx
    
    # Compute Gram matrix
    features_weighted = features * np.sqrt(weights)[:, np.newaxis]
    gram = features_weighted.T @ features_weighted
    
    return gram, basis.get_feature_names_out()


def plot_poly_orthogonality_2d(max_degree=3, savepath=None):
    """
    Plot orthogonality comparison for polynomial bases on [-1,1]^2.
    """
    if savepath is None:
        savepath = OUTPUT_DIR / 'poly_orthogonality_2d.png'
    
    print("  Computing Polynomial orthogonality...")
    gram_poly, _ = compute_poly_orthogonality_2d('polynomial', max_degree)
    
    print("  Computing Legendre orthogonality...")
    gram_leg, _ = compute_poly_orthogonality_2d('legendre', max_degree)
    
    print("  Computing Chebyshev orthogonality...")
    gram_cheb, _ = compute_poly_orthogonality_2d('chebyshev', max_degree)
    
    # Normalize to show relative orthogonality
    gram_poly_norm = gram_poly / np.max(np.abs(np.diag(gram_poly)))
    gram_leg_norm = gram_leg / np.max(np.abs(np.diag(gram_leg)))
    gram_cheb_norm = gram_cheb / np.max(np.abs(np.diag(gram_cheb)))
    
    global_max = max(np.abs(gram_poly_norm).max(), 
                     np.abs(gram_leg_norm).max(), 
                     np.abs(gram_cheb_norm).max())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    im1 = plot_single_orthogonality(axes[0], gram_poly_norm, 
                                    '(a) Polynomial (not orthogonal)', global_max)
    im2 = plot_single_orthogonality(axes[1], gram_leg_norm, 
                                    '(b) Legendre (orthogonal)', global_max)
    im3 = plot_single_orthogonality(axes[2], gram_cheb_norm, 
                                    '(c) Chebyshev (weighted orthogonal)', global_max)
    
    cbar = fig.colorbar(im3, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Normalized Inner Product', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    fig.suptitle('Orthogonality on $[-1,1]^2$', fontsize=12, fontweight='bold', y=1.02)
    
    plt.savefig(savepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {savepath}")
    plt.close()
    
    # Print statistics
    print("  Statistics:")
    for name, gram in [("Polynomial", gram_poly_norm), 
                       ("Legendre", gram_leg_norm),
                       ("Chebyshev", gram_cheb_norm)]:
        off_diag = gram.copy()
        np.fill_diagonal(off_diag, 0)
        print(f"    {name}: max_off_diag={np.max(np.abs(off_diag)):.4f}, "
              f"diag_std={np.std(np.diag(gram)):.4f}")


def plot_poly_orthogonality_sphere(max_degree=3, savepath=None):
    """
    Compare Legendre orthogonality on [-1,1]^2 vs on sphere.
    Shows that polynomial bases lose orthogonality on spherical domains.
    """
    if savepath is None:
        savepath = OUTPUT_DIR / 'poly_orthogonality_sphere.png'
    
    from spatial_basis import PolynomialBasis
    
    print("  Computing Legendre on [-1,1]^2...")
    gram_rect, _ = compute_poly_orthogonality_2d('legendre', max_degree, n_grid=100)
    
    print("  Computing Legendre on sphere...")
    # Spherical grid
    n_lat, n_lon = 100, 200
    dlat = 180.0 / n_lat
    dlon = 360.0 / n_lon
    
    lat_centers = np.linspace(-90 + dlat/2, 90 - dlat/2, n_lat)
    lon_centers = np.linspace(-180 + dlon/2, 180 - dlon/2, n_lon)
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    
    X = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    
    basis = PolynomialBasis(degree=max_degree, basis='legendre', include_bias=True)
    basis.fit(X)
    features = basis.transform(X)
    
    # Spherical weights
    weights = np.cos(np.radians(lat_grid.ravel())) * np.radians(dlat) * np.radians(dlon)
    
    features_weighted = features * np.sqrt(weights)[:, np.newaxis]
    gram_sphere = features_weighted.T @ features_weighted
    
    # Normalize
    gram_rect_norm = gram_rect / np.max(np.abs(np.diag(gram_rect)))
    gram_sphere_norm = gram_sphere / np.max(np.abs(np.diag(gram_sphere)))
    
    global_max = max(np.abs(gram_rect_norm).max(), np.abs(gram_sphere_norm).max())
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    im1 = plot_single_orthogonality(axes[0], gram_rect_norm, 
                                    '(a) Legendre on $[-1,1]^2$ (orthogonal)', global_max)
    im2 = plot_single_orthogonality(axes[1], gram_sphere_norm, 
                                    '(b) Legendre on sphere (NOT orthogonal)', global_max)
    
    cbar = fig.colorbar(im2, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Normalized Inner Product', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    fig.suptitle('Legendre Basis: Rectangle vs Sphere', fontsize=12, fontweight='bold', y=1.02)
    
    plt.savefig(savepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {savepath}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("Generating basis function visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Spherical Harmonics
    plot_sh_pyramid(max_degree=3)
    plot_hsh_pyramid(max_degree=4)
    plot_orthogonality_comparison(max_degree=5)
    plot_coords_convert_demo()
    
    # Polynomial Basis
    print("\nGenerating polynomial basis visualizations...")
    plot_poly_basis_single('polynomial', max_degree=3)
    plot_poly_basis_single('legendre', max_degree=3)
    plot_poly_basis_single('chebyshev', max_degree=3)
    plot_poly_orthogonality_2d(max_degree=3)
    plot_poly_orthogonality_sphere(max_degree=3)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

