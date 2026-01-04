# Spatial Basis

兼容 scikit-learn API 的空间基函数变换器，支持球谐函数（Spherical Harmonics, SH）、半球谐函数（Hemispherical Harmonics, HSH）和多种二维多项式函数，用于空间回归建模。

## 安装

```bash
pip install .
# 或使用 uv
uv pip install .
```

## 快速示例

```python
import numpy as np
from spatial_basis import SphericalHarmonicsBasis, PolynomialBasis

# 准备坐标：[经度, 纬度]，单位为度
X = np.column_stack([lon, lat])

# 球谐基函数
sh_basis = SphericalHarmonicsBasis(degree=2, cup=True)
X_sh = sh_basis.fit_transform(X)  # shape: (n_samples, 9)

# 多项式基函数
poly_basis = PolynomialBasis(degree=3, basis='legendre')
X_poly = poly_basis.fit_transform(X)  # shape: (n_samples, 10)
```

---

## API 参考

### SphericalHarmonicsBasis

```python
class spatial_basis.SphericalHarmonicsBasis(degree=2, cup=True,
    coords_convert_method='central_scale', pole='xyzmean',
    hemisphere_scale='auto', include_bias=True, force_norm=False)
```

从地理坐标生成球谐函数设计矩阵。根据 `cup` 参数选择球谐函数（SH）或半球谐函数（HSH）。

更多信息请参阅用户指南中的 [球面基函数](#球面基函数)。

#### Parameters

- **degree** : int, default=2

    球谐函数最高阶数 $L$。输出特征数：SH 为 $(L+1)^2$，HSH 为 $(L+1)(L+2)/2$。

- **cup** : bool, default=True

    若为 True，使用半球谐函数（HSH），仅保留偶宇称项，在半球面上保持正交性；若为 False，使用标准球谐函数（SH）。

- **coords_convert_method** : {'non', 'basic', 'central', 'central_scale'}, default='central_scale'

    坐标转换方法。`'non'`：不转换，直接使用输入作为 $(\theta, \phi)$；`'basic'`：经纬度转球坐标；`'central'`：将数据中心旋转到北极；`'central_scale'`：旋转后缩放到半球范围（推荐用于区域数据）。

- **pole** : str or tuple, default='xyzmean'

    旋转目标极点（仅 `'central'` 和 `'central_scale'` 方法生效）。`'xyzmean'`：笛卡尔坐标均值投影回球面；`'haversine'`：最小化最大球面距离的点；`(lat, lon)`：手动指定极点坐标。

- **hemisphere_scale** : str or float, default='auto'

    半球缩放因子（仅 `'central_scale'` 方法生效）。`'auto'`：HSH 时为 0.5（缩放到 90°），SH 时为 1.0（缩放到 180°）；也可传入 (0, 1] 范围的数值手动指定。

- **include_bias** : bool, default=True

    是否包含常数项 $Y_{0,0}$。

- **force_norm** : bool, default=False

    强制对输出列进行归一化。

#### Attributes

- **pole_** : tuple of float

    拟合后的极点位置 `(lat, lon)`，单位为度。

- **scale_** : float or None

    坐标缩放因子（仅 `'central_scale'` 方法时有值）。

- **n_output_features_** : int

    输出特征数量。

- **n_features_in_** : int

    输入特征数量（始终为 2）。

- **feature_names_in_** : ndarray of shape (2,)

    输入特征名称（如有）。

- **coords_converter_** : CoordsConverter

    内部坐标转换器对象。

- **terms_** : list of str

    球谐函数项列表，格式如 `['Y00', 'Y1-1', 'Y10', ...]`。

#### Methods

##### `fit(X, y=None)`

根据输入数据计算极点位置和坐标转换参数。

- **Parameters**

    - **X** : array-like of shape (n_samples, 2)

        输入坐标，格式为 `[经度, 纬度]`。

    - **y** : None

        忽略。

- **Returns**

    - **self** : object

        拟合后的变换器。

##### `transform(X)`

将坐标转换为球谐函数设计矩阵。

- **Parameters**

    - **X** : array-like of shape (n_samples, 2)

        输入坐标。

- **Returns**

    - **X_transformed** : ndarray of shape (n_samples, n_output_features_)

        球谐函数设计矩阵。

##### `fit_transform(X, y=None)`

拟合并转换数据。

- **Parameters**

    - **X** : array-like of shape (n_samples, 2)

        输入坐标。

    - **y** : None

        忽略。

- **Returns**

    - **X_transformed** : ndarray of shape (n_samples, n_output_features_)

        球谐函数设计矩阵。

##### `get_feature_names_out(input_features=None)`

获取输出特征名称。

- **Parameters**

    - **input_features** : array-like of str or None, default=None

        输入特征名称。若为 None，使用默认名称。

- **Returns**

    - **feature_names_out** : ndarray of str

        输出特征名称，格式为 `['Y:0,0', 'Y:1,-1', 'Y:1,0', 'Y:1,1', ...]`。

---

### PolynomialBasis

```python
class spatial_basis.PolynomialBasis(degree=2, include_bias=True, basis='polynomial')
```

生成二维多项式基函数设计矩阵。将二维输入转换为 $(d+1)(d+2)/2$ 个多项式特征。

更多信息请参阅用户指南中的 [平面基函数](#平面基函数)。

#### Parameters

- **degree** : int, default=2

    多项式最高阶数 $d$。输出特征数为 $(d+1)(d+2)/2$。

- **include_bias** : bool, default=True

    是否包含常数项。

- **basis** : {'polynomial', 'legendre', 'chebyshev'}, default='polynomial'

    基函数类型。`'legendre'` 和 `'chebyshev'` 在 $[-1,1]$ 上具有正交性。

#### Attributes

- **min_vals_**, **max_vals_** : ndarray of shape (2,)

    拟合时记录的输入范围。

- **range_** : ndarray of shape (2,)

    输入范围 `max_vals_ - min_vals_`，用于归一化计算。

- **n_output_features_** : int

    输出特征数量。

- **n_features_in_** : int

    输入特征数量（始终为 2）。

- **feature_names_in_** : ndarray of shape (2,)

    输入特征名称（如有）。

#### Methods

##### `fit(X, y=None)`

记录输入范围用于后续归一化。

- **Parameters**

    - **X** : array-like of shape (n_samples, 2)

        输入数据。

    - **y** : None

        忽略。

- **Returns**

    - **self** : object

        拟合后的变换器。

##### `transform(X)`

将输入转换为多项式设计矩阵。

- **Parameters**

    - **X** : array-like of shape (n_samples, 2)

        输入数据。

- **Returns**

    - **X_transformed** : ndarray of shape (n_samples, n_output_features_)

        多项式设计矩阵。

##### `fit_transform(X, y=None)`

拟合并转换数据。

- **Parameters**

    - **X** : array-like of shape (n_samples, 2)

        输入数据。

    - **y** : None

        忽略。

- **Returns**

    - **X_transformed** : ndarray of shape (n_samples, n_output_features_)

        多项式设计矩阵。

##### `get_feature_names_out(input_features=None)`

获取输出特征名称。

- **Parameters**

    - **input_features** : array-like of str or None, default=None

        输入特征名称。若为 None，使用默认名称。

- **Returns**

    - **feature_names_out** : ndarray of str

        输出特征名称，格式如 `['1', 'x0', 'x1', 'L2(x0)', ...]`。

---

## 用户指南

### 球面基函数

#### 坐标转换

球面基函数定义在球面坐标系，如果输入数据为地理坐标 $(\text{lon}, \text{lat})$，则需要先转换为球面坐标 $(\theta, \phi)$ ，其中 $\theta \in [0, \pi]$ 为余纬， $\phi \in [0, 2\pi]$ 为方位角。

`coords_convert_method` 参数控制转换方式：

| 方法 | 说明 | 适用场景 |
|:---|:---|:---|
| `'non'` | 直接输入球面坐标时无需转换 | 全球数据 |
| `'basic'` | 转换纬度为余纬度 | 全球数据 |
| `'central'` | 将极点旋转到北极 | 半球数据 |
| `'central_scale'` | 将极点旋转到北极，并缩放边界 | 球冠数据 |

![坐标转换方法](docs/coords_convert_methods.png)

`pole` 参数指定旋转的目标极点（仅 `'central'` 和 `'central_scale'` 生效）：`'xyzmean'` 使用笛卡尔质心，`'haversine'` 使用球面几何中心，或直接传入 `(lat, lon)` 手动指定。极点位置在 `fit()` 过程中计算并存储于 `basis.pole_`。

`hemisphere_scale` 参数控制缩放范围（仅 `'central_scale'` 生效）：`'auto'` 表示 HSH 时为 0.5、SH 时为 1.0；也可传入 (0, 1] 范围的数值手动指定。

#### 球谐函数与半球谐函数

球谐函数设计矩阵 $B \in \mathbb{R}^{N \times K}$ 包含 $N$ 个样本的 $K$ 个球谐函数值：

$$
B = \begin{bmatrix}
Y_{0,0}(\theta_1, \phi_1) & Y_{1,-1}(\theta_1, \phi_1) & Y_{1,0}(\theta_1, \phi_1) & \cdots & Y_{L,L}(\theta_1, \phi_1) \\\\
Y_{0,0}(\theta_2, \phi_2) & Y_{1,-1}(\theta_2, \phi_2) & Y_{1,0}(\theta_2, \phi_2) & \cdots & Y_{L,L}(\theta_2, \phi_2) \\\\
\vdots & \vdots & \vdots & \ddots & \vdots \\\\
Y_{0,0}(\theta_N, \phi_N) & Y_{1,-1}(\theta_N, \phi_N) & Y_{1,0}(\theta_N, \phi_N) & \cdots & Y_{L,L}(\theta_N, \phi_N)
\end{bmatrix}
$$

球谐函数定义为：

$$
Y_{l,m}(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi} \cdot \frac{(l-m)!}{(l+m)!}} \, P_l^{|m|}(\cos\theta) \begin{cases} \sqrt{2} \cos(m\phi) & m > 0 \\\\ 1 & m = 0 \\\\ \sqrt{2} \sin(|m|\phi) & m < 0 \end{cases}
$$

其中 $l$ 为阶数，$m$ 为阶次， $P_l^m(x)$ 为伴随勒让德多项式（含 Condon-Shortley 相位）。

![球谐函数可视化](docs/sh_pyramid.png)

半球谐函数定义为

$$
H_{l,m} = \sqrt{2} \, Y_{l,m} ,\quad l+m = \text{even.}
$$

半球谐函数是球谐函数的满足  $l+m \text{ = even.}$  的子集，在半球定义域上正交归一化

![半球谐函数可视化](docs/hsh_pyramid.png)

#### 函数特性

球谐函数/半球谐函数在各自定义域上满足正交归一性 $\langle f_i, f_j \rangle = \delta_{ij}$，其中 $\langle f_i, f_j \rangle = \int_{\Omega} f_i f_j \, d\Omega$。

![正交性验证](docs/orthogonality_comparison.png)

- SH 在全球：正交归一化（对角线 = 1，非对角线 ≈ 0）
- SH 在半球：失去正交性（对角线 = 0.5，非对角线 ≠ 0）
- HSH 在半球：正交归一化（对角线 = 1，非对角线 ≈ 0）

---

### 平面基函数

#### 多项式类型

`basis` 参数指定基函数类型：

| 类型 | 基函数 | 特性 |
|:---|:---|:---|
| `'polynomial'` | $x_1^i \cdot x_2^j$ | 标准幂多项式 |
| `'legendre'` | $L_i(x_1) \cdot L_j(x_2)$ | 在 $[-1,1]$ 上正交 |
| `'chebyshev'` | $T_i(x_1) \cdot T_j(x_2)$ | 在 $[-1,1]$ 上带权正交 |

对于 `degree=d`，生成所有满足 $i + j \le d$ 的项，特征数量为 $K = (d+1)(d+2)/2$。

![标准多项式](docs/poly_polynomial.png)
![Legendre 多项式](docs/poly_legendre.png)
![Chebyshev 多项式](docs/poly_chebyshev.png)

#### 归一化

在 `fit()` 阶段记录输入范围，`transform()` 时将输入归一化到 $[-1, 1]$：

$$
x \to 2 \cdot \frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1
$$

这对 Legendre 和 Chebyshev 多项式至关重要，因为它们在 $[-1, 1]$ 上具有最佳数值性质。

#### 函数特性

不同多项式基函数在 $[-1,1]^2$ 矩形域上的正交性：Polynomial 不正交，Legendre 正交，Chebyshev 带权正交。

![二维正交性对比](docs/poly_orthogonality_2d.png)

注意：由于球面测度 $d\Omega = \sin\theta \, d\theta \, d\phi$ 与矩形测度 $dx_1 \, dx_2$ 不同，Legendre 多项式在球面上失去正交性。

![球面正交性](docs/poly_orthogonality_sphere.png)
