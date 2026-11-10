"""
Module 4: Classical Feature Extraction & Pattern Classification (Doc Chapters 4, 10, 11, 12)
============================================================================================
Implements strictly from IMP302 Course Lecture Slides (`Doc/4, 10, 11, 12.pdf`):

1. Transform-Domain Features (Chapter 4: Image and Video Analysis):
   - Sequency-Ordered 2D Walsh-Hadamard Transform (WHT, Ch. 4.4 Slides 12-16):
     T(u,v) = H_seq * f(x,y) * H_seq^T, extracting top-left 10x10 low-sequency block (100 dims)
   - 2D Discrete Cosine Transform (DCT-II, Ch. 4.3 Slides 7-11):
     Top-left 10x10 low-frequency DCT energy block (100 dims)
   - 2D Discrete Hartley Transform (DHT, Ch. 4.2 Slides 4-6):
     cas(theta) = cos(theta) + sin(theta), extracting 10x10 low-frequency Hartley block (100 dims)

2. Boundary, Region, Moment & Gradient Descriptors (Chapter 10: Feature Extraction):
   - Boundary Fourier Descriptors (Ch. 10.1 Slides 7-8):
     Complex boundary s(k) = x(k) + j*y(k), DFT coefficients a(u), translation/scale/rotation
     invariant normalized magnitudes (8 harmonic descriptors)
   - Region Effective Diameter & Covariance Eigenvalue Eccentricity (Ch. 10.2 Slides 11-12):
     2x2 coordinate covariance matrix C, eigenvalues lambda_1 >= lambda_2, eccentricity sqrt(1 - lambda_2/lambda_1)
   - 7 Log-transformed Hu Invariant Moments (Ch. 10.2 Slides 21-23)
   - Topological Euler Number E = C - H (Connected Components minus Holes), hole area ratio,
     and vertical hole distribution
   - 4x4 Cartesian Zoning (16-d) + Horizontal/Vertical Projection Profiles (16-d)
   - Histogram of Oriented Gradients (HOG, Ch. 10 & Ch. 11.1): 8x8 cells, 9 unsigned bins,
     2x2 L2-Hys block normalization (324 dimensions) + HOG orientation rose visualization
   - Principal Component Analysis (PCA as Features, Ch. 10.3 Slides 24-25)

3. Classical Image Pattern Classifiers (Chapter 12: Image Pattern Classification):
   - Minimum-Distance Classifier ('min_dist', Ch. 12.2 Slides 9-11):
     d_j(x) = x^T m_j - 0.5 ||m_j||^2 (Euclidean prototype matching)
   - Correlation Coefficient Classifier ('corr', Ch. 12.2 Slide 12):
     Normalized Cross-Correlation template matching in feature/image space
   - Optimum Statistical Bayes Classifier for Gaussian Classes ('bayes', Ch. 12.3 Slides 13-20):
     d_j(x) = ln P(w_j) - 0.5 ln|C_j| - 0.5 (x - m_j)^T C_j^{-1} (x - m_j)
   - Multi-Layer Perceptron Neural Network ('mlp', Ch. 12.4 Slides 21-27):
     Feedforward neural network with Backpropagation
   - Distance-Weighted k-Nearest Neighbors ('knn')
   - Support Vector Machine ('svm', Ch. 11.1 Slides 10-12)
   - Random Forest ('rf')
   - Positional Syntax-Constrained Decoding for Vietnamese 2-Line License Plates
"""

import cv2
import numpy as np
from scipy.linalg import hadamard
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. CHAPTER 4: TRANSFORM-DOMAIN FEATURES (WHT, DCT, DHT)
# =============================================================================

def get_sequency_hadamard_matrix(n: int = 32) -> np.ndarray:
    """
    Construct the orthonormal n x n Walsh-Hadamard matrix ordered by Sequency
    (number of zero-crossings / sign changes per basis row: 0, 1, 2, ..., n-1)
    (Ch. 4.4 Slides 12-16).
    """
    H = hadamard(n)
    sign_changes = np.sum(np.diff(H, axis=1) != 0, axis=1)
    seq_order = np.argsort(sign_changes)
    return H[seq_order].astype(np.float32) / np.sqrt(float(n))


def get_hartley_matrix(n: int = 32) -> np.ndarray:
    """
    Construct the orthonormal 1D Discrete Hartley Transform (DHT) kernel matrix (Ch. 4.2 Slides 4-6):
        H_DHT(u, x) = (1 / sqrt(n)) * cas(2 * pi * u * x / n)
        where cas(theta) = cos(theta) + sin(theta).
    """
    u = np.arange(n, dtype=np.float32)[:, None]
    x = np.arange(n, dtype=np.float32)[None, :]
    theta = 2.0 * np.pi * u * x / float(n)
    return (np.cos(theta) + np.sin(theta)) / np.sqrt(float(n))


H32_SEQ = get_sequency_hadamard_matrix(32)
DHT32_MAT = get_hartley_matrix(32)


def compute_wht_2d(char_img_32: np.ndarray) -> np.ndarray:
    """
    Compute the full 32x32 2D Sequency-Ordered Walsh-Hadamard Transform (Ch. 4.4):
        W = H32_SEQ @ (I / 255.0) @ H32_SEQ.T
    """
    f = char_img_32.astype(np.float32) / 255.0
    return H32_SEQ @ f @ H32_SEQ.T


def extract_wht_features(char_img_32: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Extract the top-left k x k low-sequency WHT coefficients (default 10x10 = 100 dims).
    """
    wht_2d = compute_wht_2d(char_img_32)
    return wht_2d[:k, :k].flatten()


def compute_dct_2d(char_img_32: np.ndarray) -> np.ndarray:
    """
    Compute the 32x32 2D Discrete Cosine Transform (DCT-II, Ch. 4.3 Slides 7-11).
    """
    f = char_img_32.astype(np.float32) / 255.0
    return cv2.dct(f)


def extract_dct_features(char_img_32: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Extract the top-left k x k low-frequency 2D DCT coefficients (100 dims, Ch. 4.3).
    """
    dct_2d = compute_dct_2d(char_img_32)
    return dct_2d[:k, :k].flatten()


def compute_dht_2d(char_img_32: np.ndarray) -> np.ndarray:
    """
    Compute the 32x32 2D Discrete Hartley Transform (Ch. 4.2 Slides 4-6):
        H(u,v) = (1/N) sum_x sum_y f(x,y) cas(2*pi*(u*x + v*y)/N)
    Using the separable cas relation or matrix transform.
    """
    f = char_img_32.astype(np.float32) / 255.0
    # 2D separable cas transform
    return DHT32_MAT @ f @ DHT32_MAT.T


def extract_dht_features(char_img_32: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Extract the top-left k x k low-frequency 2D Hartley Transform coefficients (100 dims, Ch. 4.2).
    """
    dht_2d = compute_dht_2d(char_img_32)
    return dht_2d[:k, :k].flatten()


# =============================================================================
# 2. CHAPTER 10: FEATURE EXTRACTION (HOG, FOURIER BOUNDARY, COVARIANCE, HU)
# =============================================================================

def extract_hog_features(char_img_32: np.ndarray, cell_size: int = 8, nbins: int = 9, return_vis: bool = False):
    """
    Compute Histogram of Oriented Gradients (HOG, Ch. 10 & Ch. 11) on a 32x32 image:
    - Gradient magnitude M = sqrt(Gx^2 + Gy^2), unsigned orientation theta in [0, 180)
    - 4x4 cells of 8x8 pixels, 9 orientation bins with bilinear bin interpolation
    - 3x3 overlapping 2x2-cell blocks normalized with L2-Hys (324 dimensions)
    """
    f = char_img_32.astype(np.float32) / 255.0
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ori = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    
    n_cells = 32 // cell_size
    cell_hists = np.zeros((n_cells, n_cells, nbins), dtype=np.float32)
    bin_width = 180.0 / nbins
    
    for r in range(n_cells):
        for c in range(n_cells):
            m_cell = mag[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size].flatten()
            o_cell = ori[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size].flatten()
            bin_float = o_cell / bin_width
            b0 = np.floor(bin_float).astype(int) % nbins
            b1 = (b0 + 1) % nbins
            w1 = bin_float - np.floor(bin_float)
            w0 = 1.0 - w1
            np.add.at(cell_hists[r, c], b0, m_cell * w0)
            np.add.at(cell_hists[r, c], b1, m_cell * w1)
            
    blocks = []
    eps = 1e-5
    for r in range(n_cells - 1):
        for c in range(n_cells - 1):
            blk = cell_hists[r:r+2, c:c+2, :].flatten()
            norm = np.sqrt(np.sum(blk ** 2) + eps ** 2)
            blk = np.minimum(blk / norm, 0.2)
            blk = blk / np.sqrt(np.sum(blk ** 2) + eps ** 2)
            blocks.append(blk)
    feat_vec = np.concatenate(blocks)
    
    if not return_vis:
        return feat_vec
        
    scale = 4
    vis = np.zeros((32 * scale, 32 * scale), dtype=np.uint8)
    max_h = float(np.max(cell_hists) + 1e-6)
    radius = (cell_size * scale) * 0.44
    for r in range(n_cells):
        for c in range(n_cells):
            cy = int((r + 0.5) * cell_size * scale)
            cx = int((c + 0.5) * cell_size * scale)
            for b in range(nbins):
                strength = cell_hists[r, c, b] / max_h
                if strength < 0.08:
                    continue
                angle_rad = np.deg2rad(b * bin_width + bin_width * 0.5 + 90.0)
                dx = int(np.cos(angle_rad) * radius * strength)
                dy = int(np.sin(angle_rad) * radius * strength)
                intensity = int(np.clip(strength * 255.0 + 40.0, 0, 255))
                cv2.line(vis, (cx - dx, cy - dy), (cx + dx, cy + dy), intensity, 1, cv2.LINE_AA)
    return feat_vec, vis


def extract_fourier_descriptors(char_bin_32: np.ndarray, num_descriptors: int = 8) -> np.ndarray:
    """
    Boundary Fourier Descriptors from First Principles (Ch. 10.1 Slides 7-8):
    1. Trace outer boundary coordinates s(k) = x(k) + j * y(k), k = 0, ..., K-1.
    2. Compute 1D Discrete Fourier Transform: a(u) = (1/K) sum_{k=0}^{K-1} s(k) exp(-j 2*pi*u*k / K).
    3. Translation invariance: discard DC component a(0).
    4. Rotation & shift invariance: take magnitudes |a(u)|.
    5. Scale invariance: normalize by fundamental frequency |a(1)| + eps.
    """
    cnts, _ = cv2.findContours(char_bin_32, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros(num_descriptors, dtype=np.float32)
    largest = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    if len(largest) < 4:
        return np.zeros(num_descriptors, dtype=np.float32)
    
    # Complex sequence s(k) = x(k) + j*y(k) centered at centroid (a(0) = 0)
    z = largest[:, 0].astype(np.float64) + 1j * largest[:, 1].astype(np.float64)
    z = z - np.mean(z)
    dft_coeffs = np.fft.fft(z)
    mags = np.abs(dft_coeffs[1:])  # exclude a(0)
    if len(mags) == 0:
        return np.zeros(num_descriptors, dtype=np.float32)
    scale = mags[0] if mags[0] > 1e-6 else (np.max(mags) + 1e-6)
    norm_mags = mags / scale
    
    out = np.zeros(num_descriptors, dtype=np.float32)
    k = min(num_descriptors, len(norm_mags))
    out[:k] = norm_mags[:k].astype(np.float32)
    return out


def compute_region_covariance_descriptors(char_bin_32: np.ndarray) -> np.ndarray:
    """
    Region Descriptors from Ch. 10.2 Slides 11-12:
    - Effective diameter: d_e = 2 * sqrt(Area / pi) normalized by image width (32)
    - Covariance Matrix of foreground coordinates z_k = (x_k, y_k)^T:
          C = (1/K) sum_{k=1}^K (z_k - m)(z_k - m)^T
      Eigenvalues lambda_1 >= lambda_2 of C give principal axes dispersion;
      Eccentricity e = sqrt(1 - lambda_2 / (lambda_1 + eps)).
    Returns [effective_diam_norm, eccentricity, major_minor_ratio].
    """
    ys, xs = np.nonzero(char_bin_32 > 0)
    area = float(len(xs))
    if area < 4:
        return np.zeros(3, dtype=np.float32)
    
    eff_diam = (2.0 * np.sqrt(area / np.pi)) / 32.0
    coords = np.vstack([xs.astype(np.float64), ys.astype(np.float64)])
    cov = np.cov(coords)
    eigvals = np.linalg.eigvalsh(cov)
    l1 = max(float(eigvals[-1]), 1e-6)
    l2 = max(float(eigvals[0]), 0.0)
    eccentricity = float(np.sqrt(max(0.0, 1.0 - l2 / l1)))
    axis_ratio = float(np.sqrt(l2 / l1))
    return np.array([eff_diam, eccentricity, axis_ratio], dtype=np.float32)


def compute_topology_details(char_bin_32: np.ndarray) -> dict:
    """
    Compute topological properties (Ch. 10):
    - Number of connected components C
    - Number of enclosed holes H (with area >= 6 px)
    - Euler Number E = C - H
    - Hole area ratio & top/bottom hole counts
    """
    padded = cv2.copyMakeBorder(char_bin_32, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    cnts, hier = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    num_objects = 0
    num_holes = 0
    hole_top_y = 0.0
    hole_bot_y = 0.0
    hole_area_ratio = 0.0
    total_fg = float(np.sum(padded > 0) + 1e-5)
    
    if hier is not None:
        for i, h in enumerate(hier[0]):
            if h[3] == -1:
                num_objects += 1
            else:
                area_h = cv2.contourArea(cnts[i])
                if area_h >= 6:
                    num_holes += 1
                    hole_area_ratio += area_h / total_fg
                    M = cv2.moments(cnts[i])
                    cy = (M["m01"] / (M["m00"] + 1e-5)) / 36.0
                    if cy < 0.5:
                        hole_top_y += 1.0
                    else:
                        hole_bot_y += 1.0
    euler = num_objects - num_holes
    return {
        "num_objects": num_objects,
        "num_holes": num_holes,
        "euler_number": euler,
        "hole_area_ratio": hole_area_ratio,
        "hole_top": hole_top_y,
        "hole_bot": hole_bot_y,
    }


def extract_hu_and_topology(char_bin_32: np.ndarray) -> np.ndarray:
    """
    Extract 56-d Structural, Moment, Boundary & Topological feature vector (Ch. 10):
    - 7 log-Hu Invariant Moments (Ch. 10.2 Slides 21-23)
    - 3 Region Covariance & Effective Diameter descriptors (Ch. 10.2 Slides 11-12)
    - 8 Boundary Fourier Descriptors (Ch. 10.1 Slides 7-8)
    - 6 Topological Euler & hole features
    - 16 (4x4) Zoning density features
    - 16 (8 horizontal + 8 vertical) Projection Profile features
    """
    moments = cv2.moments(char_bin_32)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    
    reg_cov = compute_region_covariance_descriptors(char_bin_32)
    fourier_desc = extract_fourier_descriptors(char_bin_32, num_descriptors=8)
    
    t = compute_topology_details(char_bin_32)
    topo = np.array(
        [t["num_objects"], t["num_holes"], t["euler_number"], t["hole_area_ratio"], t["hole_top"], t["hole_bot"]],
        dtype=np.float32,
    )
    
    f = (char_bin_32 > 0).astype(np.float32)
    zoning = f.reshape(4, 8, 4, 8).mean(axis=(1, 3)).flatten()
    h_proj = f.reshape(8, 4, 32).mean(axis=(1, 2))
    v_proj = f.reshape(32, 8, 4).mean(axis=(0, 2))
    return np.concatenate([hu_log.astype(np.float32), reg_cov, fourier_desc, topo, zoning, h_proj, v_proj])


# =============================================================================
# 3. CHAPTER 12: CLASSICAL IMAGE PATTERN CLASSIFIERS FROM FIRST PRINCIPLES
# =============================================================================

class CorrelationCoefficientClassifier:
    """
    Correlation Coefficient Classifier / Normalized Cross-Correlation (Ch. 12.2 Slide 12):
        gamma_j(x) = ((x - mean(x))^T (m_j - mean(m_j))) / (||x - mean(x)|| * ||m_j - mean(m_j)||)
    Assigns pattern x to the class w_j maximizing the normalized correlation coefficient gamma_j(x).
    """

    def __init__(self):
        self.classes_ = None
        self.templates_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        templates = []
        for c in self.classes_:
            proto = np.mean(X[y == c], axis=0)
            proto_zm = proto - np.mean(proto)
            norm = np.linalg.norm(proto_zm) + 1e-8
            templates.append(proto_zm / norm)
        self.templates_ = np.array(templates, dtype=np.float32)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_zm = X - np.mean(X, axis=1, keepdims=True)
        norms = np.linalg.norm(X_zm, axis=1, keepdims=True) + 1e-8
        X_norm = X_zm / norms
        return X_norm @ self.templates_.T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        corr = self.decision_function(X)
        logits = corr * 10.0
        logits -= np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / (np.sum(probs, axis=1, keepdims=True) + 1e-8)

    def predict(self, X: np.ndarray) -> np.ndarray:
        corr = self.decision_function(X)
        return self.classes_[np.argmax(corr, axis=1)]


class GaussianBayesClassifier:
    """
    Optimum Statistical Bayes Classifier for Gaussian Pattern Classes (Ch. 12.3 Slides 13-20):
        d_j(x) = ln P(w_j) - 0.5 * ln|C_j| - 0.5 * (x - m_j)^T C_j^{-1} (x - m_j)
    Uses regularized class covariance matrices C_j_reg = (1 - alpha)*C_j + alpha*C_pooled + reg*I
    so that classes with few training samples remain well-conditioned in high dimensions.
    """

    def __init__(self, reg_lambda: float = 0.08, pool_alpha: float = 0.65, uniform_prior: bool = True):
        self.reg_lambda = reg_lambda
        self.pool_alpha = pool_alpha
        self.uniform_prior = uniform_prior
        self.classes_ = None
        self.means_ = None
        self.inv_covs_ = None
        self.log_dets_ = None
        self.log_priors_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Pooled within-class covariance matrix (Ch. 12.3 Slide 19 special case C_j = C)
        C_pooled = np.zeros((n_features, n_features), dtype=np.float64)
        means = []
        class_covs = []
        priors = []

        for c in self.classes_:
            Xc = X[y == c]
            nc = len(Xc)
            mc = np.mean(Xc, axis=0)
            means.append(mc)
            priors.append(1.0 / n_classes if self.uniform_prior else nc / float(n_samples))
            diff = Xc - mc
            Cc = (diff.T @ diff) / max(nc, 1)
            class_covs.append(Cc)
            C_pooled += (nc / float(n_samples)) * Cc

        self.means_ = np.array(means, dtype=np.float64)
        self.log_priors_ = np.log(np.array(priors, dtype=np.float64) + 1e-12)
        self.inv_covs_ = []
        self.log_dets_ = []

        eye = np.eye(n_features, dtype=np.float64) * self.reg_lambda
        for Cc in class_covs:
            C_reg = (1.0 - self.pool_alpha) * Cc + self.pool_alpha * C_pooled + eye
            sign, logdet = np.linalg.slogdet(C_reg)
            inv_C = np.linalg.inv(C_reg)
            self.inv_covs_.append(inv_C)
            self.log_dets_.append(logdet if sign > 0 else 0.0)

        self.inv_covs_ = np.array(self.inv_covs_, dtype=np.float64)
        self.log_dets_ = np.array(self.log_dets_, dtype=np.float64)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes), dtype=np.float64)
        for j in range(n_classes):
            diff = X - self.means_[j]
            mahal = np.sum((diff @ self.inv_covs_[j]) * diff, axis=1)
            scores[:, j] = self.log_priors_[j] - 0.5 * self.log_dets_[j] - 0.5 * mahal
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        scores -= np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / (np.sum(probs, axis=1, keepdims=True) + 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


class ClassicalCharRecognizer:
    """
    Multi-feature Classical Pattern Classifier (Chapters 4, 10, 11, 12) supporting:
    - Feature sets:
      * 'wht'          : Sequency 2D Walsh-Hadamard Transform (100-d, Ch. 4.4)
      * 'dct'          : 2D Discrete Cosine Transform (100-d, Ch. 4.3)
      * 'dht'          : 2D Discrete Hartley Transform (100-d, Ch. 4.2)
      * 'hog'          : Histogram of Oriented Gradients (324-d, Ch. 10 & 11)
      * 'hu_topo'      : 7 Hu Moments + Region Covariance + Fourier Boundary + Topology (56-d, Ch. 10)
      * 'combined'     : Hybrid Transform (WHT) + Gradient (HOG) + Structural/Moment/Fourier (480-d)
      * 'combined_pca' : PCA Eigen-Feature Subspace projection of Combined vector (Ch. 10.3 Slides 24-25)
    - Classifiers (Chapter 12):
      * 'min_dist' : Minimum-Distance Classifier (Ch. 12.2 Slides 9-11)
      * 'corr'     : Correlation Coefficient Classifier (Ch. 12.2 Slide 12)
      * 'bayes'    : Statistical Gaussian Bayes Classifier (Ch. 12.3 Slides 13-20)
      * 'mlp'      : Multi-Layer Perceptron Neural Network (Ch. 12.4 Slides 21-27)
      * 'knn'      : Distance-weighted k-Nearest Neighbors (k=3)
      * 'svm'      : RBF-Kernel Support Vector Machine (C=12.0, Ch. 11.1)
      * 'rf'       : Random Forest (200 trees)
    """

    def __init__(self, feature_mode: str = "combined", classifier_type: str = "svm", pca_dims: int = 80):
        self.feature_mode = feature_mode
        self.classifier_type = classifier_type
        self.pca_dims = pca_dims
        self.hu_scaler = StandardScaler()
        self.pca = PCA(n_components=pca_dims, random_state=42) if feature_mode == "combined_pca" else None
        self.classes_ = None
        self.clf = self._build_clf(classifier_type)

    @staticmethod
    def _build_clf(classifier_type: str):
        if classifier_type == "min_dist":
            return NearestCentroid()
        elif classifier_type == "corr":
            return CorrelationCoefficientClassifier()
        elif classifier_type == "bayes":
            return GaussianBayesClassifier(reg_lambda=0.06, pool_alpha=0.65)
        elif classifier_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=400,
                random_state=42,
                early_stopping=False,
            )
        elif classifier_type == "knn":
            return KNeighborsClassifier(n_neighbors=3, weights="distance", metric="euclidean")
        elif classifier_type == "svm":
            return SVC(C=12.0, kernel="rbf", gamma="scale", probability=True, random_state=42)
        elif classifier_type == "rf":
            return RandomForestClassifier(n_estimators=200, random_state=42)
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    @staticmethod
    def _l2_norm(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        return mat / norms

    def extract_raw_feature_blocks(self, char_items: list[dict]) -> dict[str, np.ndarray]:
        X_wht = np.array([extract_wht_features(it["area_32"]) for it in char_items], dtype=np.float32)
        X_dct = np.array([extract_dct_features(it["area_32"]) for it in char_items], dtype=np.float32)
        X_dht = np.array([extract_dht_features(it["area_32"]) for it in char_items], dtype=np.float32)
        X_hog = np.array([extract_hog_features(it["gray_32"]) for it in char_items], dtype=np.float32)
        X_hu = np.array([extract_hu_and_topology(it["bin_32"]) for it in char_items], dtype=np.float32)
        return {"wht": X_wht, "dct": X_dct, "dht": X_dht, "hog": X_hog, "hu_topo": X_hu}

    def prepare_features(
        self,
        X_wht: np.ndarray,
        X_hog: np.ndarray,
        X_hu: np.ndarray,
        fit_scaler: bool = False,
        X_dct: np.ndarray | None = None,
        X_dht: np.ndarray | None = None,
    ) -> np.ndarray:
        w = self._l2_norm(X_wht)
        g = self._l2_norm(X_hog)
        if fit_scaler:
            h_scaled = self.hu_scaler.fit_transform(X_hu)
        else:
            h_scaled = self.hu_scaler.transform(X_hu)
        h = self._l2_norm(h_scaled)
        
        if self.feature_mode == "wht":
            return w
        elif self.feature_mode == "dct":
            return self._l2_norm(X_dct) if X_dct is not None else w
        elif self.feature_mode == "dht":
            return self._l2_norm(X_dht) if X_dht is not None else w
        elif self.feature_mode == "hog":
            return g
        elif self.feature_mode == "hu_topo":
            return h
        elif self.feature_mode in ("combined", "combined_pca"):
            comb = np.hstack([w * 0.70, g * 1.75, h * 0.45])
            if self.feature_mode == "combined_pca":
                if fit_scaler:
                    n_comp = min(self.pca_dims, comb.shape[0] - 1, comb.shape[1])
                    self.pca = PCA(n_components=n_comp, random_state=42)
                    return self.pca.fit_transform(comb)
                return self.pca.transform(comb)
            return comb
        raise ValueError(f"Unknown feature_mode: {self.feature_mode}")

    def fit(self, char_items: list[dict], labels: list[str], augment_missing_letters: bool = True):
        train_items = list(char_items)
        train_labels = list(labels)
        if augment_missing_letters:
            from collections import Counter
            counts = Counter(train_labels)
            # Standard Vietnamese license plate series letters (includes A, R absent from the 115-plate dataset)
            vn_letters = ("A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "U", "V", "X", "Y", "Z")
            missing = tuple(ch for ch in vn_letters if counts[ch] == 0)
            if missing:
                syn_items, syn_labels = generate_synthetic_vietnamese_letters(missing, samples_per_letter=16)
                train_items.extend(syn_items)
                train_labels.extend(syn_labels)

            # Balance rare letter classes (1 <= count < 14, e.g. 'D' which only has 3 samples vs 'B' with 12)
            # using affine jitter on real dataset patches + synthetic prototypes so 5-fold SVM Platt scaling is unbiased
            rng = np.random.default_rng(42)
            for ch in vn_letters:
                if 0 < counts[ch] < 14:
                    real_ch_items = [it for it, lb in zip(char_items, labels) if lb == ch]
                    needed = 14 - counts[ch]
                    for k_aug in range(needed):
                        base_it = real_ch_items[k_aug % len(real_ch_items)]
                        ang = float(rng.uniform(-4.0, 4.0))
                        tx = float(rng.uniform(-1.0, 1.0))
                        ty = float(rng.uniform(-1.0, 1.0))
                        M = cv2.getRotationMatrix2D((16.0, 16.0), ang, float(rng.uniform(0.95, 1.05)))
                        M[0, 2] += tx
                        M[1, 2] += ty
                        b32 = cv2.warpAffine(base_it["bin_32"], M, (32, 32), flags=cv2.INTER_NEAREST, borderValue=0)
                        a32 = cv2.warpAffine(base_it["area_32"], M, (32, 32), flags=cv2.INTER_LINEAR, borderValue=0)
                        g32 = cv2.warpAffine(base_it["gray_32"], M, (32, 32), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        train_items.append({**base_it, "bin_32": b32, "area_32": a32, "gray_32": g32})
                        train_labels.append(ch)
                    if ch == "D":
                        syn_d, syn_d_lbl = generate_synthetic_vietnamese_letters(("D",), samples_per_letter=8, seed=99)
                        train_items.extend(syn_d)
                        train_labels.extend(syn_d_lbl)

        blocks = self.extract_raw_feature_blocks(train_items)
        X = self.prepare_features(
            blocks["wht"],
            blocks["hog"],
            blocks["hu_topo"],
            fit_scaler=True,
            X_dct=blocks["dct"],
            X_dht=blocks["dht"],
        )
        y = np.array(train_labels)
        self.classes_ = np.unique(y)
        self.clf.fit(X, y)
        return self

    def predict_chars(self, char_items: list[dict], apply_syntax: bool = True) -> tuple[str, list[tuple[str, float]]]:
        """
        Predict the character sequence for a segmented license plate.
        If `apply_syntax=True`, enforces Vietnamese 2-line motorbike plate syntax rules:
        - Positions 0, 1 (province code): digits '0'-'9'
        - Position 2 (series letter): uppercase letters 'A'-'Z'
        - Position 3 (series code): alphanumeric ('0'-'9' or 'A'-'Z')
        - Positions 4..end (bottom row serial number): digits '0'-'9'
        """
        if not char_items:
            return "", []
            
        blocks = self.extract_raw_feature_blocks(char_items)
        X = self.prepare_features(
            blocks["wht"],
            blocks["hog"],
            blocks["hu_topo"],
            fit_scaler=False,
            X_dct=blocks["dct"],
            X_dht=blocks["dht"],
        )
        
        if hasattr(self.clf, "predict_proba"):
            scores = self.clf.predict_proba(X)
        elif hasattr(self.clf, "decision_function"):
            raw_dec = self.clf.decision_function(X)
            scores = np.exp(raw_dec - np.max(raw_dec, axis=1, keepdims=True))
            scores /= np.sum(scores, axis=1, keepdims=True)
        else:
            dists = np.linalg.norm(X[:, None, :] - self.clf.centroids_[None, :, :], axis=2)
            scores = np.exp(-dists * 4.0)
            scores /= np.sum(scores, axis=1, keepdims=True)
            
        classes = list(self.clf.classes_)
        preds = []
        for idx, item in enumerate(char_items):
            prob_vec = scores[idx].copy()
            if apply_syntax:
                row = item.get("row", 0 if idx < 4 else 1)
                for c_i, ch in enumerate(classes):
                    is_digit = ch.isdigit()
                    is_alpha = ch.isalpha()
                    if row == 1 or idx in (0, 1) or idx >= 4:
                        if not is_digit:
                            prob_vec[c_i] *= 1e-4
                    elif idx == 2:
                        if not is_alpha:
                            prob_vec[c_i] *= 1e-4
                    elif idx == 3 and len(preds) >= 3 and preds[2][0] == "A":
                        # Vietnamese <50cc motorbike plates with series 'A' always use 2-letter code (AA, AB, AC...)
                        if not is_alpha:
                            prob_vec[c_i] *= 1e-4

            # Chapter 10 Topological & Boundary Disambiguation for 'D' vs 'B', 'E' vs 'C', and '7' vs '1':
            topo = compute_topology_details(item["bin_32"])
            if "B" in classes and "D" in classes:
                i_b, i_d = classes.index("B"), classes.index("D")
                if max(prob_vec[i_b], prob_vec[i_d]) >= 0.05:
                    waist_bar = float(np.mean(item["bin_32"][14:19, 14:19] > 0))
                    if waist_bar < 0.26 and topo["num_holes"] <= 1:
                        prob_vec[i_d] *= 2.8
                        prob_vec[i_b] *= 0.25
                    elif waist_bar > 0.42 or topo["num_holes"] >= 2:
                        prob_vec[i_b] *= 1.8
                        prob_vec[i_d] *= 0.30

            if "C" in classes and "E" in classes and idx == 2:
                i_c, i_e = classes.index("C"), classes.index("E")
                if max(prob_vec[i_c], prob_vec[i_e]) >= 0.05 and topo["num_holes"] == 0:
                    mid_tongue = float(np.mean(item["bin_32"][13:19, 11:22] > 0))
                    if mid_tongue > 0.34:
                        prob_vec[i_e] *= 2.6
                        prob_vec[i_c] *= 0.30

            if "1" in classes and "7" in classes and (idx != 2):
                i_1, i_7 = classes.index("1"), classes.index("7")
                if max(prob_vec[i_1], prob_vec[i_7]) >= 0.22:
                    top_xs = np.where(np.any(item["bin_32"][2:8, :] > 0, axis=0))[0]
                    mid_ws = [np.sum(item["bin_32"][r, :] > 0) for r in range(12, 25) if np.sum(item["bin_32"][r, :] > 0) > 0]
                    w_top = float(top_xs[-1] - top_xs[0] + 1) if len(top_xs) > 0 else 0.0
                    w_mid = float(np.median(mid_ws)) if mid_ws else 1.0
                    if w_top >= 11.0 and (w_top / (w_mid + 1e-5)) >= 2.25:
                        prob_vec[i_7] = max(prob_vec[i_7] * 3.0, prob_vec[i_1] * 1.35)
                        prob_vec[i_1] *= 0.25
            best_i = int(np.argmax(prob_vec))
            conf = float(prob_vec[best_i] / (np.sum(prob_vec) + 1e-8))
            preds.append((classes[best_i], conf))
            
        plate_str = "".join(ch for ch, _ in preds)
        return plate_str, preds


def generate_synthetic_vietnamese_letters(
    letters: tuple[str, ...] = ("A", "R"),
    samples_per_letter: int = 20,
    seed: int = 42,
) -> tuple[list[dict], list[str]]:
    """
    Generate synthetic 32x32 character items (bin_32, area_32, gray_32) for Vietnamese license plate
    uppercase letters that are absent in the 115-plate dataset (such as 'A' and 'R').
    Uses condensed Vietnamese DIN-style geometry (width ~19-22, height ~42-48 before padding).
    """
    rng = np.random.default_rng(seed)
    items = []
    labels = []
    for ch in letters:
        for i in range(samples_per_letter):
            bh = int(rng.integers(42, 49))
            bw = int(rng.integers(18, 23))
            canvas = np.zeros((bh, bw), dtype=np.uint8)
            stroke = int(rng.choice([4, 5, 6, 7]))
            
            if ch == "A":
                top_w = int(rng.integers(3, 8))
                cx = bw // 2 + int(rng.integers(-1, 2))
                bar_y = int(bh * rng.uniform(0.60, 0.76))
                pts_left = np.array([[cx - top_w // 2, 2], [2, bh - 2]], dtype=np.int32)
                pts_right = np.array([[cx + top_w // 2, 2], [bw - 3, bh - 2]], dtype=np.int32)
                cv2.line(canvas, tuple(pts_left[0]), tuple(pts_left[1]), 255, stroke, cv2.LINE_AA)
                cv2.line(canvas, tuple(pts_right[0]), tuple(pts_right[1]), 255, stroke, cv2.LINE_AA)
                cv2.line(canvas, (2, bar_y), (bw - 3, bar_y), 255, max(3, stroke - 1), cv2.LINE_AA)
                apex_fill = int(bh * rng.uniform(0.22, 0.52))
                cv2.fillPoly(
                    canvas,
                    [np.array([[cx - top_w // 2, 2], [cx + top_w // 2, 2], [bw // 2 + 4, apex_fill], [bw // 2 - 4, apex_fill]])],
                    255,
                )
            else:
                tmp = np.zeros((64, 64), dtype=np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX if i % 2 == 0 else cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(tmp, ch, (8, 52), font, 1.8, 255, stroke + 1, cv2.LINE_AA)
                ys, xs = np.nonzero(tmp > 40)
                if len(xs) > 0:
                    crop = tmp[ys.min():ys.max()+1, xs.min():xs.max()+1]
                    canvas = cv2.resize(crop, (bw, bh), interpolation=cv2.INTER_AREA)

            angle = float(rng.uniform(-5.0, 5.0))
            M = cv2.getRotationMatrix2D((bw / 2.0, bh / 2.0), angle, 1.0)
            warped_bin = cv2.warpAffine(canvas, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0)
            _, p_bin = cv2.threshold(warped_bin, 110, 255, cv2.THRESH_BINARY)

            bg_val = float(rng.uniform(195.0, 248.0))
            fg_val = float(rng.uniform(25.0, 85.0))
            alpha = (warped_bin.astype(np.float32) / 255.0)
            p_gray = (bg_val * (1.0 - alpha) + fg_val * alpha)
            p_gray = cv2.GaussianBlur(p_gray, (3, 3), rng.uniform(0.5, 1.1))
            p_gray = np.clip(p_gray + rng.normal(0, 4.0, p_gray.shape), 0, 255).astype(np.uint8)

            pad_x = max(2, (bh - bw) // 3) if bw < bh * 0.45 else 2
            bin_pad = cv2.copyMakeBorder(p_bin, 2, 2, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
            gray_pad = cv2.copyMakeBorder(p_gray, 2, 2, pad_x, pad_x, cv2.BORDER_REPLICATE)

            bin_32 = cv2.resize(bin_pad, (32, 32), interpolation=cv2.INTER_NEAREST)
            area_32 = cv2.resize(bin_pad, (32, 32), interpolation=cv2.INTER_AREA)
            gray_32 = cv2.resize(gray_pad, (32, 32), interpolation=cv2.INTER_CUBIC)

            items.append({
                "pos_idx": 2,
                "row": 0,
                "bbox": (0, 0, bw, bh),
                "centroid": (bw / 2.0, bh / 2.0),
                "label_id": 1,
                "bin_32": bin_32,
                "area_32": area_32,
                "gray_32": gray_32,
            })
            labels.append(ch)
    return items, labels


def format_vietnamese_plate(raw_str: str) -> str:
    """
    Format an 8- or 9-character raw string (e.g. '59L206377' or '52U78693')
    into standard readable Vietnamese plate format ('59-L2 063.77' or '52-U7 8693').
    """
    if len(raw_str) == 9:
        return f"{raw_str[:2]}-{raw_str[2:4]} {raw_str[4:7]}.{raw_str[7:]}"
    elif len(raw_str) == 8:
        return f"{raw_str[:2]}-{raw_str[2:4]} {raw_str[4:]}"
    return raw_str
