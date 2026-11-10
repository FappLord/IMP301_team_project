"""
Module 1 (CORE THEME): License Plate Image Degradation & Restoration (Chapters 1, 2, 3, 4, 8 in Doc/)
=====================================================================================================
Strictly aligned with IMP302 Course Lecture Notes (`Doc/1..8.pdf`):

1. Chapter 1 (`1. Basic Gray-Level Image Processing.pdf`):
   - Full-Scale Histogram Stretch (Contrast Stretching, Section 1.1 P9)
   - Nonlinear Point Operations: Logarithmic & Power-Law Gamma (Section 1.1 P10)
   - Multi-sample Image Averaging for Noise Reduction (Section 1.1 P12-13)

2. Chapter 2 (`2. Image and Video Processing.pdf`):
   - Linear Spatial Smoothing: Box Filter & Lowpass Gaussian Filter (Section 2.1 & 2.4 P32-33)
   - Nonlinear Edge-Preserving Smoothing: Median Smoother & Weighted Median Filter (Section 2.2 P11-15)
   - Spatial Sharpening: Laplacian Sharpening, Unsharp Masking & Highboost Filtering (Section 2.4 P34-37)

3. Chapter 3 (`3. Image Degradation- Restoration and Resconstruction.pdf` - Core Chapter):
   - Section 3.1 & 3.4: Linear Position-Invariant Degradation Model g(x,y) = h(x,y)*f(x,y) + eta(x,y)
   - Section 3.2 (P6-12): All 6 Noise Probability Density Functions (PDFs):
     * Gaussian Noise, Rayleigh Noise, Erlang (Gamma) Noise, Exponential Noise,
       Uniform Noise, and Impulse (Salt-and-Pepper) Noise + Periodic Sinusoidal Noise
   - Section 3.3 (P13-22): Spatial Mean Filters & Frequency-Domain Notch Filtering:
     * Arithmetic Mean, Geometric Mean, Harmonic Mean, Contraharmonic Mean Filter (Order Q)
     * Butterworth Notch Reject Filter in 2D DFT domain for periodic noise removal
   - Section 3.5 (P30-31) & Chapter 4 Section 4.6 (P22-25):
     * 2D Discrete Wavelet Transform (DWT) Multi-Scale Denoising
   - Section 3.6, 3.7, 3.8 (P32-39): Deconvolution & Restoration:
     * Direct/Radially-Limited Inverse Filtering (Section 3.8 P39)
     * Minimum Mean Square Error (Wiener) Filtering (Section 3.6 P32-34)
     * Constrained Least Squares (CLS) Filtering with Laplacian Operator P(u,v) (Section 3.7 P35-38)

4. Chapter 8 (`8. Image and Video Rendering and Assessment.pdf`):
   - Section 8.1 & 8.3 (P3, P17-21): Distortion Criteria (MSE, SNR, PSNR) and
     Structural Similarity Index Measure (SSIM)
"""

import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


# =============================================================================
# CHAPTER 1: POINT OPERATIONS & FULL-SCALE HISTOGRAM STRETCH (Doc/1.pdf P6-13)
# =============================================================================

def full_scale_histogram_stretch(img: np.ndarray) -> np.ndarray:
    """
    Full-Scale Histogram Stretch / Contrast Stretching (Chapter 1, Section 1.1, Slide 9):
        g(x,y) = (2^B - 1) * (f(x,y) - min) / (max - min)
    Maps pixel intensities to span the full dynamic range [0, 255].
    """
    def _stretch_ch(ch: np.ndarray) -> np.ndarray:
        f = ch.astype(np.float32)
        f_min, f_max = float(np.min(f)), float(np.max(f))
        if f_max - f_min < 1e-5:
            return ch.copy()
        stretched = 255.0 * (f - f_min) / (f_max - f_min)
        return np.clip(stretched, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _stretch_ch(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _stretch_ch(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def logarithmic_point_operation(img: np.ndarray) -> np.ndarray:
    """
    Nonlinear Logarithmic Point Operation (Chapter 1, Section 1.1, Slide 10):
        g(x,y) = c * log(1 + |f(x,y)|), where c = 255 / log(1 + max(f))
    Expands dark gray levels while compressing bright highlights.
    """
    def _log_ch(ch: np.ndarray) -> np.ndarray:
        f = ch.astype(np.float32)
        c = 255.0 / np.log(1.0 + float(np.max(f)) + 1e-5)
        return np.clip(c * np.log(1.0 + f), 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _log_ch(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _log_ch(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def image_averaging_denoise(img: np.ndarray, K_frames: int = 8, sigma: float = 12.0, seed: int = 42) -> np.ndarray:
    """
    Image-Averaging for Noise Reduction (Chapter 1, Section 1.1, Slides 12-13):
        g_bar(x,y) = (1/K) * sum_{i=1}^K g_i(x,y)
        E{g_bar} = f(x,y), and sigma^2_{g_bar} = (1/K) * sigma^2_eta.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros_like(img, dtype=np.float32)
    for _ in range(K_frames):
        n = rng.normal(0.0, sigma, img.shape).astype(np.float32)
        acc += np.clip(img.astype(np.float32) + n, 0, 255)
    return np.clip(acc / float(K_frames), 0, 255).astype(np.uint8)


# =============================================================================
# CHAPTER 3.2: NOISE PROBABILITY DENSITY FUNCTIONS (Doc/3.pdf Slides 6-12)
# =============================================================================

def add_gaussian_noise(img: np.ndarray, sigma: float = 10.0, mean: float = 0.0, seed: int = 42) -> np.ndarray:
    """
    Gaussian Noise PDF (Chapter 3, Section 3.2, Slide 7):
        p(z) = (1 / (sqrt(2*pi)*sigma)) * exp(-(z - z_bar)^2 / (2*sigma^2))
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_rayleigh_noise(img: np.ndarray, a: float = -8.0, b: float = 160.0, seed: int = 42) -> np.ndarray:
    """
    Rayleigh Noise PDF (Chapter 3, Section 3.2, Slide 8):
        p(z) = (2/b)*(z - a)*exp(-(z - a)^2 / b) for z >= a, 0 for z < a
        Mean = a + sqrt(pi*b/4), Variance = b*(4 - pi)/4.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(1e-6, 1.0, img.shape).astype(np.float32)
    z = a + np.sqrt(-b * np.log(u))
    # Center zero-mean offset slightly so image isn't washed out
    z_centered = z - (a + np.sqrt(np.pi * b / 4.0))
    return np.clip(img.astype(np.float32) + z_centered, 0, 255).astype(np.uint8)


def add_erlang_gamma_noise(img: np.ndarray, a: float = 0.25, b: int = 3, seed: int = 42) -> np.ndarray:
    """
    Erlang (Gamma) Noise PDF (Chapter 3, Section 3.2, Slide 9):
        p(z) = (a^b * z^(b-1) / (b-1)!) * exp(-a*z) for z >= 0
        Mean = b / a, Variance = b / a^2.
    """
    rng = np.random.default_rng(seed)
    z = rng.gamma(shape=float(b), scale=1.0 / a, size=img.shape).astype(np.float32)
    z_centered = z - (float(b) / a)
    return np.clip(img.astype(np.float32) + z_centered, 0, 255).astype(np.uint8)


def add_uniform_noise(img: np.ndarray, a: float = -18.0, b: float = 18.0, seed: int = 42) -> np.ndarray:
    """
    Uniform Noise PDF (Chapter 3, Section 3.2, Slide 11):
        p(z) = 1 / (b - a) if a <= z <= b else 0.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(a, b, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + z, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(
    img: np.ndarray,
    prob: float = 0.04,
    pa: float | None = None,
    pb: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Impulse (Salt-and-Pepper) Noise PDF (Chapter 3, Section 3.2, Slide 12):
        p(z) = P_a for z = a (pepper=0), P_b for z = b (salt=255), 0 otherwise.
    """
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    p_pep = pa if pa is not None else (prob / 2.0)
    p_salt = pb if pb is not None else (prob / 2.0)
    u = rng.random(img.shape[:2])
    noisy[u < p_pep] = 0
    noisy[u > 1.0 - p_salt] = 255
    return noisy


def add_periodic_noise(
    img: np.ndarray,
    amplitude: float = 24.0,
    u0: int = 24,
    v0: int = 18,
    return_peaks: bool = False,
):
    """
    Periodic Sinusoidal Interference Noise (Chapter 3, Section 3.3, Slides 18-22):
        eta(x,y) = A * sin(2*pi*(u0*x/M + v0*y/N))
    Creates sharp conjugate impulse peaks at (+-u0, +-v0) in the 2D Fourier spectrum.
    """
    H, W = img.shape[:2]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    sinusoid = amplitude * np.sin(2.0 * np.pi * (u0 * yy / float(H) + v0 * xx / float(W))).astype(np.float32)
    if img.ndim == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0].astype(np.float32) + sinusoid, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        out = np.clip(img.astype(np.float32) + sinusoid, 0, 255).astype(np.uint8)
    if return_peaks:
        return out, [(u0, v0)]
    return out


def simulate_uneven_illumination(
    img: np.ndarray,
    shadow_strength: float = 0.55,
    strength: float | None = None,
    gamma: float = 1.30,
) -> np.ndarray:
    """
    Simulate uneven illumination (diagonal shadow + nonlinear gamma darkening).
    """
    s = strength if strength is not None else shadow_strength
    f = img.astype(np.float32) / 255.0
    f_gamma = np.power(f, gamma)
    h, w = img.shape[:2]
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    illum_map = 1.0 - s * (0.65 * xx + 0.35 * yy)
    if img.ndim == 3:
        illum_map = illum_map[:, :, None]
    return np.clip(f_gamma * illum_map * 255.0, 0, 255).astype(np.uint8)


# =============================================================================
# CHAPTER 2 & 3.3: SPATIAL MEAN, NONLINEAR & FREQUENCY NOTCH FILTERS
# =============================================================================

def arithmetic_mean_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Arithmetic Mean Filter (Chapter 3, Section 3.3, Slide 13):
        f_hat(x,y) = (1 / mn) * sum_{(s,t) in S_xy} g(s,t)
    """
    return cv2.blur(img, (ksize, ksize))


def geometric_mean_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Geometric Mean Filter (Chapter 3, Section 3.3, Slide 14):
        f_hat(x,y) = [ prod_{(s,t) in S_xy} g(s,t) ]^(1 / mn)
                 = exp( (1 / mn) * sum_{(s,t) in S_xy} ln(g(s,t)) )
    Achieves smoothing comparable to arithmetic mean but loses less image detail.
    """
    def _geo_ch(ch: np.ndarray) -> np.ndarray:
        log_f = np.log(np.maximum(ch.astype(np.float32), 1.0))
        mean_log = cv2.blur(log_f, (ksize, ksize))
        return np.clip(np.exp(mean_log), 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _geo_ch(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _geo_ch(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def harmonic_mean_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Harmonic Mean Filter (Chapter 3, Section 3.3, Slide 15):
        f_hat(x,y) = mn / sum_{(s,t) in S_xy} (1 / g(s,t))
    Works well for salt noise and Gaussian noise.
    """
    def _harm_ch(ch: np.ndarray) -> np.ndarray:
        inv_f = 1.0 / np.maximum(ch.astype(np.float32), 1.0)
        mean_inv = cv2.blur(inv_f, (ksize, ksize))
        return np.clip(1.0 / (mean_inv + 1e-8), 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _harm_ch(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _harm_ch(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def contraharmonic_mean_filter(img: np.ndarray, ksize: int = 3, Q: float = 1.5) -> np.ndarray:
    """
    Contraharmonic Mean Filter (Chapter 3, Section 3.3, Slide 16):
        f_hat(x,y) = ( sum_{(s,t) in S_xy} g(s,t)^(Q+1) ) / ( sum_{(s,t) in S_xy} g(s,t)^Q )
    Q > 0 eliminates pepper noise; Q < 0 eliminates salt noise; Q=0 is arithmetic mean; Q=-1 is harmonic mean.
    """
    def _chm_ch(ch: np.ndarray) -> np.ndarray:
        f = np.maximum(ch.astype(np.float32), 1.0)
        num = cv2.blur(np.power(f, Q + 1.0), (ksize, ksize))
        den = cv2.blur(np.power(f, Q), (ksize, ksize))
        return np.clip(num / (den + 1e-8), 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _chm_ch(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _chm_ch(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def median_denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Standard 2D Nonlinear Median Smoother (Chapter 2, Section 2.2, Slide 11).
    """
    return cv2.medianBlur(img, ksize)


def weighted_median_filter(img: np.ndarray, center_weight: int = 3) -> np.ndarray:
    """
    3x3 Center-Weighted Median Smoother (Chapter 2, Section 2.2, Slides 12-15):
        Weight mask W = [[1, 2, 1], [2, center_weight, 2], [1, 2, 1]]
    Duplicates pixels in the sliding window according to W before computing the median,
    preserving fine character strokes much better than a standard 3x3 median filter!
    """
    weights = np.array([[1, 2, 1], [2, int(center_weight), 2], [1, 2, 1]], dtype=int)

    def _wmed_ch(ch: np.ndarray) -> np.ndarray:
        pad = np.pad(ch, 1, mode="edge")
        H, W = ch.shape
        samples = []
        for dr in range(3):
            for dc in range(3):
                patch = pad[dr:dr + H, dc:dc + W]
                for _ in range(weights[dr, dc]):
                    samples.append(patch)
        stack = np.stack(samples, axis=0)
        return np.median(stack, axis=0).astype(np.uint8)

    if img.ndim == 2:
        return _wmed_ch(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _wmed_ch(ycrcb[:, :, 0])
    ycrcb[:, :, 1] = cv2.medianBlur(ycrcb[:, :, 1], 3)
    ycrcb[:, :, 2] = cv2.medianBlur(ycrcb[:, :, 2], 3)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def unsharp_masking_highboost(
    img: np.ndarray,
    k: float = 1.2,
    k_boost: float | None = None,
    sigma: float = 1.2,
) -> np.ndarray:
    """
    Unsharp Masking and Highboost Filtering (Chapter 2, Section 2.4, Slide 37):
        1. Blur original image: f_bar(x,y) = G_sigma * f(x,y)
        2. Subtract blurred from original (mask): g_mask(x,y) = f(x,y) - f_bar(x,y)
        3. Add weighted mask back: g(x,y) = f(x,y) + k * g_mask(x,y)
           (k = 1: Unsharp masking; k > 1: Highboost filtering).
    """
    gain = k_boost if k_boost is not None else k
    blur = cv2.GaussianBlur(img, (5, 5), sigma)
    f = img.astype(np.float32)
    g_mask = f - blur.astype(np.float32)
    return np.clip(f + gain * g_mask, 0, 255).astype(np.uint8)


def butterworth_notch_reject_filter(
    img: np.ndarray,
    notch_centers: list[tuple[int, int]] = [(24, 18)],
    notch_Step_uv: list[tuple[int, int]] | None = None,
    D0: float = 6.0,
    d0: float | None = None,
    n_order: int = 2,
    order: int | None = None,
):
    """
    Frequency-Domain Butterworth Notch Reject Filter (Chapter 3, Section 3.3, Slides 18-22):
        H_NR(u,v) = prod_{k=1}^Q [ 1 / (1 + (D0k / D_k(u,v))^(2n)) ] * [ 1 / (1 + (D0k / D_{-k}(u,v))^(2n)) ]
    Eliminates periodic sinusoidal interference by placing symmetric notches at (+-u_k, +-v_k).
    """
    centers = notch_Step_uv if notch_Step_uv is not None else notch_centers
    radius = d0 if d0 is not None else D0
    n_ord = order if order is not None else n_order

    def _notch_ch(ch: np.ndarray):
        H, W = ch.shape
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        cy, cx = H // 2, W // 2
        H_nr = np.ones((H, W), dtype=np.float32)
        for uk, vk in centers:
            Dk = np.sqrt((yy - (cy + uk)) ** 2 + (xx - (cx + vk)) ** 2) + 1e-6
            D_mk = np.sqrt((yy - (cy - uk)) ** 2 + (xx - (cx - vk)) ** 2) + 1e-6
            H_k = (1.0 / (1.0 + (radius / Dk) ** (2 * n_ord))) * (1.0 / (1.0 + (radius / D_mk) ** (2 * n_ord)))
            H_nr *= H_k
        F_shift = fftshift(fft2(ch.astype(np.float32)))
        G_shift = F_shift * H_nr
        spec_before = np.log1p(np.abs(F_shift)).astype(np.float32)
        spec_after = np.log1p(np.abs(G_shift)).astype(np.float32)
        rec = np.real(ifft2(ifftshift(G_shift)))
        return np.clip(rec, 0, 255).astype(np.uint8), spec_before, spec_after

    if img.ndim == 2:
        out, sb, sa = _notch_ch(img)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_out, sb, sa = _notch_ch(ycrcb[:, :, 0])
        ycrcb[:, :, 0] = y_out
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    if notch_Step_uv is not None or d0 is not None or order is not None:
        return out, sb, sa
    return out, sb


# =============================================================================
# CHAPTER 3.4, 3.6, 3.7, 3.8: MOTION BLUR PSF, INVERSE, WIENER & CLS FILTERS
# =============================================================================

def motion_blur_psf(length: int = 7, angle: float = 12.0, angle_deg: float | None = None) -> np.ndarray:
    """
    Generate a 2D Linear Motion Blur Point Spread Function h(x,y) (Chapter 3, Section 3.4).
    """
    ang = angle_deg if angle_deg is not None else angle
    length = max(1, int(length))
    if length == 1:
        return np.ones((1, 1), dtype=np.float32)
    size = length if length % 2 == 1 else length + 1
    psf = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    rad = np.deg2rad(ang)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    half = length / 2.0
    for t in np.linspace(-half, half, length * 8):
        x = center + t * cos_a
        y = center - t * sin_a
        ix, iy = int(np.round(x)), int(np.round(y))
        if 0 <= ix < size and 0 <= iy < size:
            psf[iy, ix] += 1.0
    total = np.sum(psf)
    if total > 0:
        psf /= total
    else:
        psf[center, center] = 1.0
    return psf


def apply_motion_blur(img: np.ndarray, length: int = 7, angle: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """Convolve image with linear motion blur PSF."""
    psf = motion_blur_psf(length, angle)
    blurred = cv2.filter2D(img, -1, psf, borderType=cv2.BORDER_REPLICATE)
    return blurred, psf


def synthesize_degraded_image(
    img: np.ndarray,
    blur_length: int = 7,
    blur_len: int | None = None,
    blur_angle: float = 12.0,
    noise_sigma: float = 7.0,
    darken_gamma: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear Position-Invariant Degradation Model (Chapter 3, Section 3.1 & 3.4):
        g(x,y) = h(x,y) * f(x,y) + eta(x,y)
    """
    b_len = blur_len if blur_len is not None else blur_length
    f = img.astype(np.float32) / 255.0
    if darken_gamma != 1.0:
        f = np.power(f, darken_gamma)
    base = np.clip(f * 255.0, 0, 255).astype(np.uint8)
    blurred, psf = apply_motion_blur(base, length=b_len, angle=blur_angle)
    degraded = add_gaussian_noise(blurred, sigma=noise_sigma, seed=seed)
    return degraded, psf


def inverse_filter_deconvolution(
    img: np.ndarray,
    psf: np.ndarray,
    eps_cutoff: float = 0.08,
    eps: float | None = None,
) -> np.ndarray:
    """
    Pseudo-Inverse / Radially-Limited Inverse Filter (Chapter 3, Section 3.8, Slide 39):
        F_hat(u,v) = G(u,v) / H(u,v) if |H(u,v)| >= eps_cutoff else 0
    Demonstrates noise amplification (N(u,v)/H(u,v)) when |H(u,v)| is small.
    """
    cutoff = eps if eps is not None else eps_cutoff
    def _inv_ch(channel: np.ndarray) -> np.ndarray:
        pad_h, pad_w = psf.shape[0] * 2, psf.shape[1] * 2
        g_pad = np.pad(channel.astype(np.float32) / 255.0, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        H_full, W_full = g_pad.shape
        h_otf = np.zeros((H_full, W_full), dtype=np.float32)
        kh, kw = psf.shape
        h_otf[:kh, :kw] = psf
        h_otf = np.roll(np.roll(h_otf, -(kh // 2), axis=0), -(kw // 2), axis=1)
        G_fft = fft2(g_pad)
        H_fft = fft2(h_otf)
        H_mag = np.abs(H_fft)
        inv_f = np.where(H_mag >= cutoff, 1.0 / (H_fft + 1e-8), 0.0)
        f_rec = np.real(ifft2(G_fft * inv_f))[pad_h:H_full - pad_h, pad_w:W_full - pad_w]
        return np.clip(f_rec * 255.0, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _inv_ch(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _inv_ch(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def constrained_least_squares_filter(img: np.ndarray, psf: np.ndarray, gamma: float = 0.015) -> np.ndarray:
    """
    Constrained Least Squares (CLS) Filtering (Chapter 3, Section 3.7, Slides 35-38):
        F_hat(u,v) = [ H*(u,v) / (|H(u,v)|^2 + gamma * |P(u,v)|^2) ] * G(u,v)
    where p(x,y) = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]] is the 2D Laplacian operator.
    Unlike standard Wiener filtering, CLS only requires knowledge of noise variance/mean
    and uses the smoothness constraint ||p * f_hat||^2.
    """
    def _cls_ch(channel: np.ndarray) -> np.ndarray:
        pad_h, pad_w = psf.shape[0] * 2, psf.shape[1] * 2
        g_pad = np.pad(channel.astype(np.float32) / 255.0, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        H_full, W_full = g_pad.shape
        h_otf = np.zeros((H_full, W_full), dtype=np.float32)
        kh, kw = psf.shape
        h_otf[:kh, :kw] = psf
        h_otf = np.roll(np.roll(h_otf, -(kh // 2), axis=0), -(kw // 2), axis=1)

        p_lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        p_otf = np.zeros((H_full, W_full), dtype=np.float32)
        p_otf[:3, :3] = p_lap
        p_otf = np.roll(np.roll(p_otf, -1, axis=0), -1, axis=1)

        G_fft = fft2(g_pad)
        H_fft = fft2(h_otf)
        P_fft = fft2(p_otf)
        W_cls = np.conj(H_fft) / (np.abs(H_fft) ** 2 + gamma * (np.abs(P_fft) ** 2) + 1e-8)
        f_rec = np.real(ifft2(W_cls * G_fft))[pad_h:H_full - pad_h, pad_w:W_full - pad_w]
        return np.clip(f_rec * 255.0, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _cls_ch(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _cls_ch(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _wiener_channel(channel: np.ndarray, psf: np.ndarray, K: float = 0.02, gamma_cls: float = 0.045) -> np.ndarray:
    """
    Minimum Mean Square Error (Wiener) + CLS Hybrid Deconvolution (Chapter 3, Sections 3.6 & 3.7):
        F_hat(u,v) = [ (1 + K) * H*(u,v) / (|H(u,v)|^2 + K + gamma * |P(u,v)|^2) ] * G(u,v)
    """
    pad_h, pad_w = psf.shape[0] * 2, psf.shape[1] * 2
    g_pad = np.pad(channel.astype(np.float32) / 255.0, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    H_full, W_full = g_pad.shape
    
    h_otf = np.zeros((H_full, W_full), dtype=np.float32)
    kh, kw = psf.shape
    h_otf[:kh, :kw] = psf
    h_otf = np.roll(np.roll(h_otf, -(kh // 2), axis=0), -(kw // 2), axis=1)
    
    p_lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    p_otf = np.zeros((H_full, W_full), dtype=np.float32)
    p_otf[:3, :3] = p_lap
    p_otf = np.roll(np.roll(p_otf, -1, axis=0), -1, axis=1)
    
    G_fft = fft2(g_pad)
    H_fft = fft2(h_otf)
    P_fft = fft2(p_otf)
    
    H_conj = np.conj(H_fft)
    H_mag2 = np.abs(H_fft) ** 2
    P_mag2 = np.abs(P_fft) ** 2
    wiener_filter = (1.0 + K) * H_conj / (H_mag2 + K + gamma_cls * P_mag2)
    
    F_hat = wiener_filter * G_fft
    f_rec = np.real(ifft2(F_hat))
    f_crop = f_rec[pad_h:H_full - pad_h, pad_w:W_full - pad_w]
    return np.clip(f_crop * 255.0, 0, 255).astype(np.uint8)


def wiener_deconvolution(img: np.ndarray, psf: np.ndarray, K: float = 0.02, gamma_cls: float = 0.045) -> np.ndarray:
    """
    Wiener Deconvolution for grayscale or BGR color images.
    """
    if img.ndim == 2:
        return _wiener_channel(img, psf, K=K, gamma_cls=gamma_cls)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _wiener_channel(ycrcb[:, :, 0], psf, K=K, gamma_cls=gamma_cls)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# =============================================================================
# CHAPTER 3.5 & CHAPTER 4.6: 2D WAVELET DWT DENOISING (Doc/3.pdf P30-31, Doc/4.pdf P22-25)
# =============================================================================

def dwt2_haar(img_2d: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Single-level 2D Discrete Wavelet Transform (Haar orthonormal basis, Chapter 4 Section 4.6).
    Returns (LL, (LH, HL, HH)).
    """
    H, W = img_2d.shape
    pad_r = H % 2
    pad_c = W % 2
    if pad_r or pad_c:
        x = np.pad(img_2d, ((0, pad_r), (0, pad_c)), mode="symmetric").astype(np.float32)
    else:
        x = img_2d.astype(np.float32)
        
    s = np.sqrt(2.0)
    L_rows = (x[:, 0::2] + x[:, 1::2]) / s
    H_rows = (x[:, 0::2] - x[:, 1::2]) / s
    
    LL = (L_rows[0::2, :] + L_rows[1::2, :]) / s
    LH = (L_rows[0::2, :] - L_rows[1::2, :]) / s
    HL = (H_rows[0::2, :] + H_rows[1::2, :]) / s
    HH = (H_rows[0::2, :] - H_rows[1::2, :]) / s
    return LL, (LH, HL, HH)


def idwt2_haar(LL: np.ndarray, details: tuple[np.ndarray, np.ndarray, np.ndarray], target_shape: tuple[int, int]) -> np.ndarray:
    """
    Single-level 2D Inverse Discrete Wavelet Transform (Chapter 4 Section 4.6).
    """
    LH, HL, HH = details
    s = np.sqrt(2.0)
    h_sub, w_sub = LL.shape
    
    L_rows = np.zeros((h_sub * 2, w_sub), dtype=np.float32)
    H_rows = np.zeros((h_sub * 2, w_sub), dtype=np.float32)
    
    L_rows[0::2, :] = (LL + LH) / s
    L_rows[1::2, :] = (LL - LH) / s
    H_rows[0::2, :] = (HL + HH) / s
    H_rows[1::2, :] = (HL - HH) / s
    
    out = np.zeros((h_sub * 2, w_sub * 2), dtype=np.float32)
    out[:, 0::2] = (L_rows + H_rows) / s
    out[:, 1::2] = (L_rows - H_rows) / s
    return out[:target_shape[0], :target_shape[1]]


def _shrink_subband(subband: np.ndarray, sigma_noise: float, mode: str = "soft", method: str = "bayesshrink") -> np.ndarray:
    if method == "visushrink":
        T = sigma_noise * np.sqrt(2.0 * np.log(max(subband.size, 2))) * 0.55
    else:
        var_y = float(np.mean(subband ** 2))
        sigma_x = np.sqrt(max(var_y - sigma_noise ** 2, 0.0))
        if sigma_x < 1e-6:
            return np.zeros_like(subband)
        T = (sigma_noise ** 2) / sigma_x
        
    if mode == "hard":
        return np.where(np.abs(subband) >= T, subband, 0.0)
    return np.sign(subband) * np.maximum(np.abs(subband) - T, 0.0)


def wavelet_denoise(
    img: np.ndarray,
    levels: int = 1,
    mode: str = "soft",
    method: str = "bayesshrink",
    sigma_est: float | None = None,
    return_info: bool = False,
):
    """
    Wavelet Denoising for Image Enhancement (Chapter 3 Section 3.5, Slides 30-31):
    1. Decompose image into multiple wavelet levels (LL, LH, HL, HH)
    2. Threshold high-frequency detail coefficients (LH, HL, HH)
    3. Reconstruct image via Inverse DWT
    """
    def _denoise_2d(ch: np.ndarray) -> tuple[np.ndarray, dict]:
        coeffs = []
        shapes = []
        curr = ch.astype(np.float32)
        for _ in range(levels):
            shapes.append(curr.shape)
            LL, details = dwt2_haar(curr)
            coeffs.append(details)
            curr = LL
            
        HH1 = coeffs[0][2]
        sigma_hat = float(sigma_est) if sigma_est is not None else float(np.median(np.abs(HH1)) / 0.6745)
        
        thresh_coeffs = []
        for lvl, (LH, HL, HH) in enumerate(coeffs):
            scale_factor = 1.0 / (1.2 ** lvl)
            s_lvl = sigma_hat * scale_factor
            t_LH = _shrink_subband(LH, s_lvl, mode=mode, method=method)
            t_HL = _shrink_subband(HL, s_lvl, mode=mode, method=method)
            t_HH = _shrink_subband(HH, s_lvl, mode=mode, method=method)
            thresh_coeffs.append((t_LH, t_HL, t_HH))
            
        rec = curr
        for lvl in reversed(range(levels)):
            rec = idwt2_haar(rec, thresh_coeffs[lvl], shapes[lvl])
            
        info = {
            "sigma_hat": sigma_hat,
            "LL": curr,
            "level1_raw": coeffs[0],
            "level1_denoised": thresh_coeffs[0],
        }
        return np.clip(rec, 0, 255).astype(np.uint8), info

    if img.ndim == 2:
        out, info = _denoise_2d(img)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_den, info = _denoise_2d(ycrcb[:, :, 0])
        ycrcb[:, :, 0] = y_den
        ycrcb[:, :, 1] = cv2.medianBlur(ycrcb[:, :, 1], 3)
        ycrcb[:, :, 2] = cv2.medianBlur(ycrcb[:, :, 2], 3)
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    if return_info:
        return out, info
    return out


def spatial_denoise(img: np.ndarray, filter_type: str = "bilateral") -> np.ndarray:
    """
    Spatial domain smoothing filters (Chapter 2 & Chapter 3.3):
    - 'box' / 'arithmetic': 3x3 Arithmetic Mean Filter
    - 'geometric': 3x3 Geometric Mean Filter
    - 'harmonic': 3x3 Harmonic Mean Filter
    - 'contraharmonic': 3x3 Contraharmonic Mean Filter
    - 'gaussian': 5x5 Gaussian Lowpass Filter
    - 'median': 3x3 Median Smoother
    - 'weighted_median': 3x3 Center-Weighted Median Filter (Ch. 2 P12-15)
    - 'bilateral': Edge-preserving Bilateral Filter
    """
    if filter_type in ("box", "arithmetic"):
        return arithmetic_mean_filter(img, 3)
    elif filter_type == "geometric":
        return geometric_mean_filter(img, 3)
    elif filter_type == "harmonic":
        return harmonic_mean_filter(img, 3)
    elif filter_type == "contraharmonic":
        return contraharmonic_mean_filter(img, 3, Q=0.5)
    elif filter_type == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 1.2)
    elif filter_type == "median":
        return cv2.medianBlur(img, 3)
    elif filter_type == "weighted_median":
        return weighted_median_filter(img)
    elif filter_type == "bilateral":
        return cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)
    raise ValueError(f"Unknown filter_type: {filter_type}")


def enhance_contrast(img: np.ndarray, method: str = "clahe", clip_limit: float = 2.0, tile_grid: tuple[int, int] = (6, 6)) -> np.ndarray:
    """
    Gray-level contrast enhancement (Chapter 1 Section 1.1):
    - 'stretch': Full-Scale Histogram Stretch
    - 'log': Logarithmic Point Operation
    - 'he': Global Histogram Equalization
    - 'clahe': Contrast Limited Adaptive Histogram Equalization
    - 'gamma': Power-law Gamma Correction
    """
    if method == "stretch":
        return full_scale_histogram_stretch(img)
    elif method == "log":
        return logarithmic_point_operation(img)
    if img.ndim == 2:
        if method == "he":
            return cv2.equalizeHist(img)
        elif method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
            return clahe.apply(img)
        elif method == "gamma":
            lut = np.array([((i / 255.0) ** 0.8) * 255.0 for i in range(256)], dtype=np.uint8)
            return cv2.LUT(img, lut)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = enhance_contrast(ycrcb[:, :, 0], method=method, clip_limit=clip_limit, tile_grid=tile_grid)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    raise ValueError(f"Unknown contrast method: {method}")


def full_restoration_pipeline(
    degraded_img: np.ndarray,
    psf: np.ndarray | None = None,
    wiener_K: float = 0.008,
    gamma_cls: float = 0.005,
    use_wavelet: bool = True,
    use_clahe: bool = False,
    degradation_type: str = "motion_awgn",
) -> tuple[np.ndarray, dict]:
    """
    Unified Adaptive Restoration Pipeline (Chapter 3):
    Handles diverse degradation modes:
    - 'motion_awgn': 2D Wavelet DWT Shrinkage -> Wiener/CLS Deconvolution
    - 'salt_pepper': 3x3 Weighted Median Filter -> Unsharp Masking
    - 'periodic': Frequency-Domain Butterworth Notch Reject Filter
    - 'rayleigh' / 'erlang' / 'awgn': 2D Wavelet DWT + Full-Scale Histogram Stretch
    """
    stages = {"input": degraded_img.copy()}
    curr = degraded_img.copy()
    dwt_info = {}

    if degradation_type == "salt_pepper":
        curr = weighted_median_filter(curr)
        stages["wavelet_denoised"] = curr.copy()
        stages["after_wavelet"] = curr.copy()
        curr = unsharp_masking_highboost(curr, k=0.4, sigma=1.0)
        stages["after_wiener"] = curr.copy()
    elif degradation_type == "periodic":
        curr, notch_mask = butterworth_notch_reject_filter(curr, notch_centers=[(24, 18)], D0=6.0, n_order=2)
        dwt_info["notch_mask"] = notch_mask
        stages["wavelet_denoised"] = curr.copy()
        stages["after_wavelet"] = curr.copy()
        stages["after_wiener"] = curr.copy()
    else:
        if use_wavelet:
            curr, dwt_info = wavelet_denoise(curr, levels=1, mode="soft", method="bayesshrink", return_info=True)
        stages["wavelet_denoised"] = curr.copy()
        stages["after_wavelet"] = curr.copy()
        if psf is not None and psf.size > 1:
            curr = wiener_deconvolution(curr, psf, K=wiener_K, gamma_cls=gamma_cls)
        stages["after_wiener"] = curr.copy()

    if use_clahe:
        curr = enhance_contrast(curr, method="clahe", clip_limit=1.4, tile_grid=(6, 6))
    stages["restored"] = curr.copy()
    stages["dwt_info"] = dwt_info
    return curr, stages


# =============================================================================
# CHAPTER 8: OBJECTIVE IMAGE QUALITY ASSESSMENT (Doc/8.pdf Slides 3, 17-21)
# =============================================================================

def compute_mse(ref: np.ndarray, target: np.ndarray) -> float:
    """Mean Squared Error (Chapter 8, Section 8.1 & 8.3)."""
    r = ref.astype(np.float64)
    t = target.astype(np.float64)
    return float(np.mean((r - t) ** 2))


def compute_snr(ref: np.ndarray, target: np.ndarray) -> float:
    """Signal-to-Noise Ratio (dB, Chapter 8 Section 8.1 Slide 3)."""
    r = ref.astype(np.float64)
    t = target.astype(np.float64)
    sig_pow = float(np.mean(r ** 2))
    noise_pow = float(np.mean((r - t) ** 2))
    if noise_pow < 1e-10:
        return 100.0
    return float(10.0 * np.log10((sig_pow + 1e-12) / noise_pow))


def compute_psnr(ref: np.ndarray, target: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = compute_mse(ref, target)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_ssim(ref: np.ndarray, target: np.ndarray) -> float:
    """
    Structural Similarity Index Measure (Chapter 8, Section 8.3, Slides 18-21):
        SSIM(x,y) = [l(x,y)]^alpha * [c(x,y)]^beta * [s(x,y)]^gamma
                  = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1)*(sigma_x^2 + sigma_y^2 + C2))
    """
    if ref.ndim == 3:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    if target.ndim == 3:
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        
    x = ref.astype(np.float64)
    y = target.astype(np.float64)
    
    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2
    
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    sigma_x2 = cv2.GaussianBlur(x ** 2, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y ** 2, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
    
    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map))


def evaluate_quality(ref: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return dictionary of MSE, SNR, PSNR, and SSIM (both lowercase and uppercase keys)."""
    mse = compute_mse(ref, target)
    snr = compute_snr(ref, target)
    psnr = compute_psnr(ref, target)
    ssim = compute_ssim(ref, target)
    return {
        "mse": mse,
        "snr": snr,
        "psnr": psnr,
        "ssim": ssim,
        "MSE": mse,
        "SNR": snr,
        "PSNR": psnr,
        "SSIM": ssim,
    }
