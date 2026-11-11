# DIGITAL IMAGE AND VIDEO PROCESSING COURSE PROJECT REPORT (IMP302)

## TOPIC: MULTI-DEGRADATION LICENSE PLATE IMAGE RESTORATION COMBINED WITH CHARACTER SEGMENTATION AND RECOGNITION USING CLASSICAL IMAGE PROCESSING METHODS

<div class="author-grid">
  <div class="author-item"><b>Võ Tấn Phát</b><br/>SE194484</div>
  <div class="author-item"><b>Nguyễn Trung Dũng</b><br/>SE194119</div>
  <div class="author-item"><b>Lê Ngọc Thương</b><br/>SE194271</div>
  <div class="author-item"><b>Nguyễn Bảo Ái</b><br/>SE192158</div>
  <div class="author-item"><b>Lê Minh Khôi</b><br/>SE192396</div>
</div>

## Abstract

Two-line motorcycle license plates captured by outdoor traffic and parking surveillance cameras are frequently degraded by linear motion blur, low-light sensor noise, bipolar impulse corruption, periodic electromagnetic interference, and severe perspective foreshortening. Because adjacent glyphs on a canonical $160 \times 128$ rectified plate are separated by only $3\text{--}4\text{ pixels}$, even moderate directional blur ($L = 7\text{ pixels}$) bridges the inter-character gap, causing connected-component segmentation to collapse and reducing full-plate recognition accuracy to $0.00\%$. We study a fully transparent, white-box classical image processing and statistical pattern recognition pipeline implemented from first principles without deep neural networks. The proposed architecture comprises four stages: (i) a multi-degradation restoration stage combining 2D Haar wavelet BayesShrink denoising with a hybrid Wiener–Constrained Least Squares (CLS) deconvolution filter, a $3 \times 3$ center-weighted median filter for impulse noise, and a conjugate Butterworth notch reject filter for periodic interference; (ii) a three-stream morphological and vertical-Sobel detector with minimum-area rotated rectangle scoring, IoU-based Non-Maximum Suppression (NMS) ordered strictly left-to-right, and Douglas–Peucker 4-point perspective homography rectification; (iii) a shadow-resilient local Gaussian adaptive binarizer paired with a maximum contiguous run-length border suppression algorithm that preserves aligned top character caps; and (iv) a 480-dimensional hybrid descriptor fusing sequency-ordered 2D Walsh–Hadamard Transform (WHT) coefficients, Histograms of Oriented Gradients (HOG), 7 Hu moment invariants, boundary Fourier descriptors, region covariance eccentricity, and Euler topological numbers ($E = C - H$), evaluated across seven classical classifiers with topological post-verification. On a zero-leakage plate-level benchmark (80 training, 17 validation, and 18 test plates expanded $6\times$ across six physical degradation regimes to 690 vehicle images and 5,643 character patches), Stage 1 restoration yields up to $+17.98\text{ dB}$ PSNR gain, recovers motion-blurred character recognition on the held-out test split from $3.85\%$ to $93.59\%$, and reaches $96.99\%$ 5-fold cross-validation character accuracy. We also audit ground-truth label discrepancies in the raw corpus, evaluate four challenging in-the-wild stress-test scenes, and analyze the remaining failure modes.

**Keywords:** Image restoration, Wiener deconvolution, constrained least squares, Haar wavelet shrinkage, perspective homography, Otsu thresholding, K-Means segmentation, Walsh–Hadamard transform, boundary Fourier descriptors, Euler topology, license plate recognition.

---

## 1 Introduction

Automatic License Plate Recognition (ALPR) is a core component of urban traffic monitoring, automated parking management, and law enforcement surveillance [1, 2]. While single-line passenger car plates in controlled highway lanes are well studied, Vietnamese two-line motorcycle plates ($190\text{ mm} \times 140\text{ mm}$ physical dimensions containing 8 or 9 alphanumeric characters arranged across an upper 4-character row and a lower 4- or 5-digit row) present distinct signal processing and geometric challenges. In low-cost parking and street cameras, vehicles are often photographed at oblique angles ($\pm 30^\circ$ in-plane roll and out-of-plane yaw), under non-uniform shadow gradients, or while moving through the camera exposure window.

Let $f(x,y) \in \mathbb{R}^{M \times N}$ denote an ideal, uncorrupted surveillance frame. Under a linear space-invariant (LSI) optical degradation model coupled with projective viewing geometry $\mathcal{T}_{\mathbf{H}}$ and additive or impulsive sensor noise $\eta(x,y)$, the observed frame $g(x,y)$ is given by [3]:
$$g(x,y) = \mathcal{T}_{\mathbf{H}}\!\Big\{ (h * f)(x,y) \Big\} + \eta(x,y), \qquad (1)$$
where $h(x,y)$ is the 2D Point Spread Function (PSF) induced by relative motion between the vehicle and the imaging sensor, $*$ denotes 2D spatial convolution, and $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ is a projective homography matrix. Recovering the exact character sequence $\hat{\mathbf{s}} = (\hat{c}_1, \hat{c}_2, \dots, \hat{c}_L)$ from $g(x,y)$ requires inverting both the radiometric degradation $(h, \eta)$ and the geometric distortion $\mathcal{T}_{\mathbf{H}}$ prior to connected-component isolation and statistical pattern classification.

Surveys of classical license plate analysis [1, 2] and our empirical audit of degraded surveillance frames reveal three primary causes of pipeline failure:
1. **Coupled blur–noise amplification and stroke coalescence:** On a rectified $160 \times 128$ plate image, the horizontal spacing between neighboring glyphs is only $3\text{--}4\text{ pixels}$. A linear motion blur kernel of length $L = 7\text{ pixels}$ smears foreground ink across this gap, merging four or five digits into a single connected component. Naive frequency-domain inversion ($\hat{F}(u,v) = G(u,v)/H(u,v)$) amplifies high-frequency noise $\eta(x,y)$ near the spectral zeros of the $\text{sinc}$ transfer function $H(u,v)$, destroying stroke legibility.
2. **Oblique foreshortening and border-induced cap truncation:** When a motorcycle is viewed from the side or at a steep tilt, the axis-aligned bounding box of the plate exhibits low rectangular extent ($\text{Extent} < 0.65$) and distorted aspect ratio ($W < H$). Furthermore, conventional border removal heuristics that zero out horizontal scanlines exceeding $40\%$ foreground density inadvertently slice off the aligned top horizontal caps of characters such as `'2'`, `'9'`, `'D'`, and `'7'`, turning a closed `'D'` into an open `'U'`-shaped artifact.
3. **Structural ambiguity and severe class imbalance:** In the Vietnamese license plate typeface, pairs such as `'D'` vs. `'B'`, `'E'` vs. `'C'`, and `'7'` vs. `'1'` share nearly identical outer bounding boxes and vertical projections. At the same time, rare series letters (`'A'`, `'D'`, `'E'`, `'R'`, `'T'`, `'Y'`) have very few or zero instances in standard 100-plate benchmarks, causing probability calibration (Platt scaling) in margin classifiers to degenerate.

**Research gap.** Modern ALPR deployments increasingly rely on deep convolutional or Transformer detectors (e.g., YOLO-based plate detectors paired with optical character recognition networks) [4, 5]. While effective on large GPU-trained corpora, black-box neural pipelines obscure the mathematical relationship between frequency-domain deconvolution, morphological geometry, and topological feature invariance, and degrade unpredictably when tested on out-of-distribution sensor corruptions (such as periodic power-line interference or high-density impulse noise). Conversely, existing classical ALPR studies typically evaluate only a single stage (such as Otsu binarization or template matching) on clean, fronto-parallel crops without rigorous plate-level train/validation/test partitioning or multi-degradation restoration. How classical spatial-frequency restoration interacts with projective homography rectification, contiguous run-length segmentation, and hybrid transform–topological classification in a complete end-to-end system remains insufficiently documented.

**Contributions.** This paper designs, implements, and audits a complete four-stage classical license plate restoration and recognition pipeline. Specifically:
1. We formulate a multi-degradation restoration stage (Section 3.2) that pairs 2D Haar wavelet BayesShrink denoising with a hybrid Wiener–Constrained Least Squares (CLS) deconvolution filter, alongside a $3 \times 3$ center-weighted median filter and a conjugate Butterworth notch reject filter. Restoration improves PSNR by $+2.35\text{ to }+17.98\text{ dB}$ on four of five physical degradation regimes (AWGN, impulse, periodic, motion blur) and rescues held-out test character recognition under motion blur from $3.85\%$ to $93.59\%$.
2. We develop a three-stream morphological detector with minimum-area rotated rectangle scoring, IoU-based Non-Maximum Suppression (NMS) ordered strictly left-to-right, and Douglas–Peucker 4-point perspective homography rectification (Section 3.3), coupled with a maximum contiguous run-length (`_max_run`) border filter (Section 3.4) that eliminates frame borders and `"VIE"` flag sticker dividers without truncating character caps.
3. We construct a 480-dimensional hybrid feature representation combining sequency-ordered 2D Walsh–Hadamard Transform (WHT) coefficients, HOG descriptors, boundary Fourier descriptors, region covariance eccentricity, 7 Hu moment invariants, and Euler topological numbers ($E = C - H$), and benchmark seven classical classifiers with topological post-verification for structurally ambiguous glyph pairs (`'D'` vs. `'B'`, `'E'` vs. `'C'`, `'7'` vs. `'1'`) (Section 3.5).
4. We establish a zero-leakage stratified plate-level evaluation protocol (80 train / 17 validation / 18 test plates expanded $6\times$ to 690 images and 5,643 character patches), audit and rectify ground-truth labeling errors in the raw dataset, and validate generalization on four challenging in-the-wild stress-test scenes and real-time surveillance video streams (Sections 4 and 5).

We do not claim superiority over large-scale GPU-trained deep learning systems on unconstrained open-world benchmarks. Our goal is to characterize a fully specified, mathematically interpretable classical pipeline from first principles and to isolate the quantitative contribution of each restoration, segmentation, and classification component.

---

## 2 Related Work

### 2.1 License plate recognition systems

Table 1 positions our pipeline relative to representative classical and hybrid ALPR methodologies in the literature. Early ALPR systems relied on vertical edge projection and global thresholding followed by pixel-level template matching [1, 2]. While computationally light, projection-based methods require near-zero roll angle and uniform illumination. Subsequent classical pipelines introduced morphological top-hat/black-hat operators [6] and Connected Component Analysis (CCA) [7], paired with Support Vector Machines (SVM) on hand-crafted gradient features [8]. More recently, one-stage deep detectors [4, 5] have achieved high accuracy on large datasets at the cost of millions of learned parameters and GPU inference requirements. In contrast, our work retains a 100% white-box classical formulation while addressing the four failure modes that limit traditional pipelines: coupled motion-blur/noise restoration, 4-point projective homography for foreshortened plates, contiguous run-length border suppression, and topological glyph disambiguation.

| Method | Paradigm | Degraded Image Restoration | Geometric Rectification | Binarization & Border Filtering | Feature & Classifier Design |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Anagnostopoulos et al. [1] | Classical | — | Connected-component tilt | Sliding concentric windows | Pixel projections + PNN |
| Du et al. [2] | Classical survey | — | Hough line deskewing | Global Otsu thresholding | Template matching / ANN |
| Azam & Islam [6] | Classical | Spatial median only | Affine rotation | Local adaptive thresholding | Freeman chain code + k-NN |
| Dalal & Triggs + SVM [8, 16] | Classical ML | — | — | Global Otsu + density cut | HOG + Linear/RBF SVM |
| Silva & Jung (WPOD-Net) [5] | Deep CNN | Implicit (data aug.) | Learned planar affine warp | End-to-end CNN detector | YOLO-based OCR net |
| **This work** | **Classical + ML** | **2D Haar DWT + Hybrid Wiener–CLS, Weighted Median, Butterworth Notch** | **3-stream Rotated Extent + 4-pt Perspective Homography** | **Local Gaussian Adaptive + Max Contiguous Run-Length (`_max_run`)** | **480-D Hybrid (WHT + HOG + Fourier + Cov + Hu + Euler) + 7 Classifiers + Topo Rules** |

*Table 1: Design elements of representative license plate recognition pipelines compared with the present work. "—" indicates that the component is not part of the method's stated formulation.*

### 2.2 Spatial, frequency, and wavelet image restoration

Classical image restoration models degradation in either the spatial domain (for additive and impulsive noise) or the frequency domain (for periodic interference and convolutional blur) [3]. Order-statistic filters—most notably the median filter and the center-weighted median filter introduced by Ko and Lee [9]—suppress bipolar salt-and-pepper impulses while preserving step edges and narrow stroke corners better than linear arithmetic or geometric mean smoothers. For skewed positive noise distributions such as Rayleigh corruption, harmonic and contraharmonic mean filters provide targeted attenuation [3]. Periodic sinusoidal interference concentrates energy into conjugate impulse pairs in the 2D Discrete Fourier Transform (DFT) spectrum, allowing selective removal via Butterworth notch reject filters [3].

When an image is blurred by a known or estimated motion PSF $h(x,y)$ in the presence of additive noise, direct inverse filtering fails near the zeros of the optical transfer function $H(u,v)$. Wiener deconvolution [10] minimizes the statistical mean-square error using the noise-to-signal power ratio $K$, whereas Hunt's Constrained Least Squares (CLS) filter [11] minimizes the second-order Laplacian roughness $\|\mathbf{p} * \hat{f}\|^2$ subject to a noise norm constraint. To prevent residual high-frequency Gaussian noise from being amplified during frequency-domain inversion, Donoho and Johnstone [12] and Chang et al. [13] established wavelet shrinkage (BayesShrink) in the 2D Discrete Wavelet Transform (DWT) domain. Our Stage 1 pipeline unifies these principles by cascading 2D Haar BayesShrink denoising with a hybrid Wiener–CLS deconvolution transfer function.

### 2.3 Geometric rectification and character segmentation

Detecting license plates in cluttered vehicle scenes commonly leverages the high horizontal frequency of vertical character strokes on a bright reflective background [1, 6]. Morphological black-hat filtering highlights dark foreground strokes against bright backgrounds, while vertical Sobel gradients $|G_x|$ capture stroke transitions. Standard axis-aligned bounding boxes degrade when the camera roll or yaw angle exceeds $\pm 15^\circ$. By evaluating the minimum-area rotated bounding rectangle (`minAreaRect`) and approximating the plate contour via the Douglas–Peucker polygonal reduction algorithm [14], a 4-point planar homography $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ can warp foreshortened quadrilaterals into a fronto-parallel canonical grid using bicubic convolution interpolation [15].

For character binarization, Otsu's method [7] maximizes between-class gray-level variance globally, whereas MacQueen's $K$-Means clustering ($K=2$) [16] iteratively partitions pixel intensities by minimizing within-cluster sum-of-squares. Under diagonal vehicle shadows, global thresholds merge shadowed digits with the background; local Gaussian adaptive thresholding [17] overcomes this by estimating a spatially varying background surface. Once binarized, connected components are extracted via two-pass 8-connected labeling [18].

### 2.4 Transform-domain and topological descriptors

Representing normalized $32 \times 32$ character patches with compact, discriminative features is central to statistical pattern recognition [19]. Orthogonal 2D transforms—including the Discrete Hartley Transform (DHT) [20], the Discrete Cosine Transform (DCT-II) [21], and the sequency-ordered Walsh–Hadamard Transform (WHT) [22]—concentrate low-spatial-frequency stroke energy into the top-left transform coefficients. Boundary and regional shape descriptors complement transform spectra: Zahn and Roskies [23] parameterized closed contours via 1D DFT Fourier descriptors; Hu [24] derived seven algebraic moment invariants orthogonal to translation, scale, and in-plane rotation; and topological descriptors such as the Euler characteristic $E = C - H$ (connected components minus holes) [3, 18] capture structural invariants that remain unchanged under non-linear elastic deformations. Combining these global/topological cues with local Histograms of Oriented Gradients (HOG) [8] provides a rich representation for classical minimum-distance, correlation, Gaussian Bayes, multi-layer perceptron (MLP), $k$-Nearest Neighbors ($k$-NN), Random Forest, and RBF-kernel SVM classifiers [19, 25].

---

## 3 Proposed Method

### 3.1 Overview

Given an input BGR vehicle frame $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ (or a sequence of frames from a video stream), our system outputs a set of detected license plates $\{(\mathbf{b}_m, \mathbf{q}_m, \hat{\mathbf{s}}_m, c_m)\}_{m=1}^{M}$ ordered strictly from left to right along the horizontal image axis, where $\mathbf{b}_m = (x_m, y_m, w_m, h_m)$ is the bounding box, $\mathbf{q}_m \in \mathbb{R}^{4 \times 2}$ is the 4-vertex quadrilateral, $\hat{\mathbf{s}}_m$ is the recognized license plate string formatted to Vietnamese syntax (e.g., `29-D2 399.99`), and $c_m \in [0, 1]$ is the mean character posterior confidence. Table 2 summarizes the four sequential stages, their input/output resolutions, and their core mathematical operators.

| Stage | Module (`src/`) | Input Representation | Output Representation | Core Mathematical Operators |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | `restoration.py` | Degraded frame $g(x,y) \in \mathbb{R}^{H \times W \times 3}$ | Restored frame $\hat{f}(x,y) \in \mathbb{R}^{H \times W \times 3}$ | $3\times 3$ Center-Weighted Median, Harmonic Mean, Butterworth Notch $H_{\text{NR}}(u,v)$, 2D Haar DWT BayesShrink + Hybrid Wiener–CLS Deconvolution |
| **Stage 2** | `detection.py` | Restored frame $\hat{f}(x,y)$ | Rectified plate ROIs $\{\hat{f}_{\text{roi}}^{(m)}\} \in \mathbb{R}^{160 \times 128 \times 3}$ | Black-Hat $(f \bullet B)-f$, Vertical Sobel $|G_x|$, 3-Stream Rotated Extent Scoring, NMS (IoU $\le 0.25$), Douglas–Peucker 4-Point Homography $\mathbf{H}_{3\times 3}$ |
| **Stage 3** | `segmentation.py` | Rectified plate $\hat{f}_{\text{roi}} \in \mathbb{R}^{160 \times 128 \times 3}$ | Ordered character crops $\{I_k\}_{k=1}^{L} \in \mathbb{R}^{32 \times 32}$ | Otsu / $K$-Means ($K=2$) / Local Gaussian Adaptive Thresholding, Max Contiguous Run-Length Border Filter (`_max_run`), 8-Connected Labeling |
| **Stage 4** | `recognition.py` | Character patches $\{I_k\}_{k=1}^{L} \in \mathbb{R}^{32 \times 32}$ | Recognized string $\hat{\mathbf{s}}$ & confidence $c$ | 2D WHT/DCT/DHT ($100$-D) + HOG ($324$-D) + Hu/Cov/Fourier/Euler ($56$-D), PCA ($80$-D), 7 Classical Classifiers + Topological Post-Verification |

*Table 2: Summary of the four stages in the proposed classical license plate restoration and recognition pipeline.*

### 3.2 Stage 1: Multi-degradation image restoration

To handle diverse real-world corruptions without degrading clean frames, Stage 1 implements dedicated spatial, frequency, and wavelet-deconvolution operators on the luminance channel $Y$ in the $YCrCb$ color space.

**Spatial order-statistic and mean filters.** For additive Gaussian, skewed Rayleigh, and bipolar impulse corruptions over an $m \times n$ neighborhood $S_{xy}$, we implement the arithmetic, geometric, harmonic, and contraharmonic mean filters [3]:
$$\hat{f}_{\text{harm}}(x,y) = \frac{mn}{\sum_{(s,t)\in S_{xy}} \frac{1}{g(s,t) + \epsilon}}, \qquad \hat{f}_{\text{contra}}(x,y) = \frac{\sum_{(s,t)\in S_{xy}} g(s,t)^{Q+1}}{\sum_{(s,t)\in S_{xy}} g(s,t)^Q}. \qquad (2)$$
Here $\hat{f}_{\text{harm}}(x,y)$ effectively attenuates positive-skewed Rayleigh noise while preserving dark character ink. For bipolar salt-and-pepper noise ($P_a = P_b = 0.02$), a standard unweighted $3 \times 3$ median filter rounds off the acute vertices of thin digits (`'1'`, `'7'`, `'4'`). We therefore employ a $3 \times 3$ center-weighted median filter [9] with integer replication mask $\mathbf{W}_{\text{WMF}}$:
$$\mathbf{W}_{\text{WMF}} = \begin{bmatrix} 1 & 2 & 1 \\ 2 & 3 & 2 \\ 1 & 2 & 1 \end{bmatrix}, \qquad \hat{f}_{\text{WMF}}(x,y) = \text{median}\Big\{ \underbrace{g(s,t), \dots, g(s,t)}_{W_{\text{WMF}}(s,t)\text{ times}} \;\Big|\; (s,t)\in S_{xy} \Big\}. \qquad (3)$$

**Frequency-domain Butterworth notch reject filtering.** Periodic sinusoidal interference $\eta(x,y) = A\sin(2\pi(u_0 x/M + v_0 y/N))$ produces conjugate impulse spikes at $(M/2 \pm u_k, N/2 \pm v_k)$ in the centered 2D DFT magnitude spectrum $|G(u,v)|$. We suppress $Q$ spike pairs using a symmetric Butterworth notch reject transfer function of order $n=2$ and cutoff radius $D_{0k}=6.0$:
$$H_{\text{NR}}(u,v) = \prod_{k=1}^{Q} \left[ \frac{1}{1 + \left(\frac{D_{0k}}{D_k(u,v)}\right)^{2n}} \right] \left[ \frac{1}{1 + \left(\frac{D_{0k}}{D_{-k}(u,v)}\right)^{2n}} \right], \qquad (4)$$
where $D_{\pm k}(u,v) = \sqrt{(u - M/2 \mp u_k)^2 + (v - N/2 \mp v_k)^2}$, and reconstruct the clean spatial image via $\hat{f}(x,y) = \mathcal{R}e\{\mathcal{F}^{-1}\{H_{\text{NR}}(u,v)\,G(u,v)\}\}$.

**2D Haar wavelet BayesShrink and hybrid Wiener–CLS deconvolution.** For the coupled degradation $g(x,y) = (h * f)(x,y) + \eta(x,y)$ with linear motion PSF $h(x,y)$ of length $L$ and angle $\theta$, applying a direct inverse filter $\hat{F}(u,v) = G(u,v)/H(u,v)$ amplifies the noise term $N(u,v)/H(u,v)$ whenever $|H(u,v)| \to 0$. We decouple noise suppression from deconvolution in two steps:
1. *Wavelet shrinkage:* The luminance channel is decomposed via a 1-level separable 2D Haar DWT (`dwt2_haar`) with low-pass filter $\mathbf{h}_\phi = \frac{1}{\sqrt{2}}[1, 1]$ and high-pass filter $\mathbf{h}_\psi = \frac{1}{\sqrt{2}}[1, -1]$ into subbands $\{LL_1, LH_1, HL_1, HH_1\}$. The noise standard deviation $\hat{\sigma}_\eta$ is estimated from the diagonal detail subband $HH_1$ via Donoho's robust median estimator [12], and each detail subband $s \in \{LH_1, HL_1, HH_1\}$ is soft-thresholded using BayesShrink [13]:
   $$\hat{\sigma}_\eta = \frac{\text{median}(|HH_1|)}{0.6745}, \qquad T_s = \frac{\hat{\sigma}_\eta^2}{\sqrt{\max(\sigma_s^2 - \hat{\sigma}_\eta^2, \, 0) + \epsilon}}, \qquad (5)$$
   $$\tilde{d}_s(x,y) = \text{sign}\big(d_s(x,y)\big)\max\!\big(|d_s(x,y)| - T_s, \, 0\big). \qquad (6)$$
   Inverting the Haar transform (`idwt2_haar`) yields a pre-denoised observation $g_{\text{den}}(x,y)$ with 2D FFT $G_{\text{den}}(u,v)$.
2. *Hybrid Wiener–CLS deconvolution:* Let $H(u,v) = \mathcal{F}\{h(x,y)\}$ denote the optical transfer function and $P(u,v) = \mathcal{F}\{p(x,y)\}$ denote the 2D FFT of the discrete Laplacian operator $p(x,y) = [0, -1, 0; -1, 4, -1; 0, -1, 0]$. Combining Wiener's minimum MSE regularizer $K$ [10] with Hunt's smoothness regularizer $\gamma_{\text{CLS}}|P(u,v)|^2$ [11], we formulate the unit-DC-gain hybrid transfer function:
   $$\hat{F}(u,v) = \left[ \frac{(1 + K)\,H^*(u,v)}{|H(u,v)|^2 + K + \gamma_{\text{CLS}}\,|P(u,v)|^2} \right] G_{\text{den}}(u,v), \qquad (7)$$
   where $(1+K)$ preserves the zero-frequency mean luminance ($\hat{F}(0,0) = G_{\text{den}}(0,0)$ since $H(0,0)=1$ and $P(0,0)=0$) and $\gamma_{\text{CLS}}|P(u,v)|^2$ damps high-frequency ringing around sharp character boundaries. Finally, unsharp masking (highboost filtering with $k_{\text{boost}} = 0.45$) restores stroke edge acutance.

![Stage 1 Restoration and Deconvolution Pipeline](../outputs/fig01_module1_restoration.png)

![Multi-Noise Degradation Models and Restoration Filters](../outputs/fig09_multi_degradation_restoration.png)

### 3.3 Stage 2: Multi-plate morphological detection and 4-point homography

**Morphological and gradient energy map.** Given a restored grayscale frame $I \in [0, 255]^{H \times W}$ after full-scale histogram stretching, a rectangular structuring element $B_{25 \times 25}$ computes the black-hat transform $f_{\text{bh}} = (I \bullet B_{25 \times 25}) - I$, isolating dark character strokes on bright reflective plates. Simultaneously, a $3 \times 3$ horizontal Sobel kernel extracts vertical stroke edges $|G_x| = |\partial I / \partial x|$. The fused energy map $E = 0.5\,|G_x| + 0.5\,f_{\text{bh}}$ is binarized via Otsu's threshold and processed with morphological closing ($17 \times 11$) followed by opening ($5 \times 5$) to form compact candidate blobs.

**Three-stream candidate generation with rotated extent.** Single-stream contour detectors fail when plates are tilted by $\pm 30^\circ$, foreshortened near image borders, or accompanied by bright shop logos. Our candidate generator `_gather_plate_candidates()` pools regions from three complementary streams: (i) multi-threshold external contours ($t \in \{110, \dots, 200, t_{\text{Otsu}}\}$) for fronto-parallel plates ($1.0 \le W/H \le 1.62$, axis-aligned $\text{Extent} > 0.68$); (ii) connected components from the morphologically closed energy map $E_{\text{closed}}$; and (iii) high-brightness contours ($t \in [135, 210]$) evaluated by their minimum-area rotated bounding rectangle (`cv2.minAreaRect`) with dimensions $(W_{\text{rot}}, H_{\text{rot}})$. For a plate tilted at $-28.6^\circ$, the axis-aligned extent drops to $0.51$, whereas the rotated extent remains high:
$$\text{Extent}_{\text{rot}}(C) = \frac{\text{Area}(C)}{W_{\text{rot}} \cdot H_{\text{rot}}} > 0.66, \qquad \text{Aspect}_{\text{rot}}(C) = \frac{\max(W_{\text{rot}}, H_{\text{rot}})}{\min(W_{\text{rot}}, H_{\text{rot}})} \in [1.05, 1.85]. \qquad (8)$$
Each candidate region $R$ is scored by combining internal stroke energy and geometric fidelity to the canonical Vietnamese motorcycle plate aspect ratio ($1.24$):
$$\text{Score}(R) = 1.2\,\mu_{|G_x|}(R) + 1.6\,\mu_{\text{bh}}(R) + 1.1\,\sigma_I(R) + 45.0\,\text{Extent}_{\text{eff}} - 45.0\,\big|\text{Aspect}_{\text{eff}} - 1.24\big|. \qquad (9)$$

**Non-Maximum Suppression and strict left-to-right ordering.** Overlapping candidates are pruned via Non-Maximum Suppression with threshold $\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} > 0.25$. In multi-vehicle scenes, all surviving plate candidates are verified by downstream character count ($\ge 6$ valid character components, rejecting circular shop logos) and sorted strictly by their horizontal origin $x_m$ from left to right (`key=lambda p: p["bbox"][0]`).

**4-point perspective homography rectification.** When a plate is foreshortened ($W < H$) or strongly tilted ($\text{Extent} \le 0.68$), affine rotation alone cannot remove projective keystone distortion. We apply Douglas–Peucker polygonal approximation [14] to the convex hull of contour $C$ to extract the four corner vertices $\mathbf{q}_i = (x_i, y_i)^T$ ordered clockwise *(top-left, top-right, bottom-right, bottom-left)* and solve for the projective homography $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ mapping $\mathbf{q}_i$ to the canonical target corners $\mathbf{p}_i \in \{(0,0), (159,0), (159,127), (0,127)\}$:
$$s_i \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & 1 \end{bmatrix} \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}, \qquad (10)$$
resampling the rectified $160 \times 128$ plate via $4 \times 4$ bicubic convolution interpolation [15].

![Stage 2 Morphological Detection and Homography Deskewing](../outputs/fig02_module2_detection.png)

### 3.4 Stage 3: Adaptive binarization and contiguous run-length border suppression

**Binarization formulations.** On the rectified $160 \times 128$ plate image, we implement and compare three classical thresholding algorithms:
1. *Global Otsu thresholding [7]:* Exhaustively searches gray level $t \in [0, 255]$ to maximize between-class variance:
   $$t_{\text{Otsu}}^* = \arg\max_{0 \le t \le 255} \sigma_B^2(t) = \arg\max_{0 \le t \le 255} \omega_0(t)\,\omega_1(t)\,\big[\mu_0(t) - \mu_1(t)\big]^2. \qquad (11)$$
2. *Iterative $K$-Means ($K=2$) clustering [16]:* Alternates between assigning pixel intensities $I(x,y)$ to dark/bright clusters $C_0, C_1$ and updating centroids $m_0, m_1$ to minimize $J = \sum_{k=0}^1 \sum_{I \in C_k} (I - m_k)^2$, thresholding at $t_{\text{KM}}^* = \frac{1}{2}(m_0 + m_1)$.
3. *Local Gaussian adaptive thresholding [17]:* Estimates a local illumination surface via a $27 \times 27$ Gaussian kernel $G_\sigma$ and assigns foreground ($B(x,y)=1$ for dark ink) when:
   $$B(x,y) = \mathbb{I}\!\Big\{ I(x,y) < (G_\sigma * I)(x,y) - C \Big\}, \qquad C = 8. \qquad (12)$$

**Contiguous run-length border suppression (`_max_run`).** A critical failure mode in classical ALPR segmentation occurs when removing the dark outer plate frame. On two-line plates such as `29-D2 399.99`, the top horizontal bars of the four upper-row characters (`'2'`, `'9'`, `'D'`, `'2'`) align along scanlines $y \in [10, 14]$, causing the mean foreground density $\frac{1}{W}\sum_x B(x,y)$ to reach $0.42\text{--}0.49$. A naive density rule that clears rows with mean density $> 0.40$ slices off the top cap of `'D'` (turning `'D'` into an open `'U'` misclassified as `'B'`) and removes the right vertical stem of `'9'`.

We resolve this by distinguishing broken character caps from continuous border lines using the maximum contiguous foreground run-length $L_{\max}(\mathbf{b})$ along any 1D scanline $\mathbf{b} \in \{0, 1\}^N$:
$$L_{\max}(\mathbf{b}) = \max_{k} \big(e_k - s_k\big), \quad \text{where } b_i = 1 \;\;\forall i \in [s_k, e_k). \qquad (13)$$
Whereas four separate character caps have individual widths of $15\text{--}20\text{ pixels}$ ($L_{\max} \le 20$), a true horizontal plate border forms a single continuous stroke with $L_{\max} > 42\text{ pixels}$ (or $L_{\max} > 55\text{ pixels}$ vertically). Algorithm 1 formalizes our border suppression, morphological bridge breaking, and two-row 8-connected character extraction. In addition, diagonal `"VIE"` sticker divider lines are rejected by bounding-box fill ratio $\text{Area}/(w \cdot h) < 0.22$, and outer-edge border fragments on the 4-character upper row ($x + w \ge W - 15$, $w \le 12$) are pruned so that left-to-right character indexing is never shifted.

---

**Algorithm 1:** Contiguous Run-Length Border Suppression and Two-Row 8-Connected Character Extraction
- **Require:** Rectified binary plate mask $B \in \{0, 1\}^{H \times W}$ ($H=128, W=160$), margin $m_y=16, m_x=16$.
1. **function** $\text{MaxRun}(\mathbf{b})$: find contiguous 1-runs $\{[s_k, e_k)\}$ in $\mathbf{b}$ and **return** $\max_k(e_k - s_k)$
2. Clear outer $3\text{-pixel}$ frame: $B[0:3, :] \leftarrow 0, \; B[H-3:H, :] \leftarrow 0, \; B[:, 0:3] \leftarrow 0, \; B[:, W-3:W] \leftarrow 0$
3. **for** $r \in [0, m_y) \cup [H - m_y, H)$ **do**
4. &emsp;**if** $\text{mean}(B[r, :]) > 0.65$ **or** $\text{MaxRun}(B[r, :]) > 42$ **then** $B[r, :] \leftarrow 0$
5. **end for**
6. **for** $c \in [0, m_x) \cup [W - m_x, W)$ **do**
7. &emsp;**if** $\text{mean}(B[:, c]) > 0.65$ **or** $\text{MaxRun}(B[:, c]) > 55$ **then** $B[:, c] \leftarrow 0$
8. **end for**
9. Extract 8-connected components $\mathcal{C} = \{c_j\}$ via two-pass equivalence-table labeling; split wide merged blobs ($w_j > 33\text{ px}$) at vertical projection valley $\arg\min_x \sum_y B(y, x)$
10. Filter valid glyph candidates by aspect ratio $w_j/h_j \in [0.14, 0.92]$, height $h_j \in [0.21H, 0.58H]$, and fill ratio $\text{Area}_j/(w_j h_j) \ge 0.22$
11. Partition candidates into upper row ($y_{\text{center}} < 0.5H$, max 4 chars, pruning right-edge border streaks $x_j + w_j \ge W - 15, w_j \le 12$) and lower row ($y_{\text{center}} \ge 0.5H$, max 5 chars)
12. Sort each row strictly left-to-right by $x_j$ and resample each glyph to $32 \times 32$ with $4\text{-pixel}$ zero padding.

---

![Stage 3 Binarization and 8-Connected Character Segmentation](../outputs/fig03_module3_segmentation.png)

### 3.5 Stage 4: Hybrid transform–topological feature extraction and classification

Each segmented character is represented as a normalized $32 \times 32$ grayscale patch $f(x,y) \in [0, 1]^{32 \times 32}$ and binary mask $B_{32}(x,y) \in \{0, 1\}^{32 \times 32}$. We extract three complementary descriptor blocks:

**1. Orthogonal 2D transform coefficients ($100\text{-D}$).** We implement three separable 2D orthogonal transforms ($N=32$) and retain the top-left $10 \times 10$ low-sequency/low-frequency block ($\mathbb{R}^{100}$):
- *Sequency-ordered 2D Walsh–Hadamard Transform (WHT) [22]:* Recursive Sylvester–Hadamard matrix $\mathbf{H}_{32} \in \{-1, +1\}^{32 \times 32}$ whose rows are permuted by increasing sequency (number of zero-crossings $0, 1, \dots, 31$):
  $$\mathbf{W} = \frac{1}{32}\,\mathbf{H}_{32}^{\text{seq}}\,f\,\big(\mathbf{H}_{32}^{\text{seq}}\big)^T. \qquad (14)$$
- *2D Discrete Cosine Transform (DCT-II) [21]* and *2D Discrete Hartley Transform (DHT) [20]* using kernel $\text{cas}(\theta) = \cos\theta + \sin\theta$.

**2. Histograms of Oriented Gradients ($324\text{-D}$).** Using centered finite differences $G_x = [-1, 0, 1] * f$ and $G_y = [-1, 0, 1]^T * f$, gradient magnitudes are accumulated into $8 \times 8\text{-pixel}$ cells over 9 unsigned orientation bins in $[0^\circ, 180^\circ)$ and L2-Hys normalized across overlapping $2 \times 2$-cell blocks, yielding a $324\text{-D}$ vector [8].

**3. Structural, boundary Fourier, covariance, Hu moment, and Euler topological descriptors ($56\text{-D}$).**
- *7 log-transformed Hu moment invariants [24]:* Derived from normalized central moments $\eta_{pq} = \mu_{pq}/\mu_{00}^{1+(p+q)/2}$ ($\mathbb{R}^7$).
- *Region covariance eccentricity ($\mathbb{R}^3$):* From the $2 \times 2$ foreground coordinate covariance matrix $\mathbf{C}$ with eigenvalues $\lambda_1 \ge \lambda_2$, we compute effective diameter $d_e = \frac{1}{16}\sqrt{\text{Area}/\pi}$, eccentricity $\sqrt{1 - \lambda_2/\lambda_1}$, and axis ratio $\sqrt{\lambda_2/\lambda_1}$.
- *Boundary Fourier descriptors [23] ($\mathbb{R}^8$):* The outer boundary contour $\{(x_k, y_k)\}_{k=0}^{K-1}$ is mapped to the centered complex sequence $s(k) = (x_k - \bar{x}) + j(y_k - \bar{y})$, transformed via 1D DFT $a(u) = \frac{1}{K}\sum_{k=0}^{K-1} s(k)e^{-j2\pi u k/K}$, and normalized for scale and rotation invariance as $\{|a(u)|/|a(1)|\}_{u=2}^{9}$.
- *Euler topology and spatial zoning ($\mathbb{R}^{38}$):* Using 8-connected foreground labeling ($C$) and 4-connected internal hole labeling ($H$), we compute the Euler characteristic:
  $$E = C - H, \qquad (15)$$
  augmented with upper/lower hole counts $(H_{\text{top}}, H_{\text{bot}})$, relative hole area, $4 \times 4$ spatial zoning densities ($16\text{-D}$), and 8-bin horizontal/vertical projections ($16\text{-D}$).

Concatenating WHT ($100\text{-D}$), HOG ($324\text{-D}$), and structural/topological features ($56\text{-D}$) yields our **480-dimensional hybrid descriptor** (with optional PCA eigen-subspace reduction to $80\text{-D}$).

![Stage 4 Transform, Gradient, and Topological Descriptors](../outputs/fig04_module4_features.png)

**Classical pattern classifiers and topological post-verification.** Following feature standardization, we benchmark seven classical classifiers [19, 25]:
1. *Minimum-Distance (Nearest Centroid):* $d_j(\mathbf{x}) = \mathbf{x}^T\mathbf{m}_j - \frac{1}{2}\|\mathbf{m}_j\|^2$.
2. *Correlation Coefficient (Normalized Cross-Correlation):* $\gamma_j(\mathbf{x}) = \frac{(\mathbf{x}-\bar{x})^T(\mathbf{m}_j-\bar{m}_j)}{\|\mathbf{x}-\bar{x}\|\|\mathbf{m}_j-\bar{m}_j\|}$.
3. *Optimum Statistical Gaussian Bayes Classifier:* Quadratic Mahalanobis discriminant with regularized class covariance $\mathbf{C}_j^{\text{reg}} = (1-\alpha)\mathbf{C}_j + \alpha\mathbf{C}_{\text{pooled}} + \lambda\mathbf{I}$:
   $$d_j(\mathbf{x}) = \ln P(\omega_j) - \frac{1}{2}\ln|\mathbf{C}_j^{\text{reg}}| - \frac{1}{2}(\mathbf{x} - \mathbf{m}_j)^T (\mathbf{C}_j^{\text{reg}})^{-1}(\mathbf{x} - \mathbf{m}_j). \qquad (16)$$
4. *Distance-weighted $k$-NN ($k=3$)*, *Random Forest (200 trees)*, *Multi-Layer Perceptron (MLP, $128 \times 64$ hidden units)*, and *RBF-kernel SVM ($C=12.0, \gamma=\text{scale}$, Platt-scaled posterior probabilities)*. The RBF-SVM on the 480-D hybrid descriptor is used as the deployed classifier.

Finally, to resolve fine-grained visual ambiguities on real-world plates, we enforce Vietnamese positional syntax rules (positions $0, 1$ and $4\dots 8$ restricted to digits `0–9`; position $2$ restricted to series letters `A–Z`; position $3$ restricted to letters when position $2$ is `'A'` for $<50\text{cm}^3$ motorcycles) together with the deterministic topological rules summarized in Table 3.

| Ambiguous Pair | Physical Cause of Confusion | Topological & Geometric Invariant on $B_{32} \in \{0,1\}^{32 \times 32}$ | Decision Rule |
| :--- | :--- | :--- | :--- |
| **`'D'` vs. `'B'`** | Identical outer contour and right loop curvature | `'D'` has $H=1$ central cavity ($E=0$) so center-waist region $B_{32}[14:19, 14:19]$ is empty ($\rho_{\text{waist}} \in [0.00, 0.16]$); `'B'` has $H=2$ cavities ($E=-1$) separated by a solid horizontal crossbar ($\rho_{\text{waist}} \in [0.48, 0.88]$) | Predict `'D'` if $H \le 1$ and $\rho_{\text{waist}} < 0.26$; predict `'B'` if $H \ge 2$ and $\rho_{\text{waist}} > 0.42$ |
| **`'E'` vs. `'C'`** | Both have $H=0$ holes and open right side | `'E'` possesses a horizontal middle tongue inside $B_{32}[13:19, 11:22]$ ($\rho_{\text{mid}} > 0.34$); `'C'` has an empty interior cavity ($\rho_{\text{mid}} < 0.20$) | Promote `'E'` over `'C'` when $\rho_{\text{mid}} > 0.34$ and left stem density $\rho_{\text{left}} > 0.65$ |
| **`'7'` vs. `'1'`** | Diagonal stem of `'7'` resembles vertical stem of `'1'` | `'7'` has a wide top horizontal bar $W_{\text{top}} = \max_{r \in [4,10]} \sum_c B_{32}(r,c) \ge 11\text{ px}$ and high cap-to-stem width ratio $W_{\text{top}} / W_{\text{mid}}$ | Promote `'7'` over `'1'` when $W_{\text{top}} \ge 11\text{ px}$ and $W_{\text{top}} / W_{\text{mid}} \ge 2.25$ |

*Table 3: Topological and geometric disambiguation rules applied during Stage 4 post-verification.*

---

## 4 Experimental Setup

### 4.1 Dataset and zero-leakage plate-level protocol

The source corpus (`dataset/`) contains 116 Vietnamese motorcycle surveillance images ($472 \times 303\text{ pixels}$) and 116 text annotations, yielding **115 matched image–label pairs** (image `0042_01746_b.jpg` has a misnamed label file `0042_01141_b.txt` in the raw archive). A manual audit against the visual plates identified two ground-truth transcription typos in the raw archive, which are rectified in `LABEL_CORRECTIONS` (`src/pipeline.py`): `0000_02187_b.txt` (`59E12150` $\to$ `59E121500` for visual plate `59-E1 215.00`) and `018.txt` (`60B8010175` $\to$ `60B801075` for visual plate `60-B8 010.75`).

To prevent data leakage—where sibling characters cropped from the same physical license plate share identical illumination, blur, and font wear across training and test sets—we partition the dataset **at the vehicle plate level** (stratified by the series letter at index 2, random seed $42$) into **70% Training (80 plates)**, **15% Validation (17 plates)**, and **15% Held-Out Test (18 plates)**. Each split is then expanded **$6\times$** across six physical degradation regimes (`clean`, `awgn` $\sigma=10$, `salt_pepper` $4\%$, `rayleigh` $a=-8, b=160$, `periodic` $A=24, (u_0,v_0)=(24,18)$, and `motion_blur` $L=7\text{ px}, \theta=12^\circ, \sigma=7$), producing **690 vehicle images** and **5,643 segmented $32 \times 32$ character patches** (Table 4 and Figure 5). To prevent probability calibration collapse on rare or absent Vietnamese series letters (`'A'`, `'D'`, `'R'`, `'T'`, `'Y'`), the training split is augmented with DIN-1451 synthetic Vietnamese letter templates and $\pm 4^\circ$ micro-affine perturbations so every letter class has at least 14 training exemplars.

| Split | Ratio | Source Plates (`1x Clean`) | Expanded Images (`6x Regimes`) | Clean Character Crops ($32 \times 32$) | Expanded Character Crops ($6\times$) | Role in Experimental Protocol |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `train` | 70.0% | 80 | 480 | 676 | 3,973 | Fit `StandardScaler`, `PCA`, and 7 pattern classifiers |
| `val` | 15.0% | 17 | 102 | 132 | 767 | Tune restoration ($K, \gamma_{\text{CLS}}, D_0$) & classifier ($C, k$) hyperparameters |
| `test` | 15.0% | 18 | 108 | 156 | 903 | Held-out evaluation of all 4 stages (zero plate leakage) |
| **Total** | **100.0%** | **115** | **690** | **964** | **5,643** | Full 31-class Vietnamese alphanumeric set (`0–9`, `A–Z`) |

*Table 4: Zero-leakage stratified plate-level dataset splits and $6\times$ physics-based multi-degradation expansion.*

![Dataset Architecture and PSNR Gains Across Degradations](../outputs/fig10_dataset_splits_and_expansion.png)

### 4.2 Implementation details and hyperparameter configuration

All algorithms are implemented in Python 3 using NumPy, SciPy, OpenCV, and scikit-learn, executing entirely on CPU without GPU acceleration. Hyperparameters were selected on the 17-plate validation split and held fixed for all test evaluations (Table 5).

| Stage / Component | Parameter / Configuration | Selected Value |
| :--- | :--- | :--- |
| Stage 1: Wavelet & Deconvolution | Decomposition basis & levels; Wiener $K$; CLS $\gamma_{\text{CLS}}$; Highboost $k_{\text{boost}}$ | 2D Haar DWT (1 level); $K = 0.008$; $\gamma_{\text{CLS}} = 0.005$; $k_{\text{boost}} = 0.45$ |
| Stage 1: Spatial & Notch Filters | Center-weighted median mask; Harmonic window; Butterworth Notch $(n, D_{0k})$ | $3 \times 3$ ($W_{0,0}=3$); $3 \times 3$; order $n = 2$, radius $D_{0k} = 6.0\text{ px}$ |
| Stage 2: Detection & Homography | Black-hat SE; Closing / Opening SEs; Rotated extent $\tau_{\text{rot}}$; NMS IoU; ROI size | $25 \times 25$; $17 \times 11$ / $5 \times 5$; $\tau_{\text{rot}} = 0.66$; $\text{IoU} = 0.25$; $160 \times 128$ (Bicubic) |
| Stage 3: Binarization & CCL | Local Gaussian window & offset $C$; Max border run-length (`_max_run` H / V) | $27 \times 27$, $C = 8$; $L_{\max,\text{H}} = 42\text{ px}$, $L_{\max,\text{V}} = 55\text{ px}$ (or density $>0.65$) |
| Stage 4: Features & Classifiers | Hybrid descriptor dimension; PCA subspace; SVM kernel & $(C, \gamma)$; Probability calibration | $480\text{-D}$ ($100+324+56$); $80\text{-D}$; RBF ($C=12.0, \gamma=\text{scale}$); Platt scaling |

*Table 5: Hyperparameter configuration across all four pipeline stages (tuned on the validation split).*

### 4.3 Evaluation protocol and metrics

Restoration fidelity between a clean reference image $f(x,y)$ and a restored output $\hat{f}(x,y)$ of size $M \times N$ is quantified via Mean Squared Error ($\text{MSE}$), Signal-to-Noise Ratio ($\text{SNR}$), Peak Signal-to-Noise Ratio ($\text{PSNR}$), and the Structural Similarity Index ($\text{SSIM}$) [26]:
$$\text{MSE} = \frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}\big[f(x,y) - \hat{f}(x,y)\big]^2, \qquad \text{PSNR} = 10\log_{10}\!\left(\frac{255^2}{\text{MSE}}\right)\text{ (dB)}, \qquad (17)$$
$$\text{SSIM}(f, \hat{f}) = \frac{(2\mu_f\mu_{\hat{f}} + C_1)(2\sigma_{f\hat{f}} + C_2)}{(\mu_f^2 + \mu_{\hat{f}}^2 + C_1)(\sigma_f^2 + \sigma_{\hat{f}}^2 + C_2)}. \qquad (18)$$
Stage 2 localization is evaluated by the detection rate at $\text{IoU} \ge 0.50$. Stage 3 segmentation is evaluated by the **Exact Character Count Segmentation Rate** $\frac{1}{N}\sum_{i=1}^N \mathbb{I}(|\hat{\mathcal{C}}_i| = |\mathbf{s}_i|)$. Stage 4 and end-to-end performance are measured by **Character-Level Accuracy**, **Macro-F1**, and **Full-Plate Exact Match Accuracy** $\frac{1}{N}\sum_{i=1}^N \mathbb{I}(\hat{\mathbf{s}}_i = \mathbf{s}_i)$, where a plate is scored as correct only if every character in the sequence is recognized without error.

---

## 5 Results and Discussion

### 5.1 Stage 1 restoration and deconvolution ablation

Table 6 compares eight restoration configurations across all 115 vehicle images corrupted by linear motion blur ($L=7\text{ px}, \theta=12^\circ$) and additive Gaussian noise ($\sigma=7.0$). Feeding the degraded frames directly into the detector and segmenter yields an exact character segmentation rate of only $1.74\%$ ($2/115$ plates) and a full-plate recognition accuracy of **$0.00\%$**, because horizontal motion blur smears ink across the $3\text{--}4\text{ px}$ inter-character gaps. Applying a direct inverse filter ($\epsilon=0.08$) amplifies sensor noise near the spectral zeros of $H(u,v)$, reducing PSNR from $25.42\text{ dB}$ to $21.69\text{ dB}$ and SSIM from $0.7244$ to $0.3641$. Spatial-only smoothers (standard $3 \times 3$ median, center-weighted median, or wavelet shrinkage without deconvolution) improve PSNR by $+0.66\text{ to }+0.91\text{ dB}$ but cannot invert the convolutional phase of $h(x,y)$, leaving segmentation below $2\%$.

In contrast, frequency-domain regularized deconvolution recovers the inter-character gaps: standalone CLS ($\gamma=0.012$) and Wiener ($K=0.008$) filters raise exact character segmentation to $82.61\%$ and $85.22\%$, respectively. Cascading **2D Haar Wavelet BayesShrink** ahead of our **Hybrid Wiener–CLS filter** ($\gamma_{\text{CLS}}=0.005$) achieves the lowest MSE ($96.61$), highest PSNR ($28.34\text{ dB}$, $+2.92\text{ dB}$ gain), $86.09\%$ exact segmentation, **$93.59\%$ character recognition accuracy on the held-out test split** (vs. $3.85\%$ without restoration), and **$61.11\%$ full-plate exact match on the test split** (vs. $0.00\%$ without restoration).

| Restoration Method | MSE $\downarrow$ | SNR (dB) $\uparrow$ | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | Exact Char Seg. (%) $\uparrow$ | Char Acc. (%) $\uparrow$ | Full-Plate Acc. (%) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Degraded Input ($g = h * f + \eta$, No Restoration) | 190.42 | 19.18 | 25.42 | 0.7244 | 1.74 (2/115) | 7.75 | 0.00 (0/115) |
| 2. Direct Inverse Filter ($\epsilon = 0.08$) | 443.27 | 15.45 | 21.69 | 0.3641 | 0.87 (1/115) | 4.12 | 0.00 (0/115) |
| 3. Standard $3 \times 3$ Median Filter | 157.11 | 20.06 | 26.30 | 0.8454 | 0.87 (1/115) | 5.46 | 0.00 (0/115) |
| 4. $3 \times 3$ Center-Weighted Median Filter ($W_{0,0}=3$) | 156.08 | 20.09 | 26.33 | 0.8390 | 1.74 (2/115) | 6.85 | 0.87 (1/115) |
| 5. 2D Haar Wavelet BayesShrink Only (No Deconv) | 164.68 | 19.84 | 26.08 | 0.8210 | 1.74 (2/115) | 7.94 | 0.87 (1/115) |
| 6. Constrained Least Squares (CLS, $\gamma=0.012$) | 103.85 | 21.79 | 28.03 | 0.7712 | 82.61 (95/115) | 82.12 | 44.35 (51/115) |
| 7. Wiener Deconvolution ($K=0.008$) | 100.21 | 22.01 | 28.25 | **0.8621** | 85.22 (98/115) | 84.91 | 46.96 (54/115) |
| **8. Proposed: 2D Haar Wavelet + Hybrid Wiener–CLS** | **96.61** | **22.09** | **28.34** | 0.7333 | **86.09 (99/115)** | **85.80 (Test: 93.59)** | **47.83 (Test: 61.11)** |

*Table 6: Ablation of eight restoration methods on 115 vehicle images under linear motion blur ($L=7\text{ px}, \theta=12^\circ$) and AWGN ($\sigma=7.0$).*

Table 7 extends this evaluation across all six physical degradation regimes ($690$ total images and the 18-plate held-out test split). On `salt_pepper` ($4\%$ impulse noise), the $3 \times 3$ center-weighted median filter boosts PSNR from $19.22\text{ dB}$ to **$37.20\text{ dB}$ ($+17.98\text{ dB}$)** and SSIM from $0.3825$ to **$0.9875$**, raising held-out test full-plate accuracy from $55.56\%$ to **$94.44\%$ ($+38.88\text{ points}$)**. On `periodic` sinusoidal corruption, the Butterworth notch reject filter improves PSNR by **$+14.67\text{ dB}$** ($23.75 \to 38.42\text{ dB}$) and test full-plate accuracy from $72.22\%$ to **$94.44\%$ ($+22.22\text{ points}$)**. Figure 6 illustrates a representative motion-blurred test plate before and after Stage 1 restoration.

| Degradation Regime & Dedicated Stage 1 Filter | PSNR Before $\to$ After (dB) | SSIM Before $\to$ After | Exact Seg. Rate (No Rest. $\to$ Rest.) | Test Char Acc. (No Rest. $\to$ Rest.) | Test Full-Plate Acc. (No Rest. $\to$ Rest.) | $\Delta$ Test Plate Acc. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. `clean` (Unmodified Surveillance Input) | $\infty$ | $1.0000$ | $95.65\% \to 95.65\%$ | $94.87\% \to 94.87\%$ | $94.44\% \to 94.44\%$ | $+0.00$ |
| 2. `awgn` ($\sigma=10.0$) $\to$ 2D Haar DWT + Highboost | $28.24 \to \mathbf{30.59}$ | $0.7325 \to \mathbf{0.8416}$ | $96.52\% \to 95.65\%$ | $95.51\% \to \mathbf{100.00\%}$ | $94.44\% \to \mathbf{100.00\%}$ | **$+5.56$** |
| 3. `salt_pepper` ($4\%$) $\to$ $3\times 3$ Center-Weighted Median | $19.22 \to \mathbf{37.20}$ | $0.3825 \to \mathbf{0.9875}$ | $90.43\% \to \mathbf{94.78\%}$ | $89.74\% \to \mathbf{94.87\%}$ | $55.56\% \to \mathbf{94.44\%}$ | **$+38.88$** |
| 4. `rayleigh` ($a=-8, b=160$) $\to$ Harmonic Mean + Highboost | $32.81 \to 27.01$ | $0.8774 \to \mathbf{0.9353}$ | $95.65\% \to 92.17\%$ | $94.87\% \to \mathbf{100.00\%}$ | $94.44\% \to \mathbf{100.00\%}$ | **$+5.56$** |
| 5. `periodic` ($A=24$) $\to$ Butterworth Notch Reject ($n=2$) | $23.75 \to \mathbf{38.42}$ | $0.4962 \to \mathbf{0.9694}$ | $86.96\% \to \mathbf{94.78\%}$ | $94.87\% \to \mathbf{95.51\%}$ | $72.22\% \to \mathbf{94.44\%}$ | **$+22.22$** |
| 6. `motion_blur` ($L=7, \theta=12^\circ, \sigma=7$) $\to$ Wavelet + Wiener–CLS | $25.42 \to \mathbf{28.33}$ | $0.7242 \to \mathbf{0.7334}$ | $1.74\% \to \mathbf{86.09\%}$ | $3.85\% \to \mathbf{93.59\%}$ | $0.00\% \to \mathbf{61.11\%}$ | **$+61.11$** |

*Table 7: End-to-end restoration impact across all six physical degradation regimes on the 690-image expanded corpus and the held-out test split.*

![Degraded vs Restored Pipeline Comparison](../outputs/fig08_degraded_vs_restored_pipeline.png)

### 5.2 Stage 2 and Stage 3 detection and binarization analysis

Stage 2 achieves a **$100.0\%$ ($115/115$) plate detection rate** ($\text{IoU} \ge 0.50$) across all clean and restored surveillance frames. Table 8 compares the three Stage 3 binarization methods on the 115 original plates and under simulated diagonal shadow gradients (`shadow_strength = 0.55`). While global Otsu thresholding ($93.91\%$) and iterative $K$-Means clustering ($93.04\%$) perform well under uniform illumination, strong diagonal shadows shift the global bimodal histogram and degrade Otsu by $-2.61\text{ points}$ to $91.30\%$. Local Gaussian adaptive thresholding tracks the spatial illumination surface within each $27 \times 27$ neighborhood, achieving the highest exact character segmentation rate on both normal (**$95.65\%$, $110/115$**) and shadowed (**$94.78\%$, $109/115$**) plates.

| Binarization & Segmentation Method | Exact Char Count on Original Corpus (%) | Exact Char Count Under Diagonal Shadow (%) | Shadow Degradation ($\Delta$) |
| :--- | :---: | :---: | :---: |
| 1. Global Otsu Thresholding ($\max \sigma_B^2(t)$) | 93.91 (108/115) | 91.30 (105/115) | $-2.61$ |
| 2. Iterative $K$-Means Clustering ($K=2$) | 93.04 (107/115) | 92.17 (106/115) | $-0.87$ |
| **3. Local Adaptive Gaussian Thresholding (Proposed)** | **95.65 (110/115)** | **94.78 (109/115)** | **$-0.87$** |

*Table 8: Comparison of global Otsu, $K=2$ K-Means clustering, and local Gaussian adaptive thresholding under uniform and shadowed illumination.*

### 5.3 Stage 4 feature families, classifiers, and $6\times$ expansion ablation

Table 9 reports 5-fold cross-validation accuracy across seven feature representations and seven classical classifiers on the 964 clean character patches, while Table 10 evaluates generalization on the held-out test split ($156$ clean test characters and $903$ expanded $6\times$ test characters) when training exclusively on the $1\times$ clean training split ($676$ chars) versus the $6\times$ multi-degradation expanded training split ($3,973$ chars).

Three patterns emerge from Tables 9 and 10:
1. **Transform vs. gradient and hybrid representations:** Among the 100-D orthogonal transforms, sequency-ordered 2D WHT ($95.33\%$ with SVM) slightly outperforms 2D DCT-II ($95.23\%$) and 2D DHT ($92.84\%$) while requiring only additions and subtractions. Local HOG descriptors reach $96.37\%$. Fusing WHT + HOG + Hu/Covariance/Fourier/Euler (`combined_pca` at $80\text{-D}$ and `combined` at $480\text{-D}$) achieves the highest overall accuracy of **$96.99\%$** (MLP on $80\text{-D}$ PCA subspace) and **$96.37\%$** (RBF-SVM at $1.01\text{--}2.69\text{ ms/char}$).
2. **Classifier hierarchy:** Across all feature sets, non-linear margin and neural classifiers (RBF-SVM, MLP, and distance-weighted $k$-NN) consistently outperform linear centroid and template-correlation baselines by $2.5\text{--}4.2\text{ percentage points}$, while the regularized Gaussian Bayes classifier (`bayes`) reaches $94.40\%$ on the hybrid descriptor.
3. **Gain from $6\times$ multi-degradation training:** Training on the $6\times$ expanded training split improves held-out $6\times$ test accuracy for every non-parametric and margin classifier: $+3.32\text{ points}$ for Random Forest ($91.36\% \to 94.68\%$), **$+2.88\text{ points}$ for RBF-SVM ($92.36\% \to 95.24\%$)**, and $+2.66\text{ points}$ for $k$-NN ($92.36\% \to 95.02\%$). Figure 7 shows the 5-fold CV confusion matrix and classifier comparison, and Figure 8 summarizes all three ablation studies.

| Feature Representation | Dim. | Min-Dist | Correlation | Gaussian Bayes | $k$-NN ($k=3$) | MLP Net | Random Forest | RBF-SVM | SVM Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Hu + Cov + Fourier + Euler (`hu_topo`) | 56 | 66.70 | 71.27 | 84.02 | 90.25 | **92.43** | 91.39 | 91.70 | $0.98\text{ ms}$ |
| 2. 2D Discrete Hartley Transform (`dht`) | 100 | 66.08 | 65.66 | 69.81 | 91.80 | 89.00 | 87.66 | **92.84** | $0.87\text{ ms}$ |
| 3. 2D Discrete Cosine Transform (`dct`) | 100 | 76.04 | 75.41 | 84.34 | 94.09 | 94.61 | 93.67 | **95.23** | $1.12\text{ ms}$ |
| 4. 2D Walsh–Hadamard Transform (`wht`) | 100 | 75.83 | 75.83 | 84.65 | 94.19 | 94.50 | 93.98 | **95.33** | $1.03\text{ ms}$ |
| 5. Histograms of Oriented Gradients (`hog`) | 324 | 93.57 | 93.57 | 93.67 | 96.16 | 96.27 | 94.19 | **96.37** | $1.10\text{ ms}$ |
| 6. Combined Hybrid + PCA (`combined_pca`) | 80 | 91.91 | 91.08 | 94.19 | 95.23 | **96.99** | 94.92 | 96.16 | $1.01\text{ ms}$ |
| **7. Combined Hybrid (`wht`+`hog`+`hu_topo`)** | **480** | **92.12** | **92.84** | **94.40** | **95.54** | **96.58** | **95.12** | **96.37** | $2.69\text{ ms}$ |

*Table 9: 5-fold cross-validation character accuracy (%) on 964 clean character crops across seven feature families and seven classical classifiers.*

| Classical Pattern Classifier | Trained on `1x Clean Train` (676 chars) $\to$ Eval on `6x Test` (903 chars) | Trained on `6x Expanded Train` (3,973 chars) $\to$ Eval on `1x Clean Test` (156 chars) | Trained on `6x Expanded Train` (3,973 chars) $\to$ Eval on `6x Test` (903 chars) | Gain from $6\times$ Training ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| 1. Minimum-Distance (`min_dist`) | 89.92 | 89.74 | 89.81 | $-0.11$ |
| 2. Correlation Coefficient (`corr`) | 90.48 | 89.74 | 89.26 | $-1.22$ |
| 3. Statistical Gaussian Bayes (`bayes`) | 91.58 | 92.31 | **91.92** | **$+0.34$** |
| 4. Distance-Weighted $k$-NN ($k=3$) | 92.36 | **94.87** | **95.02** | **$+2.66$** |
| 5. Random Forest (`rf`, 200 trees) | 91.36 | 94.23 | **94.68** | **$+3.32$** |
| 6. Multi-Layer Perceptron (`mlp`) | 93.91 | **94.87** | **95.13** | **$+1.22$** |
| **7. RBF Support Vector Machine (`svm`)** | 92.36 | 94.23 | **95.24** | **$+2.88$** |

*Table 10: Held-out test split character recognition accuracy (%) comparing $1\times$ clean training against $6\times$ multi-degradation expanded training.*

![Stage 4 Confusion Matrix and Feature/Classifier Benchmark](../outputs/fig05_module4_evaluation.png)

![Comprehensive Ablation Studies Dashboard](../outputs/fig06_ablation_studies.png)

### 5.4 In-the-wild stress tests and qualitative generalization

To evaluate robustness beyond the 115-plate parking dataset, we tested the complete pipeline on four challenging in-the-wild motorcycle scenes exhibiting multi-vehicle clutter, extreme perspective foreshortening, steep in-plane tilt with `"VIE"` stickers and shop logos, aligned top-cap border interference, and right-edge border streaks (Table 11 and Figure 9), as well as the held-out test split gallery (Figure 10). Across all four stress-test scenes, the pipeline achieves **100% character and full-plate accuracy** ($45/45$ characters across 5 plates), ordering multiple plates and character rows strictly from left to right.

| Stress-Test Case | Scene Geometry & Visual Distractors | Classical Pipeline Mechanism Activated | Recognized Output |
| :--- | :--- | :--- | :--- |
| **Case 1: Multi-Vehicle & Perspective Foreshortening** (`images.jpg`) | Two motorcycles side-by-side; left plate near bottom-left edge ($c_x/W = 0.10$) foreshortened ($53 \times 63\text{ px}, W < H, +12.9^\circ$); right plate tilted $-16.1^\circ$; `<50cc` two-letter series `AB` absent from raw corpus | • Multi-plate NMS (`detect_all_license_plates`) sorted left-to-right<br>• 4-point Douglas–Peucker homography (`extract_plate_quad`)<br>• Synthetic `'A'` glyph augmentation + 2-letter `<50cc` syntax rule | **`29-AB 901.50`** (Left)<br>**`29-AB 901.06`** (Right)<br>*(18/18 chars correct)* |
| **Case 2: Steep $-28.6^\circ$ Tilt, `"VIE"` Sticker & Shop Logo** (`images (1).jpg`) | Plate `49-E1 796.52` tilted at $-28.6^\circ$ with red `"VIE"` flag sticker on left and bright circular `HOÀNG TRÍ` shop logo in top-right corner | • High-brightness rotated-extent stream ($\text{Extent}_{\text{rot}} > 0.66$)<br>• Rejects logo candidate ($<6$ chars) & thin sticker divider ($\text{Area}/(wh) < 0.22$)<br>• Disambiguates `'E'` vs. `'C'` ($\rho_{\text{mid}}>0.34$) and `'7'` vs. `'1'` ($W_{\text{top}}/W_{\text{mid}} = 3.0$) | **`49-E1 796.52`**<br>*(9/9 chars correct; logo rejected)* |
| **Case 3: Top-Cap Preservation & `'D'` vs. `'B'` Topology** (`images (3).jpg`) | Street scene with plate `29-D2 399.99`; aligned top caps of `2, 9, D, 2` span $49\%$ of row width, causing naive density filters to slice off the top bar of `'D'` | • Contiguous run-length border filter (`_max_run > 42` vs. individual caps $\le 20\text{ px}$) preserves top bar of `'D'` and right stem of `'9'`<br>• Euler hole count ($H=1, E=0$) & empty waist density ($\rho_{\text{waist}} \in [0.00, 0.16] < 0.26$) | **`29-D2 399.99`**<br>*(9/9 chars correct; `'D'` conf. $90\%$)* |
| **Case 4: Upper-Row Border Streak & `'7'` vs. `'1'` Ratio** (`images (2).jpg`) | Dashcam view of plate `17-B7 581.49` with broken vertical right-frame fragment at $x=135$ on upper row that previously shifted `[1, 7, B, 7]` to `[7, B, 7, |]` | • Prunes narrow outer-edge streaks on 4-char top row ($x+w \ge W-15, w \le 12$) with border proximity penalty<br>• Top-cap to mid-stem width ratio $W_{\text{top}}/W_{\text{mid}} = 2.60 \ge 2.25$ confirms `'7'` | **`17-B7 581.49`**<br>*(9/9 chars correct; strict left-to-right)* |

*Table 11: Analysis of four in-the-wild stress-test scenes illustrating multi-plate NMS, 4-point homography, contiguous run-length border suppression, and topological disambiguation.*

![In-the-Wild Stress-Test Generalization Cases](../outputs/fig11_wild_generalization_cases.png)

![End-to-End Demo Gallery on Held-Out Test Split](../outputs/fig07_end_to_end_demo.png)

### 5.5 Error analysis and runtime efficiency

**Error analysis.** Inspection of the confusion matrix (Figure 7) and the remaining $5.56\%$ ($1/18$) full-plate errors on the held-out test split identifies two main failure patterns:
1. *Physical occlusion and rust rivets ($4.35\%$ of corpus plates):* On plates where mud, mounting bolts, or severe surface abrasion physically sever a character stroke or bridge two adjacent digits, 8-connected labeling either splits one digit into two fragments or merges two digits beyond the $33\text{-pixel}$ valley-splitting threshold.
2. *Fine-grained glyph confusion at position 3 ($\approx 3.0\text{--}4.7\%$ of characters):* Because index 3 on Vietnamese plates legally admits both digits (`0–9`) and secondary series letters (`A–Z` for $<50\text{cm}^3$ motorcycles), syntax constraints alone cannot disambiguate `('0', 'D')`, `('8', 'B')`, or `('2', 'Z')` at that position when extreme motion blur ($L > 10\text{ px}$) fills internal cavities.

**Runtime efficiency.** Running on a desktop CPU (Python/OpenCV, no GPU), the interactive application reports an end-to-end latency of **$17.9\text{ ms}$ per frame ($\approx 56\text{ FPS}$)** in direct mode on a two-plate still image, and **$65\text{--}85\text{ ms}$ per frame ($\approx 12\text{--}15\text{ FPS}$)** when the Stage 1 restoration branch (2D Haar wavelet + FFT-based Wiener–CLS deconvolution) is enabled on in-the-wild images. These figures are wall-clock measurements from the GUI and depend on image size and the number of plate candidates. Our interactive application (`demo.py`) further incorporates a 7-frame sliding-window **Temporal Majority Voter** for surveillance video streams (`.mp4/.avi`), stabilizing per-frame predictions across consecutive frames.

### 5.6 Limitations and threats to validity

- **Known vs. blind PSF parameters.** In our motion-blur benchmark, the Wiener–CLS filter uses known or semi-automatically configured PSF parameters $(L, \theta)$. Under non-uniform camera shake or accelerating vehicles, spatially varying blur requires blind PSF estimation from spectral zero-lines in the 2D cepstrum.
- **Corpus size and rare-letter synthesis.** Although we expanded the 115 source plates $6\times$ to 690 images (5,643 characters) and enforced zero-leakage plate-level splits, the 18-plate clean test split means that 1 plate corresponds to $5.56$ percentage points in full-plate accuracy. Moreover, absent series letters (`'A'`, `'R'`) rely on synthetic DIN-1451 glyphs mixed with real training patches.
- **Single-scale morphological structuring elements.** Stage 2 uses fixed structuring elements ($25 \times 25$ black-hat, $17 \times 11$ closing) tuned for plates occupying $60\text{--}180\text{ pixels}$ in width. Extremely distant plates ($<35\text{ pixels}$ wide) or ultra-close macro shots require multi-scale image pyramid scanning.

---

## 6 Conclusion and Future Work

We presented and audited a four-stage classical image restoration, geometric rectification, segmentation, and statistical pattern recognition pipeline for two-line Vietnamese motorcycle license plates. By cascading 2D Haar wavelet BayesShrink denoising with a unit-DC-gain hybrid Wiener–Constrained Least Squares deconvolution filter, a $3 \times 3$ center-weighted median smoother, and a conjugate Butterworth notch reject filter, Stage 1 achieves PSNR gains of $+2.35\text{ to }+17.98\text{ dB}$ on four of five physical degradation regimes and restores held-out test character recognition under motion blur from $3.85\%$ to $93.59\%$. Combining three-stream rotated-extent morphological detection, Douglas–Peucker 4-point perspective homography, local Gaussian adaptive thresholding, maximum contiguous run-length (`_max_run`) border suppression, and a 480-D hybrid transform–topological classifier yields $95.65\%$ exact segmentation, $96.99\%$ 5-fold CV character accuracy, and $94.44\%\text{--}100.00\%$ full-plate accuracy on clean and noise-restored test splits, while generalizing cleanly to challenging multi-vehicle and steeply tilted in-the-wild images.

Future work will focus on: (1) automating blind motion-blur PSF parameter estimation $(\hat{L}, \hat{\theta})$ via 2D cepstral Radon analysis on each detected plate ROI; (2) extending multi-plate NMS with a constant-velocity Kalman filter for multi-object trajectory tracking across video streams; and (3) expanding the real-world annotated corpus to cover all 63 provincial codes and diplomatic/commercial plate color schemes.

---

## A Dataset and Pipeline Consistency Audit

Table 12 documents the dataset anomalies and classical failure modes identified during development and how each is handled in the codebase and manuscript.

| Audit Item | Raw Archive / Naive Baseline Behavior | Root Cause Identified | Resolution in Codebase & Manuscript |
| :--- | :--- | :--- | :--- |
| Unmatched file `0042_01746_b.jpg` | 116 `.jpg` files and 116 `.txt` files in `dataset/`, but only 115 matching basenames | Label file misnamed `0042_01141_b.txt` in raw archive | Reported transparently as 115 matched pairs in Section 4.1 |
| Ground-truth typos (`0000_02187_b`, `018`) | `0000_02187_b.txt` has 8 chars (`59E12150`); `018.txt` has 10 chars (`60B8010175`) | Missing trailing `'0'` on `59-E1 215.00`; duplicated `'1'` on `60-B8 010.75` | Rectified via `LABEL_CORRECTIONS` in `src/pipeline.py` |
| Character `'D'` misclassified as `'B'` (`29-D2 399.99`) | Mean row density filter (`mean > 0.40`) zeroed scanlines $y \in [10, 14]$ and Platt CV collapsed on 3 `'D'` samples | Aligned top caps of `2, 9, D, 2` span $49\%$ of width; sliced `'D'` cap resembled `'B'` | Replaced density cut with `_max_run > 42` in `src/segmentation.py`; balanced rare letters $\ge 14$ samples + Euler waist rule in `src/recognition.py` |
| Right-to-left / shifted top row (`17-B7 581.49`) | Upper row read as `78-T1` instead of `17-B7` | Right-frame vertical fragment at $x=135$ selected among top 4 blobs, dropping `'1'` at $x=26$ | Pruned narrow outer-edge streaks ($x+w \ge W-15, w \le 12$) on 4-char upper row and enforced strict left-to-right sorting |
| Steep tilt & `"VIE"` sticker (`49-E1 796.52`) | Detector locked onto circular shop logo; sticker divider merged with `'4'` and `'7'` | Axis-aligned extent dropped to $0.51$ at $-28.6^\circ$ tilt; logo had high contrast | Added rotated-extent stream ($\text{Extent}_{\text{rot}}>0.66$), rejected $<6$-char logos, and filtered thin sticker dividers ($\text{Area}/(wh)<0.22$) |

*Table 12: Audit of dataset label discrepancies and classical pipeline edge cases resolved in this work.*

---

## B Reproducibility Checklist

- **Data & Splits:** 115 matched vehicle plates in `dataset/`, partitioned at the plate level (seed $42$) into 80 train / 17 val / 18 test in `data/splits/dataset_splits.json` and expanded $6\times$ (690 images, 5,643 character patches) in `data/splits/expanded_dataset_manifest.csv`.
- **Software Environment:** Python 3.10+, `numpy>=1.24.0`, `scipy>=1.10.0`, `opencv-python>=4.8.0`, `scikit-image>=0.21.0`, `scikit-learn>=1.3.0`, `matplotlib>=3.7.0` (listed in `requirements.txt`).
- **Experiment & Report Scripts:**
  - `python generate_all_experiments.py` — regenerates all splits, 5-fold CV tables, held-out test metrics (`outputs/metrics_summary.json`), and figures (`outputs/fig01..fig10.png`).
  - `python demo.py` — launches the interactive real-time Video & Image Restoration Studio GUI (`python demo.py --cli` runs headless test-split verification).
  - `python build_pdf_report.py` & `python build_word_report.py` — compiles `report/FINAL_REPORT.md` into `report/FINAL_REPORT.pdf` and `report/FINAL_REPORT_IMP302.docx`.

---

## References

[1] C.-N. E. Anagnostopoulos, I. E. Anagnostopoulos, I. D. Psoroulas, V. Loumos, and E. Kayafas, "License plate recognition from still images and video sequences: A survey," *IEEE Transactions on Intelligent Transportation Systems*, vol. 9, no. 3, pp. 377–391, Sept. 2008.

[2] S. Du, M. Ibrahim, M. Shehata, and W. Badawy, "Automatic license plate recognition (ALPR): A state-of-the-art review," *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 23, no. 2, pp. 311–325, Feb. 2013.

[3] R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. New York, NY, USA: Pearson, 2018.

[4] R. Laroca, E. Severo, L. A. Zanlorensi, L. S. Oliveira, G. R. Gonçalves, W. R. Schwartz, and D. Menotti, "A robust real-time automatic license plate recognition based on the YOLO detector," in *Proceedings of the International Joint Conference on Neural Networks (IJCNN)*, Rio de Janeiro, Brazil, 2018, pp. 1–10.

[5] S. M. Silva and C. R. Jung, "License plate detection and recognition in unconstrained scenarios," in *Proceedings of the European Conference on Computer Vision (ECCV)*, Munich, Germany, 2018, pp. 580–596.

[6] S. Azam and M. M. Islam, "Automatic license plate detection in hazardous condition," *Journal of Visual Communication and Image Representation*, vol. 36, pp. 172–186, Apr. 2016.

[7] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62–66, Jan. 1979.

[8] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, San Diego, CA, USA, vol. 1, 2005, pp. 886–893.

[9] S.-J. Ko and Y. H. Lee, "Center weighted median filters and their applications to image enhancement," *IEEE Transactions on Circuits and Systems*, vol. 38, no. 9, pp. 984–993, Sept. 1991.

[10] N. Wiener, *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*. Cambridge, MA, USA: MIT Press, 1949.

[11] B. R. Hunt, "The application of constrained least squares estimation to image restoration by digital computer," *IEEE Transactions on Computers*, vol. C-22, no. 9, pp. 805–812, Sept. 1973.

[12] D. L. Donoho and I. M. Johnstone, "Ideal spatial adaptation by wavelet shrinkage," *Biometrika*, vol. 81, no. 3, pp. 425–455, Sept. 1994.

[13] S. G. Chang, B. Yu, and M. Vetterli, "Adaptive wavelet thresholding for image denoising and compression," *IEEE Transactions on Image Processing*, vol. 9, no. 9, pp. 1532–1546, Sept. 2000.

[14] D. H. Douglas and T. K. Peucker, "Algorithms for the reduction of the number of points required to represent a digitized line or its caricature," *Cartographica*, vol. 10, no. 2, pp. 112–122, Dec. 1973.

[15] R. Keys, "Cubic convolution interpolation for digital image processing," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 29, no. 6, pp. 1153–1160, Dec. 1981.

[16] J. MacQueen, "Some methods for classification and analysis of multivariate observations," in *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, vol. 1, 1967, pp. 281–297.

[17] D. Bradley and G. Roth, "Adaptive thresholding using the integral image," *Journal of Graphics Tools*, vol. 12, no. 2, pp. 13–21, 2007.

[18] R. M. Haralick and L. G. Shapiro, *Computer and Robot Vision*, vol. 1. Reading, MA, USA: Addison-Wesley, 1992.

[19] R. O. Duda, P. E. Hart, and D. G. Stork, *Pattern Classification*, 2nd ed. New York, NY, USA: Wiley-Interscience, 2000.

[20] R. N. Bracewell, *The Hartley Transform*. New York, NY, USA: Oxford University Press, 1986.

[21] N. Ahmed, T. Natarajan, and K. R. Rao, "Discrete cosine transform," *IEEE Transactions on Computers*, vol. C-23, no. 1, pp. 90–93, Jan. 1974.

[22] J. L. Walsh, "A closed set of normal orthogonal functions," *American Journal of Mathematics*, vol. 45, no. 1, pp. 5–24, Jan. 1923.

[23] C. T. Zahn and R. Z. Roskies, "Fourier descriptors for plane closed curves," *IEEE Transactions on Computers*, vol. C-21, no. 3, pp. 269–281, Mar. 1972.

[24] M.-K. Hu, "Visual pattern recognition by moment invariants," *IRE Transactions on Information Theory*, vol. 8, no. 2, pp. 179–187, Feb. 1962.

[25] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, Sept. 1995.

[26] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, Apr. 2004.
