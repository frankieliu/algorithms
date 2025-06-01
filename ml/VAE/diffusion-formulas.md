To understand **Stable Diffusion** (a latent diffusion model for text-to-image generation), you’ll need to learn several key formulas across **diffusion models**, **VAEs**, and **transformers**. Below is a structured breakdown of the essential equations:

---

### **1. Diffusion Process (Forward & Reverse)**
Stable Diffusion operates in **latent space** (compressed via a VAE), but the core math relies on diffusion theory.

#### **(A) Forward Diffusion (Noising Process)**
Gradually adds Gaussian noise to an image over \( T \) steps:
\[
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})
\]
- \( x_t \): Noisy image at step \( t \).
- \( \beta_t \): Noise schedule (small values for early steps, larger later).
- **Closed-form for arbitrary \( t \)**:
  \[
  q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)\mathbf{I})
  \]
  where \( \alpha_t = 1-\beta_t \), \( \bar{\alpha}_t = \prod_{i=1}^t \alpha_i \).

#### **(B) Reverse Diffusion (Denoising Process)**
Learns to undo noise step-by-step:
\[
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
\]
- \( \mu_\theta \): Neural network predicting the mean.
- \( \Sigma_\theta \): Often fixed to \( \sigma_t^2 \mathbf{I} \).

---

### **2. Loss Function (Training Objective)**
Stable Diffusion minimizes the **noise prediction loss** (simplified ELBO):
\[
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
\]
- \( \epsilon \): Actual noise added during forward diffusion.
- \( \epsilon_\theta \): Neural network (U-Net) predicting noise.
- \( x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon \).

---

### **3. Latent Space Compression (VAE)**
Stable Diffusion works in a compressed latent space \( z \) to reduce compute:
\[
z = \text{Encoder}(x), \quad \hat{x} = \text{Decoder}(z)
\]
- **VAE Loss** (combines reconstruction + KL divergence):
  \[
  \mathcal{L}_{\text{VAE}} = \|x - \hat{x}\|^2 + \beta D_{KL}(q(z|x) \| p(z))
  \]
  - \( p(z) = \mathcal{N}(0, \mathbf{I}) \) (standard Gaussian prior).

---

### **4. Text Conditioning (CLIP Text Encoder)**
Text prompts are encoded into embeddings \( c \) via a **frozen CLIP model**:
\[
c = \text{CLIP}(\text{"prompt"})
\]
- The U-Net \( \epsilon_\theta \) is conditioned on \( c \):
  \[
  \epsilon_\theta(x_t, t, c)
  \]

---

### **5. Classifier-Free Guidance**
Improves sample quality by trading off conditional and unconditional predictions:
\[
\hat{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
\]
- \( s \): Guidance scale (higher = more adherence to text).
- \( \emptyset \): Null token (unconditional).

---

### **6. Sampling (DDIM)**
Stable Diffusion often uses **DDIM** for faster sampling:
\[
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t, c)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t, c)
\]

---

### **Key Formulas Summary**
| Concept                  | Formula                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| **Forward Diffusion**    | \( q(x_t \| x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)\mathbf{I}) \) |
| **Noise Prediction**     | \( \mathcal{L} = \mathbb{E} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right] \) |
| **VAE Latent Space**     | \( \mathcal{L}_{\text{VAE}} = \|x - \hat{x}\|^2 + \beta D_{KL}(q(z\|x) \| p(z)) \) |
| **Classifier-Free Guidance** | \( \hat{\epsilon}_\theta = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)) \) |
| **DDIM Sampling**        | \( x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta \) |

---

### **Why These Formulas Matter**
1. **Forward/Reverse Diffusion**: Core of image generation.
2. **Noise Prediction Loss**: Trains the U-Net to denoise.
3. **VAE**: Compresses images into latent space for efficiency.
4. **CLIP + Guidance**: Aligns generated images with text prompts.
5. **DDIM**: Enables fast, high-quality sampling.

For implementation, libraries like `diffusers` (Hugging Face) abstract these equations, but understanding them is crucial for customization. Would you like a code walkthrough?