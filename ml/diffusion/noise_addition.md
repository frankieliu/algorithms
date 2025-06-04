Yes! Let's walk through **how noise is added** in diffusion models—this is a core concept.

---

## 🧊 The Noise: Gaussian and Per-Pixel

### ❓ Is the noise added per pixel?

**Yes.**
Diffusion models add **independent Gaussian noise to each pixel/channel** in the image.

So if your image is shaped:

* **(C, H, W)** — Channels (e.g., RGB), Height, Width
  Then the noise added is of the **same shape**.

---

## 🔬 Step-by-Step: Adding Noise

Let’s say you have:

* $x_0$: the original clean image (normalized to values between $[-1, 1]$ or $[0, 1]$)
* $\epsilon \sim \mathcal{N}(0, I)$: random noise sampled from a standard normal distribution (same shape as $x_0$)
* $\alpha_t$: noise schedule coefficient at timestep $t$

Then the **noisy image $x_t$** is created as:

$$
x_t = \sqrt{\alpha_t} \cdot x_0 + \sqrt{1 - \alpha_t} \cdot \epsilon
$$

This equation **blends the original image and noise**, scaled appropriately:

* $\sqrt{\alpha_t}$: keeps part of the image
* $\sqrt{1 - \alpha_t}$: adds Gaussian noise

> As $t$ increases, $\alpha_t$ decreases → less of the image is retained, more noise dominates.

---

### 🧪 What does this mean visually?

* At $t = 0$:

  $$
  x_t = x_0
  $$

  (no noise added)

* At $t = T$:

  $$
  x_T \approx \epsilon
  $$

  (image is pure noise)

* At intermediate steps:
  The image is progressively more corrupted.

---

### 📈 Noise Schedule ($\alpha_t$)

The values of $\alpha_t$ are controlled by a **beta schedule**:

$$
\alpha_t = \prod_{s=1}^t (1 - \beta_s)
$$

Where $\beta_s \in (0, 1)$ are small numbers that increase slowly with $s$.
Common schedules:

* **Linear**: $\beta_t = \text{linspace}(10^{-4}, 0.02, T)$
* **Cosine** (used in Stable Diffusion): smoother denoising

---

## 📷 Example in Code (PyTorch-like)

```python
import torch

def add_noise(x0, t, alphas_cumprod):
    """
    x0: original image (batch_size, C, H, W)
    t: timestep (batch_size,) or scalar
    alphas_cumprod: precomputed α_t values (T,)
    """
    noise = torch.randn_like(x0)
    sqrt_alpha = alphas_cumprod[t] ** 0.5
    sqrt_one_minus_alpha = (1 - alphas_cumprod[t]) ** 0.5
    return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
```

Each pixel value in each channel is modified independently by adding noise from a normal distribution.

---

## 🔁 In Summary

| Concept         | Detail                                                                            |
| --------------- | --------------------------------------------------------------------------------- |
| **Noise shape** | Same as the image: (C, H, W)                                                      |
| **Added how?**  | Per-pixel, per-channel Gaussian noise                                             |
| **Equation**    | $x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon$                        |
| **Control**     | By schedule of $\alpha_t$ (based on $\beta_t$)                                    |
| **Purpose**     | To simulate gradual destruction of the image so the model can learn to reverse it |

---

Want a visualization of this process over time or a working notebook example?
