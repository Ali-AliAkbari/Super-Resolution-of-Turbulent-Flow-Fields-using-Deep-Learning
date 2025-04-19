Based on your code, the paper, and your usage of both the model architecture diagram from the paper and `torchviz` for your own model visualization, here's a more **academic and polished** version of your `README.md` file:

---

# 🌀 Super-Resolution of Turbulent Flow Fields using Deep Learning

This repository presents an implementation of a super-resolution neural network inspired by the methodology proposed in:

> **B. Hu et al.,**  
> *Super-resolution-assisted rapid high-fidelity CFD modeling of data centers*,  
> *Building and Environment*, Volume 247, 2024, 111036.  
> [DOI: 10.1016/j.buildenv.2023.111036](https://doi.org/10.1016/j.buildenv.2023.111036)

We apply their idea of reconstructing high-resolution flow fields from low-resolution CFD data to a **turbulent flow dataset**. The model learns to map coarse CFD data to finer-resolution outputs using a U-Net-like architecture enhanced with residual blocks.

---

## 📌 Key Contributions

- Implementation of a deep CNN for super-resolution of 2D turbulent fluid flow fields.
- Adaptation of the concept from data center cooling to a general-purpose turbulence dataset.
- Visualization and evaluation of predicted vs. ground truth fields.
- Use of `torchviz` for model structure visualization.
- Use of normalized preprocessing, residual connections, and skip connections in a U-Net architecture.

---

## 🧠 Inspiration

The methodology is based on this figure from the original paper:

<p align="center">
  <img src="images/framework_paper.png" width="500"/>
</p>

Our network aims to perform a similar super-resolution task, adapted for 2D incompressible turbulence flows.

---

## 🗂 Dataset

We used:
- Low-resolution irregular CFD data: [`kmflow_sampled_data_irregnew.npz`](https://figshare.com/ndownloader/files/39214622)
- High-resolution field data: [`kf_2d_re1000_256_40seed.npy`](https://figshare.com/ndownloader/files/39181919)

These datasets contain velocity field slices of 2D turbulent flow simulations.

---

## 🧱 Network Architecture

The network consists of:
- Initial convolution layer
- **Downsampling path**: residual blocks with max pooling
- **Bottleneck**: deep residual learning
- **Upsampling path**: transposed convolutions + skip connections
- Final output convolution

<p align="center">
  <img src="images/model_architecture.png" width="600"/>
</p>

_Model architecture rendered with `torchviz`._

---

## 🔧 Training Setup

- Loss: `MSELoss` (optionally `L1Loss`)
- Optimizer: `Adam` with learning rate decay using `ReduceLROnPlateau`
- Framework: PyTorch
- Hardware: GPU or CPU-compatible

```python
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
criterion = nn.MSELoss()


Training for 40 epochs (20+20) with gradient clipping and adaptive learning rate scheduling.

---

## 📊 Example Outputs

Here is a comparison between:
- **LR input** (coarse CFD result)
- **SR output** (predicted high-resolution)
- **Ground truth** (actual CFD result)

<p align="center">
  <img src="images/results_comparison.png" width="700"/>
</p>

The model successfully reconstructs fine-scale features missing from the low-resolution input.

---

## 📈 Loss Curve

<p align="center">
  <img src="images/loss_curve.png" width="500"/>
</p>

---

## 🧾 Citation

If you use this repository, please cite the original paper:

```
@article{hu2024superresolution,
  title={Super-resolution-assisted rapid high-fidelity CFD modeling of data centers},
  author={Hu, Beichao and Yin, Zeda and Hamrani, Abderrachid and Leon, Arturo and McDaniel, Dwayne},
  journal={Building and Environment},
  volume={247},
  pages={111036},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.buildenv.2023.111036}
}
```

---

## ⚠️ Licensing

The diagrams from the paper are used under **fair use** for educational and academic purposes. The main figure has been **recreated or included with proper citation**. Please do not reuse these figures commercially.

---

## 🧩 Future Work

- Extend model to 3D turbulence fields.
- Investigate the use of POD or spectral loss functions.
- Compare performance with diffusion and transformer-based models.

---

## 📬 Contact

For questions or collaborations, feel free to reach out.

---

## ✅ To Do

- [x] Load and preprocess CFD data
- [x] Implement residual U-Net
- [x] Visualize model using `torchviz`
- [x] Compare SR output with ground truth
- [ ] Add SSIM/PSNR metrics
- [ ] Experiment with GAN-based SR

---
