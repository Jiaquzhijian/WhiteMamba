import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Utilities: normalization, surface area, approx A_d(kappa), approx log C_d(kappa)
# ============================================================

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def log_surface_area_sphere(p: int, device=None, dtype=None) -> torch.Tensor:
    """
    Surface area of S^{p-1}: |S^{p-1}| = 2 * pi^{p/2} / Gamma(p/2)
    return log |S^{p-1}|
    """
    two = torch.tensor(2.0, device=device, dtype=dtype)
    pi = torch.tensor(math.pi, device=device, dtype=dtype)
    return torch.log(two) + (p / 2.0) * torch.log(pi) - torch.lgamma(torch.tensor(p / 2.0, device=device, dtype=dtype))


def A_p_approx(kappa: torch.Tensor, p: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Approximation to A_p(kappa) = I_{p/2}(kappa) / I_{p/2-1}(kappa)
    A practical differentiable approximation (good in high-d):
        A ≈ kappa / (p/2 + sqrt(kappa^2 + (p/2)^2))
    - small kappa: ~ kappa / p
    - large kappa: ~ 1 - p/(2kappa) (close to true 1-(p-1)/(2kappa))
    """
    halfp = 0.5 * p
    halfp_t = torch.tensor(halfp, device=kappa.device, dtype=kappa.dtype)
    return kappa / (halfp_t + torch.sqrt(kappa * kappa + halfp_t * halfp_t) + eps)


def log_C_p_approx(kappa: torch.Tensor, p: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Approximation to log C_p(kappa) where vMF density:
        vMF(z; mu, kappa) = C_p(kappa) * exp(kappa * mu^T z), z in S^{p-1}
    We approximate:
    - small kappa: C_p(kappa) ~ 1/|S| => logC_small = -log|S|
    - large kappa asymptotic using I_v(kappa) ~ exp(kappa)/sqrt(2 pi kappa):
        logC_large ≈ ((p-1)/2)*(log kappa - log(2pi)) - kappa
    Smoothly interpolate between the two regimes.
    """
    device, dtype = kappa.device, kappa.dtype
    logS = log_surface_area_sphere(p, device=device, dtype=dtype)
    logC_small = -logS

    # asymptotic large-kappa
    log2pi = torch.tensor(math.log(2.0 * math.pi), device=device, dtype=dtype)
    logC_large = 0.5 * (p - 1) * (torch.log(kappa + eps) - log2pi) - kappa

    # smooth gate: around kappa0
    kappa0 = torch.tensor(10.0, device=device, dtype=dtype)
    sharp = torch.tensor(2.0, device=device, dtype=dtype)
    w = torch.sigmoid((kappa - kappa0) / sharp)  # 0 small, 1 large
    return (1.0 - w) * logC_small + w * logC_large


def kl_vmf_to_uniform_approx(kappa: torch.Tensor, p: int) -> torch.Tensor:
    """
    KL( vMF(mu,kappa) || Uniform(S^{p-1}) )
      = E_q[log q(z)] - E_q[log p0(z)]
      = (log C_p(kappa) + kappa * A_p(kappa)) + log |S^{p-1}|
    """
    logC = log_C_p_approx(kappa, p)
    A = A_p_approx(kappa, p)
    logS = log_surface_area_sphere(p, device=kappa.device, dtype=kappa.dtype)
    return logC + kappa * A + logS


# ============================================================
# Geometry: tangent projection + Exp map on S^{p-1} (p = dim)
# ============================================================

def tangent_project(e: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
    """
    Project e onto tangent space at p0:
        v = e - <e,p0> p0
    p0: unit vector (p,)
    e: (B,p)
    """
    dot = (e * p0).sum(dim=-1, keepdim=True)
    return e - dot * p0


def exp_map_sphere(v: torch.Tensor, p0: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Exp_{p0}(v) on unit sphere:
        if ||v||>0: cos(||v||) p0 + sin(||v||) v/||v||
        else: p0
    v is assumed in tangent space at p0 (orthogonal to p0).
    """
    vnorm = v.norm(dim=-1, keepdim=True)  # (B,1)
    # safe normalized direction
    v_dir = v / (vnorm + eps)
    cos = torch.cos(vnorm)
    sin = torch.sin(vnorm)
    out = cos * p0 + sin * v_dir
    # if vnorm==0, out becomes p0 automatically (sin=0, cos=1)
    return l2_normalize(out, dim=-1)


# ============================================================
# vMF sampler on S^{p-1}: 重参数化采样 (kappa固定为1)
# ============================================================

class PrecomputedWSampler:
    """
    预计算W采样器，用于kappa=1的情况
    参考: https://arxiv.org/abs/1812.04616
    """

    def __init__(self, dims: int, num_samples: int = 1000000, device='cpu'):
        """
        初始化采样器，预计算w样本
        dims: 球面维度(p)
        num_samples: 预计算的样本数量
        """
        self.dims = dims
        self.num_samples = num_samples
        self.device = device
        self.W = None

        # 预计算W
        self._precompute_W()

    def _precompute_W(self):
        """预计算W样本池"""
        epsilon = 1e-7
        x = np.arange(-1 + epsilon, 1, epsilon)
        y = 1.0 * x + np.log(1 - x ** 2) * (self.dims - 3) / 2  # kappa=1
        y = np.cumsum(np.exp(y - y.max()))
        y = y / y[-1]

        # 生成随机数并插值
        u = np.random.random(self.num_samples)
        W = np.interp(u, y, x)

        # 转换为tensor并移到设备
        self.W = torch.tensor(W, dtype=torch.float32, device=self.device)

    def sample(self, mu: torch.Tensor) -> torch.Tensor:
        """
        采样 z ~ vMF(mu, kappa=1)
        mu: (B, p) 单位向量
        return: (B, p) 单位向量
        """
        B, p = mu.shape
        assert p == self.dims, f"维度不匹配: mu维度={p}, 采样器维度={self.dims}"

        # 随机选择w
        idxs = torch.randint(0, self.num_samples, (B,), device=mu.device)
        w = self.W[idxs].unsqueeze(-1)  # (B, 1)

        # 采样随机方向nu（与mu正交）
        eps = torch.randn_like(mu)
        nu = eps - torch.sum(eps * mu, dim=-1, keepdim=True) * mu
        nu = l2_normalize(nu, dim=-1)

        # 组合得到样本
        return w * mu + torch.sqrt(1 - w ** 2 + 1e-8) * nu

    def to(self, device):
        """将采样器移到指定设备"""
        self.device = device
        if self.W is not None:
            self.W = self.W.to(device)
        return self


def sample_unit_sphere(shape: Tuple[int, int], device=None, dtype=None, eps: float = 1e-8) -> torch.Tensor:
    """在单位球面上均匀采样"""
    x = torch.randn(*shape, device=device, dtype=dtype)
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def vmf_sample(mu: torch.Tensor, kappa: torch.Tensor, max_iters: int = 100, eps: float = 1e-8) -> torch.Tensor:
    """
    通用的vMF采样器
    mu: (B,p) 单位向量
    kappa: (B,) 非负
    return: (B,p) 单位向量
    """
    B, p = mu.shape
    device, dtype = mu.device, mu.dtype

    # 如果kappa接近0 -> 球面均匀分布
    if torch.all(kappa <= 1e-6):
        return sample_unit_sphere((B, p), device=device, dtype=dtype)

    # 检查是否所有kappa都接近1（固定kappa=1的情况）
    if torch.all(torch.abs(kappa - 1.0) < 1e-3):
        # 使用预计算采样器（如果有）
        if hasattr(mu, '_precomputed_sampler'):
            sampler = mu._precomputed_sampler
        else:
            # 创建并缓存采样器
            sampler = PrecomputedWSampler(p, device=device)
            mu._precomputed_sampler = sampler
        return sampler.sample(mu)

    # 否则使用Wood算法（原算法）
    kappa = kappa.clamp_min(0.0)

    # Precompute constants
    p_minus_1 = p - 1.0
    sqrt_term = torch.sqrt(4.0 * kappa * kappa + p_minus_1 * p_minus_1)
    b = (-2.0 * kappa + sqrt_term) / p_minus_1
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + p_minus_1 * torch.log(1.0 - x0 * x0 + eps)

    # Rejection sampling for w
    w = torch.zeros(B, device=device, dtype=dtype)
    done = torch.zeros(B, device=device, dtype=torch.bool)

    for _ in range(max_iters):
        idx = (~done).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            break

        u1 = torch.rand(idx.numel(), device=device, dtype=dtype)
        u2 = torch.rand(idx.numel(), device=device, dtype=dtype)

        b_i = b[idx]
        x0_i = x0[idx]
        c_i = c[idx]
        k_i = kappa[idx]

        # proposed w
        w_prop = (1.0 - (1.0 + b_i) * u1) / (1.0 - (1.0 - b_i) * u1 + eps)
        t = k_i * w_prop + p_minus_1 * torch.log(1.0 - x0_i * w_prop + eps) - c_i

        accept = t >= torch.log(u2 + eps)
        if accept.any():
            acc_idx = idx[accept]
            w[acc_idx] = w_prop[accept]
            done[acc_idx] = True

    # If still not done for some samples, fallback to near-uniform for those
    if (~done).any():
        w[~done] = 2.0 * torch.rand((~done).sum(), device=device, dtype=dtype) - 1.0

    # Sample v uniform on S^{p-2}
    v = sample_unit_sphere((B, p - 1), device=device, dtype=dtype)
    # Compose in the coordinate system where mu is e1
    z = torch.cat([w.unsqueeze(-1), torch.sqrt((1.0 - w * w).clamp_min(0.0)).unsqueeze(-1) * v], dim=-1)  # (B,p)

    # Householder transform to rotate e1 -> mu
    e1 = torch.zeros((B, p), device=device, dtype=dtype)
    e1[:, 0] = 1.0

    u = e1 - mu  # (B,p)
    u_norm_sq = (u * u).sum(dim=-1, keepdim=True)  # (B,1)
    # If mu == e1, no rotation needed
    mask = (u_norm_sq.squeeze(-1) > 1e-12)

    z_rot = z.clone()
    if mask.any():
        u_m = u[mask]
        z_m = z[mask]
        u_norm_sq_m = u_norm_sq[mask]
        proj = (z_m * u_m).sum(dim=-1, keepdim=True) / (u_norm_sq_m + eps)
        z_rot[mask] = z_m - 2.0 * proj * u_m

    return l2_normalize(z_rot, dim=-1)


def sliced_wasserstein_2(
        z: torch.Tensor,
        u: torch.Tensor,
        num_projections: int = 256,
        eps: float = 1e-8,
        fixed_theta: torch.Tensor = None,  # 新增：固定投影矩阵
) -> torch.Tensor:
    """
    Correct Sliced Wasserstein-2 distance with fixed projection directions.
    """
    assert z.dim() == 2 and u.dim() == 2
    B, p = z.shape
    assert u.shape == (B, p)

    # 使用固定投影方向或随机生成
    if fixed_theta is not None:
        theta = fixed_theta
    else:
        theta = torch.randn(num_projections, p, device=z.device, dtype=z.dtype)
        theta = theta / (theta.norm(dim=-1, keepdim=True) + eps)

    # 1D projections: (B,L)
    z_proj = z @ theta.t()  # (B, L)
    u_proj = u @ theta.t()  # (B, L)

    # 对每个投影维度分别排序
    z_sorted = torch.sort(z_proj, dim=0).values  # (B, L)
    u_sorted = torch.sort(u_proj, dim=0).values  # (B, L)

    # 计算每个投影维度的Wasserstein距离
    w2_per_proj = ((z_sorted - u_sorted) ** 2).mean(dim=0)  # (L,)
    return w2_per_proj.mean()  # scalar


# ============================================================
# Networks
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HypersphereEncoder(nn.Module):
    """
    Unified symbol: mu_psi(x,t) in S^{p-1}, psi=(theta,phi)
    - fusion f_theta: concatenate then MLP -> e in R^p
    - tangent projection at p0
    - Exp map -> tilde_mu in S^{p-1}
    - learnable g_phi: MLP then normalize -> mu in S^{p-1}
    """

    def __init__(self, x_dim: int, t_dim: int, p: int, fusion_hidden: int = 512, proj_hidden: int = 512):
        super().__init__()
        self.p = p
        self.f_theta = MLP(x_dim + t_dim, p, hidden=fusion_hidden, depth=3, dropout=0.0)
        self.g_phi = MLP(p, p, hidden=proj_hidden, depth=2, dropout=0.0)

        # fixed reference point p0 = [1,0,...,0]
        p0 = torch.zeros(p)
        p0[0] = 1.0
        self.register_buffer("p0", p0)

        # 预计算采样器（用于Stage3）
        self._precomputed_sampler = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Stage1 fusion
        e = self.f_theta(torch.cat([x, t], dim=-1))  # (B,p)
        # Tangent projection + Exp map
        v = tangent_project(e, self.p0)  # (B,p)
        tilde_mu = exp_map_sphere(v, self.p0)  # (B,p), unit
        # Learnable refinement on sphere
        mu = self.g_phi(tilde_mu)
        mu = l2_normalize(mu, dim=-1)

        # 为mu添加采样器引用
        if self._precomputed_sampler is None:
            self._precomputed_sampler = PrecomputedWSampler(self.p, device=mu.device)
        mu._precomputed_sampler = self._precomputed_sampler

        return mu


class KappaNet(nn.Module):
    """
    kappa_eta(x,t) >= 0 (independent of mu)
    Input x, t -> scalar kappa (B,)
    """

    def __init__(self, x_dim: int, t_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = MLP(x_dim + t_dim, 1, hidden=hidden, depth=3, dropout=0.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        k = self.mlp(torch.cat([x, t], dim=-1)).squeeze(-1)
        # softplus ensures positivity
        return F.sigmoid(k) + 1e-6


class SphereDecoder(nn.Module):
    """
    r_xi: S^{p-1} -> S^{p-1} (implemented as MLP + normalize)
    """

    def __init__(self, p: int, hidden: int = 512):
        super().__init__()
        self.mlp = MLP(p, p, hidden=hidden, depth=3, dropout=0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.mlp(z)
        return l2_normalize(out, dim=-1)


# ============================================================
# Stage2 Losses: consistency, triplet, uniformity
# ============================================================

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1)


def align_loss(x: torch.Tensor, y: torch.Tensor, alpha: float = 2) -> torch.Tensor:
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x: torch.Tensor, t: float = 2) -> torch.Tensor:
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def consistency_loss(mu_a: torch.Tensor, mu_p: torch.Tensor, alpha: float = 2) -> torch.Tensor:
    """一致性损失（对齐损失）：使用欧氏距离的 alpha 次幂。"""
    return align_loss(mu_a, mu_p, alpha=alpha)


def triplet_loss_cos(mu_a: torch.Tensor, mu_p: torch.Tensor, mu_n: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """
    d(u,v) = 1 - cos(u,v)
    L_tri = max(0, d(a,p) - d(a,n) + margin)
    """
    dap = 1.0 - cosine_sim(mu_a, mu_p)
    dan = 1.0 - cosine_sim(mu_a, mu_n)
    return F.relu(dap - dan + margin).mean()


def uniformity_loss(mu_all: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """均匀性损失：使用 pdist 的实现（与给定 uniform_loss 一致）。"""
    return uniform_loss(mu_all, t=t)


# ============================================================
# Stage3 ELBO objective (Spherical VAE style)
# ============================================================

def stage3_objective_spherical_vae(
        mu: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        kappa_net: nn.Module,
        decoder: nn.Module,
        beta: float = 1.0,
        gamma: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Spherical VAE objective:
      ELBO = E[log p(x|z)] - KL(q(z|x,t) || p(z))
      where q(z|x,t) = vMF(z; mu, kappa(x,t))
            p(z) = uniform on sphere
      So: loss = -E[log p(x|z)] + beta * KL(vMF || uniform)
    """
    # Get kappa from x,t (independent of mu)
    kappa = kappa_net(x, t)  # (B,)

    # Sample from posterior: z ~ vMF(mu, kappa)
    z = vmf_sample(mu, kappa)  # (B, p)

    # Reconstruction: mu_hat = decoder(z)
    mu_hat = decoder(z)  # (B, p)

    # Reconstruction loss: cosine similarity (1 - cos similarity)
    cos_sim = cosine_sim(mu, mu_hat)  # (B,)
    recon_loss = (1.0 - cos_sim).mean() * gamma

    # KL divergence: KL(vMF || uniform)
    kl_loss = kl_vmf_to_uniform_approx(kappa, p=mu.shape[-1]).mean()

    # Total loss
    loss = recon_loss + beta * kl_loss

    return {
        "loss": loss,
        "recon": recon_loss,
        "kl": kl_loss,
        "cos": cos_sim.mean(),
        "kappa_mean": kappa.mean(),
        "kappa_max": kappa.max(),
    }


def stage3_objective_swae_spherical_vae(
        mu: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        kappa_net: nn.Module,
        decoder: nn.Module,
        beta: float = 1.0,
        gamma: float = 1.0,
        num_projections: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    Spherical VAE with SWAE-style regularization:
      loss = recon(mu, r(z)) + beta * SW2( z ~ vMF(mu, kappa(x,t)), u ~ uniform )
    """
    # Get kappa from x,t
    kappa = kappa_net(x, t)  # (B,)

    # Sample from posterior: z ~ vMF(mu, kappa)
    z = vmf_sample(mu, kappa)  # (B, p)

    # Reconstruction
    mu_hat = decoder(z)  # (B, p)
    cos_sim = cosine_sim(mu, mu_hat)  # (B,)
    recon_loss = (1.0 - cos_sim).mean() * gamma

    # Prior sample: u ~ Uniform(S^{p-1})
    u = sample_unit_sphere((mu.shape[0], mu.shape[1]), device=mu.device, dtype=mu.dtype)

    # Sliced Wasserstein distance
    sw2 = sliced_wasserstein_2(z, u, num_projections=num_projections)

    # Total loss
    loss = recon_loss + beta * sw2

    return {
        "loss": loss,
        "recon": recon_loss,
        "sw2": sw2,
        "cos": cos_sim.mean(),
        "kappa_mean": kappa.mean(),
        "kappa_max": kappa.max(),
    }


# ============================================================
# Training loops
# ============================================================

@dataclass
class Stage2Config:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    margin: float = 0.2
    lam_tri: float = 1.0
    lam_unif: float = 0.1
    unif_t: float = 2.0
    epochs: int = 5
    beta: int = 10
    warmup_epochs: int = 10


@dataclass
class Stage3Config:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    beta: float = 1.0
    gamma: float = 1.0
    epochs: int = 5
    kappa_clip: float = 20.0  # optional stability clip
    num_projections: int = 64
    use_swae: bool = True  # whether to use SWAE or standard VAE objective
    warmup_epochs: int = 10


def train_stage2(
        encoder: HypersphereEncoder,
        loader,
        cfg: Stage2Config,
        device: torch.device,
        log_every: int = 50,
) -> None:
    encoder.to(device)
    encoder.train()
    opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    global_step = 0
    for ep in range(cfg.epochs):
        # epoch accumulators
        sum_loss = sum_lc = sum_lt = sum_lu = 0.0
        sum_cap = sum_can = 0.0
        n_batches = 0

        for batch in loader:
            global_step += 1
            n_batches += 1

            x_a = batch["x_a"].to(device)
            t_a = batch["t_a"].to(device)
            x_p = batch["x_p"].to(device)
            t_p = batch["t_p"].to(device)
            x_n = batch["x_n"].to(device)
            t_n = batch["t_n"].to(device)

            mu_a = encoder(x_a, t_a)
            mu_p = encoder(x_p, t_p)
            mu_n = encoder(x_n, t_n)

            Lc = consistency_loss(mu_a, mu_p)
            Lt = triplet_loss_cos(mu_a, mu_p, mu_n, margin=cfg.margin)

            mu_all = torch.cat([mu_a, mu_p, mu_n], dim=0)
            Lu = uniformity_loss(mu_all, t=cfg.unif_t)

            loss = Lc + cfg.lam_tri * Lt + cfg.lam_unif * Lu

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # stats
            cap = cosine_sim(mu_a, mu_p).mean().item()
            can = cosine_sim(mu_a, mu_n).mean().item()

            sum_loss += loss.item()
            sum_lc += Lc.item()
            sum_lt += Lt.item()
            sum_lu += Lu.item()
            sum_cap += cap
            sum_can += can

            if global_step % log_every == 0:
                print(
                    f"[Stage2][ep {ep + 1}/{cfg.epochs} step {global_step}] "
                    f"loss={sum_loss / n_batches:.4f} "
                    f"Lc={sum_lc / n_batches:.4f} Lt={sum_lt / n_batches:.4f} Lu={sum_lu / n_batches:.4f} "
                    f"cos(a,p)={sum_cap / n_batches:.4f} cos(a,n)={sum_can / n_batches:.4f}"
                )

        # epoch end print
        print(
            f"[Stage2][ep {ep + 1}/{cfg.epochs} DONE] "
            f"loss={sum_loss / n_batches:.4f} "
            f"Lc={sum_lc / n_batches:.4f} Lt={sum_lt / n_batches:.4f} Lu={sum_lu / n_batches:.4f} "
            f"cos(a,p)={sum_cap / n_batches:.4f} cos(a,n)={sum_can / n_batches:.4f}"
        )


def train_stage3(
        encoder_frozen: HypersphereEncoder,
        kappa_net: KappaNet,
        decoder: SphereDecoder,
        loader,
        cfg: Stage3Config,
        device: torch.device,
        log_every: int = 50,
) -> None:
    encoder_frozen.to(device).eval()
    for p in encoder_frozen.parameters():
        p.requires_grad_(False)

    decoder.to(device).train()

    opt = torch.optim.AdamW(
        list(decoder.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # 固定投影方向，在整个epoch中保持一致
    fixed_theta = torch.randn(cfg.num_projections, encoder_frozen.p, device=device)
    fixed_theta = fixed_theta / (fixed_theta.norm(dim=-1, keepdim=True) + 1e-8)

    global_step = 0
    for ep in range(cfg.epochs):
        sum_loss = sum_recon = sum_reg = sum_cos = 0.0
        sum_km = sum_kmax = 0.0
        n_batches = 0
        beta = min(cfg.beta, cfg.beta * (ep + 1) / cfg.warmup_epochs)
        for batch in loader:
            global_step += 1
            n_batches += 1

            x = batch["x"].to(device)
            t = batch["t"].to(device)

            with torch.no_grad():
                mu = encoder_frozen(x, t)  # (B,p)

            if cfg.use_swae:
                # 使用固定kappa=1
                kappa = torch.ones(mu.shape[0], device=mu.device)

                # 使用预计算采样器（更高效）
                if hasattr(encoder_frozen, '_precomputed_sampler'):
                    sampler = encoder_frozen._precomputed_sampler
                    z = sampler.sample(mu)
                else:
                    z = vmf_sample(mu, kappa)

                mu_hat = decoder(z)  # (B, p)
                cos_sim = cosine_sim(mu, mu_hat)  # (B,)
                recon_loss = (1.0 - cos_sim).mean() * cfg.gamma

                # Prior sample: u ~ Uniform(S^{p-1})
                u = sample_unit_sphere((mu.shape[0], mu.shape[1]), device=mu.device, dtype=mu.dtype)

                # 使用固定投影方向计算SW2
                sw2 = sliced_wasserstein_2(z, u, fixed_theta=fixed_theta)

                loss = recon_loss + beta * sw2

                outs = {
                    "loss": loss,
                    "recon": recon_loss,
                    "sw2": sw2,
                    "cos": cos_sim.mean(),
                    "kappa_mean": kappa.mean(),
                    "kappa_max": kappa.max(),
                }
            else:
                outs = stage3_objective_spherical_vae(
                    mu=mu,
                    x=x,
                    t=t,
                    kappa_net=kappa_net,
                    decoder=decoder,
                    beta=cfg.beta,
                    gamma=cfg.gamma,
                )

            loss = outs["loss"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # stats
            km = outs.get("kappa_mean", torch.tensor(0.0)).item()
            kmax = outs.get("kappa_max", torch.tensor(0.0)).item()

            sum_loss += loss.item()
            sum_recon += outs["recon"].item()
            sum_reg += outs.get("kl", outs.get("sw2", torch.tensor(0.0))).item()
            sum_cos += outs["cos"].item()
            sum_km += km
            sum_kmax += kmax

            reg_type = "kl" if not cfg.use_swae else "sw2"
            if global_step % log_every == 0:
                print(
                    f"[Stage3][ep {ep + 1}/{cfg.epochs} step {global_step}] "
                    f"loss={sum_loss / n_batches:.4f} recon={sum_recon / n_batches:.4f} "
                    f"{reg_type}={sum_reg / n_batches:.6f} cos={sum_cos / n_batches:.4f} "
                    f"kappa(mean/max)={sum_km / n_batches:.3f}/{sum_kmax / n_batches:.3f}"
                )

        print(
            f"[Stage3][ep {ep + 1}/{cfg.epochs} DONE] "
            f"loss={sum_loss / n_batches:.4f} recon={sum_recon / n_batches:.4f} "
            f"{reg_type}={sum_reg / n_batches:.6f} cos={sum_cos / n_batches:.4f} "
            f"kappa(mean/max)={sum_km / n_batches:.3f}/{sum_kmax / n_batches:.3f}"
        )


# ============================================================
# Example dataset wrappers
# ============================================================

class StructuredTripletLoader:
    """
    Synthetic structured triplets:
      latent s ~ N(0, I_k)
      x = A s + noise, t = B s + noise
      positive shares same s (small perturb), negative uses different s'
    """

    def __init__(self, steps=200, B=64, x_dim=128, t_dim=128, k=16, noise=0.1, pos_noise=0.02, device="cpu"):
        self.steps, self.B = steps, B
        self.x_dim, self.t_dim = x_dim, t_dim
        self.k = k
        self.noise = noise
        self.pos_noise = pos_noise
        self.device = device

        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        self.A = torch.randn(x_dim, k, generator=g) / math.sqrt(k)
        self.Bm = torch.randn(t_dim, k, generator=g) / math.sqrt(k)

    def __iter__(self):
        for _ in range(self.steps):
            s_a = torch.randn(self.B, self.k)
            s_p = s_a + self.pos_noise * torch.randn_like(s_a)
            s_n = torch.randn(self.B, self.k)

            x_a = s_a @ self.A.t() + self.noise * torch.randn(self.B, self.x_dim)
            t_a = s_a @ self.Bm.t() + self.noise * torch.randn(self.B, self.t_dim)

            x_p = s_p @ self.A.t() + self.noise * torch.randn(self.B, self.x_dim)
            t_p = s_p @ self.Bm.t() + self.noise * torch.randn(self.B, self.t_dim)

            x_n = s_n @ self.A.t() + self.noise * torch.randn(self.B, self.x_dim)
            t_n = s_n @ self.Bm.t() + self.noise * torch.randn(self.B, self.t_dim)

            yield {"x_a": x_a, "t_a": t_a, "x_p": x_p, "t_p": t_p, "x_n": x_n, "t_n": t_n}


class StructuredPairLoader:
    """
    Synthetic structured pairs (x,t) sharing latent s.
    """

    def __init__(self, steps=200, B=64, x_dim=128, t_dim=128, k=16, noise=0.1):
        self.steps, self.B = steps, B
        self.x_dim, self.t_dim = x_dim, t_dim
        self.k = k
        self.noise = noise

        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        self.A = torch.randn(x_dim, k, generator=g) / math.sqrt(k)
        self.Bm = torch.randn(t_dim, k, generator=g) / math.sqrt(k)

    def __iter__(self):
        for _ in range(self.steps):
            s = torch.randn(self.B, self.k)
            x = s @ self.A.t() + self.noise * torch.randn(self.B, self.x_dim)
            t = s @ self.Bm.t() + self.noise * torch.randn(self.B, self.t_dim)
            yield {"x": x, "t": t}


# ============================================================
# Quick smoke test
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda")

    x_dim, t_dim = 128, 128
    p = 64

    # Test SWAE objective with precomputed sampler
    encoder2 = HypersphereEncoder(x_dim=x_dim, t_dim=t_dim, p=p)
    stage2_loader2 = StructuredTripletLoader(steps=100, B=64, x_dim=x_dim, t_dim=t_dim, k=16, noise=0.1)
    train_stage2(encoder2, stage2_loader2, Stage2Config(epochs=1), device=device, log_every=50)

    kappa_net2 = KappaNet(x_dim=x_dim, t_dim=t_dim, hidden=256)
    decoder2 = SphereDecoder(p=p)

    stage3_loader2 = StructuredPairLoader(steps=1000, B=64, x_dim=x_dim, t_dim=t_dim, k=16, noise=0.1)

    train_stage3(encoder2, kappa_net2, decoder2, stage3_loader2,
                 Stage3Config(epochs=10, beta=100.0, use_swae=True),
                 device=device, log_every=50)

    print("OK - SWAE objective with precomputed sampler completed")

# ============================================================
# Integration: Dataset_modified_1 -> Stage2/Stage3 loaders
# ============================================================
# This section wires GoStoneTripletDataset + TripletCollator outputs
# into the numeric tensors expected by train_stage2/train_stage3.
#
# Expected by train_stage2:
#   {"x_a","t_a","x_p","t_p","x_n","t_n"}  where each is FloatTensor (B, dim)
#
# Expected by train_stage3:
#   {"x","t"} where each is FloatTensor (B, dim)
#
# The Dataset_modified_1 collator returns raw PIL images + raw text strings;
# we therefore need a backbone to produce embeddings.

from typing import Iterable, Callable, Union
from torch.utils.data import DataLoader


class ImageTextBackbone(nn.Module):
    """
    Minimal backbone interface needed by this pipeline.

    Implementations MUST provide:
      - encode_image(List[PIL.Image]) -> FloatTensor(B, D)
      - encode_text(List[str])        -> FloatTensor(B, D)

    Note:
      - Returned embeddings should be float32/float16 tensors on the SAME device as the backbone.
      - If you use CLIP-style models, you typically L2-normalize embeddings; in this pipeline you may
        choose to normalize here or let the downstream encoder handle scaling (encoder uses an MLP
        so either is OK; empirically normalized is usually stabler).
    """
    def encode_image(self, images):
        raise NotImplementedError

    def encode_text(self, texts):
        raise NotImplementedError


@torch.no_grad()
def _ensure_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype not in (torch.float32, torch.float16, torch.bfloat16) else x


class TripletEmbeddingLoader:
    """
    Wrap a DataLoader over Dataset_modified_1 outputs and yield numeric tensors for Stage2.

    Input batch format (from Dataset_modified_1.TripletCollator):
      batch["anchor"]["image"]   : List[PIL]
      batch["anchor"]["text"]    : List[str]
      batch["positive"]["image"] : List[PIL]
      batch["positive"]["text"]  : List[str]
      batch["negatives"]["image"]: List[PIL] if K==1 else List[List[PIL]]
      batch["negatives"]["text"] : List[str] if K==1 else List[List[str]]

    Output:
      {"x_a","t_a","x_p","t_p","x_n","t_n"} on CPU (train_stage2 moves them to device).
    """

    def __init__(
        self,
        dataloader: DataLoader,
        backbone: ImageTextBackbone,
        device: Union[str, torch.device] = "cuda",
        negative_index: int = 0,  # when K>1
        normalize_embeddings: bool = True,
    ):
        self.dataloader = dataloader
        self.backbone = backbone.to(device).eval()
        self.device = torch.device(device)
        self.negative_index = int(negative_index)
        self.normalize_embeddings = bool(normalize_embeddings)

    def __iter__(self):
        for batch in self.dataloader:
            # Move backbone inputs to GPU only inside backbone (PIL/text stay python objects)
            a_img = batch["anchor"]["image"]
            a_txt = batch["anchor"]["text"]
            p_img = batch["positive"]["image"]
            p_txt = batch["positive"]["text"]

            n_img = batch["negatives"]["image"]
            n_txt = batch["negatives"]["text"]
            K = int(batch.get("num_negatives", 1))

            if K == 1:
                n_img_sel = n_img
                n_txt_sel = n_txt
            else:
                # (B,K) python lists -> select K dimension
                n_img_sel = [row[self.negative_index] for row in n_img]
                n_txt_sel = [row[self.negative_index] for row in n_txt]

            # Encode on GPU
            # x_a = _ensure_float(self.backbone.encode_image(a_img).to(self.device))
            # t_a = _ensure_float(self.backbone.encode_text(a_txt).to(self.device))
            # x_p = _ensure_float(self.backbone.encode_image(p_img).to(self.device))
            # t_p = _ensure_float(self.backbone.encode_text(p_txt).to(self.device))
            # x_n = _ensure_float(self.backbone.encode_image(n_img_sel).to(self.device))
            # t_n = _ensure_float(self.backbone.encode_text(n_txt_sel).to(self.device))
            x_a, t_a = self.backbone(a_img, a_txt)
            x_a = _ensure_float(x_a.to(self.device))
            t_a = _ensure_float(t_a.to(self.device))

            x_p, t_p = self.backbone(p_img, p_txt)
            x_p = _ensure_float(x_p.to(self.device))
            t_p = _ensure_float(t_p.to(self.device))

            x_n, t_n = self.backbone(n_img_sel, n_txt_sel)
            x_n = _ensure_float(x_n.to(self.device))
            t_n = _ensure_float(t_n.to(self.device))

            if self.normalize_embeddings:
                x_a = l2_normalize(x_a, dim=-1)
                t_a = l2_normalize(t_a, dim=-1)
                x_p = l2_normalize(x_p, dim=-1)
                t_p = l2_normalize(t_p, dim=-1)
                x_n = l2_normalize(x_n, dim=-1)
                t_n = l2_normalize(t_n, dim=-1)

            # Return to CPU for the existing training loop
            yield {
                "x_a": x_a.detach().cpu(),
                "t_a": t_a.detach().cpu(),
                "x_p": x_p.detach().cpu(),
                "t_p": t_p.detach().cpu(),
                "x_n": x_n.detach().cpu(),
                "t_n": t_n.detach().cpu(),
            }


class PairEmbeddingLoader:
    """
    Wrap a DataLoader over Dataset_modified_1 outputs and yield numeric tensors for Stage3.

    Strategy:
      - Use anchor pairs only: (x,t) = (encode_image(anchor.image), encode_text(anchor.text))
      - Optionally you can choose positive pairs instead; modify `pair_role`.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        backbone: ImageTextBackbone,
        device: Union[str, torch.device] = "cuda",
        pair_role: str = "anchor",  # "anchor" or "positive"
        normalize_embeddings: bool = True,
    ):
        self.dataloader = dataloader
        self.backbone = backbone.to(device).eval()
        self.device = torch.device(device)
        if pair_role not in ("anchor", "positive"):
            raise ValueError("pair_role must be 'anchor' or 'positive'")
        self.pair_role = pair_role
        self.normalize_embeddings = bool(normalize_embeddings)

    def __iter__(self):
        for batch in self.dataloader:
            imgs = batch[self.pair_role]["image"]
            txts = batch[self.pair_role]["text"]
            #
            x, t = self.backbone(imgs, txts)

            # x = _ensure_float(self.backbone.encode_image(imgs).to(self.device))
            # t = _ensure_float(self.backbone.encode_text(txts).to(self.device))
            x = _ensure_float(x.to(self.device))
            t = _ensure_float(t.to(self.device))
            if self.normalize_embeddings:
                x = l2_normalize(x, dim=-1)
                t = l2_normalize(t, dim=-1)

            yield {"x": x.detach().cpu(), "t": t.detach().cpu()}


# ============================================================
# Example: wiring GoStoneTripletDataset into this pipeline
# ============================================================
# IMPORTANT:
# - You must provide a concrete backbone implementation.
# - If you are using Chinese-CLIP, implement ImageTextBackbone via its processor + model.
# - The only requirement is that encode_image/encode_text return same embedding dim D.
#
# Below we provide a small placeholder backbone for debugging only.

class _DummyBackbone(ImageTextBackbone):
    """Debug-only backbone: returns random embeddings with correct shapes."""
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    @torch.no_grad()
    def encode_image(self, images):
        B = len(images)
        return torch.randn(B, self.dim, device=next(self.parameters(), torch.empty(0)).device)

    @torch.no_grad()
    def encode_text(self, texts):
        B = len(texts)
        return torch.randn(B, self.dim, device=next(self.parameters(), torch.empty(0)).device)

# ============================================================
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

# class ChineseCLIPBackbone(nn.Module):
#     def __init__(self, model_id="OFA-Sys/chinese-clip-vit-base-patch16", device="cuda"):
#         super().__init__()
#         self.processor = ChineseCLIPProcessor.from_pretrained(model_id)
#         self.clip = ChineseCLIPModel.from_pretrained(model_id).to(device).eval()
#         for p in self.clip.parameters():
#             p.requires_grad = False
#         self.device = device

#     @torch.no_grad()
#     def forward(self, images_pil, texts):
#         # images_pil: list[PIL.Image] 或单张 PIL
#         inputs = self.processor(text=texts, images=images_pil,
#                                 return_tensors="pt", padding=True).to(self.device)
#         f_img = self.clip.get_image_features(pixel_values=inputs["pixel_values"])
#         f_txt = self.clip.get_text_features(input_ids=inputs["input_ids"],
#                                             attention_mask=inputs["attention_mask"])
#         return F.normalize(f_img, dim=-1), F.normalize(f_txt, dim=-1)

import torch
import torch.nn.functional as F

class ChineseCLIPBackbone(torch.nn.Module):
    # 你原来的 __init__ 保持不变（clip + processor + device 等）
    def __init__(self, model_name_or_path: str, device: torch.device = None, dtype=None):
        super().__init__()

        self.clip = ChineseCLIPModel.from_pretrained(model_name_or_path)
        self.processor = ChineseCLIPProcessor.from_pretrained(model_name_or_path)

        # 统一 device：即使外部不传，也能推断
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if dtype is not None:
            self.clip = self.clip.to(dtype=dtype)
        self.clip = self.clip.to(self.device)
        self.clip.eval()

    @torch.no_grad()
    def _encode_text_robust(self, inputs: dict) -> torch.Tensor:
        """
        Robust text features: avoid ChineseCLIPModel.get_text_features() because pooled_output can be None.
        inputs must contain: input_ids, attention_mask (optional), token_type_ids(optional)
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        token_type_ids = inputs.get("token_type_ids", None)

        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden = text_outputs.last_hidden_state  # (B,L,H)

        B, L, H = last_hidden.shape
        device = last_hidden.device

        # Decide pooling position: EOS if available else last valid token
        eos_id = None
        # best-effort: tokenizer eos id
        if hasattr(self.processor, "tokenizer") and getattr(self.processor.tokenizer, "eos_token_id", None) is not None:
            eos_id = self.processor.tokenizer.eos_token_id
        # fallback: config
        if eos_id is None and hasattr(self.clip.config, "text_config") and getattr(self.clip.config.text_config, "eos_token_id", None) is not None:
            eos_id = self.clip.config.text_config.eos_token_id

        if attention_mask is not None:
            last_idx = attention_mask.long().sum(dim=1) - 1
            last_idx = torch.clamp(last_idx, min=0)
        else:
            last_idx = torch.full((B,), L - 1, dtype=torch.long, device=device)

        if eos_id is not None:
            eos_mask = (input_ids == eos_id)               # (B,L)
            has_eos = eos_mask.any(dim=1)                  # (B,)
            rev_pos = torch.flip(eos_mask, dims=[1]).float().argmax(dim=1)  # (B,)
            eos_idx = (L - 1) - rev_pos
            idx = torch.where(has_eos, eos_idx, last_idx)
        else:
            idx = last_idx

        pooled = last_hidden[torch.arange(B, device=device), idx]  # (B,H)

        # Optional final LN (some configs have it)
        if hasattr(self.clip.text_model, "final_layer_norm") and self.clip.text_model.final_layer_norm is not None:
            pooled = self.clip.text_model.final_layer_norm(pooled)

        text_features = self.clip.text_projection(pooled)  # (B,D)
        return F.normalize(text_features, dim=-1)

    @torch.no_grad()
    def forward(self, images_pil, texts):
        """
        images_pil: List[PIL.Image]
        texts     : List[str]
        returns (f_img, f_txt) each shape (B,D) on the backbone device
        """
        # IMPORTANT: run processor once for both modalities
        inputs = self.processor(
            text=texts,
            images=images_pil,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Image features: keep using official API (stable)
        f_img = self.clip.get_image_features(pixel_values=inputs["pixel_values"])
        f_img = F.normalize(f_img, dim=-1)

        # Text features: robust path
        f_txt = self._encode_text_robust(inputs)

        return f_img, f_txt

    


def build_gostone_dataloaders_from_dataset_modified_1(
    black_stone_folder: str,
    white_stone_folder: str,
    excel_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    use_white_anchor_only: bool = True,
    num_negatives: int = 1,
):
    """
    Create (torch_dataloader, collator) using Dataset_modified_1.GoStoneTripletDataset + TripletCollator.
    """
    # Local import to keep this file runnable even if Dataset_modified_1 is not present.
    from data.Dataset_modified_1 import GoStoneTripletDataset, TripletCollator

    dataset = GoStoneTripletDataset(
        black_stone_folder=black_stone_folder,
        white_stone_folder=white_stone_folder,
        excel_path=excel_path,
        num_negatives=num_negatives,
        seed=seed,
        use_white_anchor_only=use_white_anchor_only,
        root_dir=None,
    )
    collator = TripletCollator()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    return loader


# ============================================================
# Updated smoke test entrypoint using Dataset_modified_1
# ============================================================
if __name__ == "__main__":
    # NOTE: keep the original synthetic smoke test above untouched.
    # Uncomment and edit paths below to run with your Go-stone dataset.

    device = torch.device("cuda")
    D = 512  # embedding dim from your backbone (image/text must match)
    
    # 1) Build dataset DataLoader (raw PIL+text)
    raw_loader = build_gostone_dataloaders_from_dataset_modified_1(
        black_stone_folder="/root/autodl-tmp/test_photo/test-1",
        white_stone_folder="/root/autodl-tmp/test_photo/test-1",
        excel_path="/root/autodl-tmp/test_photo/test-1/test.xls",
        batch_size=64,
        num_workers=4,
        use_white_anchor_only=True,
        num_negatives=1,
    )
    
    # 2) Provide your backbone (replace _DummyBackbone with Chinese-CLIP backbone)
    backbone = ChineseCLIPBackbone(
    model_name_or_path="OFA-Sys/chinese-clip-vit-base-patch16",
    device=device
)

    backbone = backbone.to(device)
    backbone.eval()

    
    # 3) Wrap as embedding loaders for Stage2 / Stage3
    stage2_loader = TripletEmbeddingLoader(raw_loader, backbone=backbone, device=device)
    stage3_loader = PairEmbeddingLoader(raw_loader, backbone=backbone, device=device, pair_role="anchor")
    
    # 4) Train
    encoder = HypersphereEncoder(x_dim=D, t_dim=D, p=64)
    train_stage2(encoder, stage2_loader, Stage2Config(epochs=10), device=device, log_every=50)
    
    decoder = SphereDecoder(p=64)
    kappa_net = KappaNet(x_dim=D, t_dim=D, hidden=256)
    train_stage3(encoder, kappa_net, decoder, stage3_loader, Stage3Config(epochs=10, beta=10.0, use_swae=True), device=device, log_every=50)
    
    print("OK - Dataset_modified_1 wired into Stage2/Stage3")
    pass
