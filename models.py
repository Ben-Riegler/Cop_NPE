"""Baseline models of the Copula NPE paper.

MDN-NPE, NSF-NPE and FMPE are taken from the `sbi` package at library
defaults (fit_sbi, sbi_log_prob).  The Simformer (Gloeckler et al., 2024) is
re-implemented in PyTorch: one token per scalar variable with identity, value,
condition-flag and Gaussian-Fourier time embeddings, a transformer whose
attention mask carries the task's posterior dependency graph G between
parameter tokens and the task's theta-x locality, a VE-SDE with denoising
score matching, posterior sampling by the reverse SDE with the observed
tokens clamped, and log densities by the probability-flow ODE
(fit_simformer, simformer_log_prob).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def fit_sbi(method, x, theta, seed=0, prior=None):
    from sbi.inference import FMPE, NPE
    torch.manual_seed(seed)
    trainer = FMPE(prior=prior) if method == "fmpe" \
        else NPE(prior=prior, density_estimator=method)
    trainer.append_simulations(torch.tensor(theta, dtype=torch.float32),
                               torch.tensor(np.atleast_2d(x), dtype=torch.float32)).train()
    return trainer.build_posterior()


def sbi_log_prob(posterior, theta, x, chunk=250):
    th_t = torch.tensor(theta, dtype=torch.float32)
    x_t = torch.tensor(np.atleast_2d(x), dtype=torch.float32)
    out = []
    for a in range(0, len(th_t), chunk):
        tb, xb = th_t[a:a + chunk], x_t[a:a + chunk]
        lp = None
        for shape in (0, 1):
            try:
                cand = posterior.log_prob_batched(tb.unsqueeze(shape), xb,
                                                  norm_posterior=False).reshape(-1)
                if cand.numel() == len(tb):
                    lp = cand
                    break
            except Exception:
                pass
        if lp is None:
            try:
                lp = torch.stack([posterior.log_prob(tb[i:i + 1], x=xb[i],
                                                     norm_posterior=False)[0]
                                  for i in range(len(tb))])
            except TypeError:
                lp = torch.stack([posterior.log_prob(tb[i:i + 1], x=xb[i])[0]
                                  for i in range(len(tb))])
        out.append(lp.detach().cpu().numpy())
    return np.concatenate(out)


DEV = torch.device("cuda") if torch.cuda.is_available() \
    else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


class _FourierTime(nn.Module):
    def __init__(self, dim=128, out=50, scale=16.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(dim // 2, generator=g) * scale)
        self.mlp = nn.Sequential(nn.Linear(dim, out), nn.GELU(), nn.Linear(out, out))

    def forward(self, t):
        ang = t[:, None] * self.B[None, :] * 2 * math.pi
        return self.mlp(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))


class _Block(nn.Module):
    def __init__(self, dm=50, heads=5, ff=3):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dm), nn.LayerNorm(dm)
        self.qkv, self.proj = nn.Linear(dm, 3 * dm), nn.Linear(dm, dm)
        self.ff = nn.Sequential(nn.Linear(dm, ff * dm), nn.GELU(), nn.Linear(ff * dm, dm))

    def forward(self, h, attn_mask):
        b, n, dm = h.shape
        q, k, v = self.qkv(self.ln1(h)).chunk(3, dim=-1)
        hd = dm // self.heads
        q, k, v = (a.view(b, n, self.heads, hd).transpose(1, 2) for a in (q, k, v))
        a = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=~attn_mask if attn_mask is not None else None)
        h = h + self.proj(a.transpose(1, 2).reshape(b, n, dm))
        return h + self.ff(self.ln2(h))


class SimformerNet(nn.Module):
    def __init__(self, n_tok, dm=50, layers=6, heads=5, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.id_emb, self.val, self.cond = nn.Embedding(n_tok, dm), nn.Linear(1, dm), nn.Embedding(2, dm)
        self.time = _FourierTime(out=dm, seed=seed)
        self.blocks = nn.ModuleList([_Block(dm, heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 1))

    def forward(self, v, mc, t, attn_mask):
        ids = torch.arange(v.shape[1], device=v.device)
        h = self.id_emb(ids)[None] + self.val(v[..., None]) + self.cond(mc.long()) \
            + self.time(t)[:, None, :]
        for blk in self.blocks:
            h = blk(h, attn_mask)
        return self.out(h)[..., 0]


class Simformer:
    SIG_MIN, SIG_MAX = 1e-4, 15.0
    T_MIN, T_MAX = 1e-5, 1.0

    def __init__(self, d_theta, d_x, theta_adj=None, x_adj=None, seed=0, dm=50,
                 layers=6, heads=5):
        self.d, self.dx = d_theta, d_x
        n = d_theta + d_x
        self.net = SimformerNet(n, dm, layers, heads, seed=seed).to(DEV)
        allow = np.zeros((n, n), dtype=bool)
        tt = np.ones((d_theta, d_theta), bool) if theta_adj is None \
            else np.asarray(theta_adj, bool) | np.eye(d_theta, dtype=bool)
        tx = np.ones((d_theta, d_x), bool) if x_adj is None else np.asarray(x_adj, bool)
        allow[:d_theta, :d_theta] = tt
        allow[:d_theta, d_theta:] = tx
        allow[d_theta:, :d_theta] = tx.T
        allow[d_theta:, d_theta:] = True
        np.fill_diagonal(allow, True)
        self.attn_mask = torch.from_numpy(~allow).to(DEV)
        self.rng = np.random.default_rng(seed)

    def _sigma(self, t):
        return self.SIG_MIN * (self.SIG_MAX / self.SIG_MIN) ** t

    def _sample_mc(self, b):
        n, d = self.d + self.dx, self.d
        mc = np.zeros((b, n), dtype=np.float32)
        kind = self.rng.integers(0, 5, size=b)
        mc[kind == 1, d:] = 1.0
        mc[kind == 2, :d] = 1.0
        for p, kk in ((0.3, 3), (0.7, 4)):
            rows = np.where(kind == kk)[0]
            if len(rows):
                mc[rows] = self.rng.uniform(size=(len(rows), n)) < p
        return mc

    def fit(self, eta, x, max_steps=10000, batch=512, lr=3e-4, patience=20, seed=0,
            val_frac=0.1):
        torch.manual_seed(seed)
        N = len(eta)
        v_all = np.concatenate([eta, x], axis=1).astype(np.float32)
        n_val = max(8, int(val_frac * N))
        idx = np.random.default_rng(seed + 7).permutation(N)
        tr, va = idx[n_val:], idx[:n_val]
        vt, vv = torch.from_numpy(v_all[tr]).to(DEV), torch.from_numpy(v_all[va]).to(DEV)
        gval, REP = np.random.default_rng(seed + 13), 8
        tv = torch.from_numpy((self.T_MIN + (self.T_MAX - self.T_MIN)
                               * gval.uniform(size=len(va) * REP)).astype(np.float32)).to(DEV)
        ev = torch.from_numpy(gval.normal(size=(len(va) * REP, v_all.shape[1]))
                              .astype(np.float32)).to(DEV)
        mv = torch.from_numpy(np.tile(self._sample_mc(len(va)), (REP, 1))).to(DEV)
        vvr = vv.repeat(REP, 1)
        sigv = torch.from_numpy(self._sigma(tv.cpu().numpy()).astype(np.float32)).to(DEV)[:, None]
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        best, best_state, bad = np.inf, None, 0
        bs, n_tok, d = min(batch, len(tr)), self.d + self.dx, self.d
        lnratio = math.log(self.SIG_MAX / self.SIG_MIN)
        for step in range(max_steps):
            rows = torch.randint(0, len(tr), (bs,), device=DEV)
            v0 = vt[rows]
            t = self.T_MIN + (self.T_MAX - self.T_MIN) * torch.rand(bs, device=DEV)
            kind = torch.randint(0, 5, (bs, 1), device=DEV)
            u = torch.rand(bs, n_tok, device=DEV)
            mc = torch.zeros(bs, n_tok, device=DEV)
            post = torch.cat([torch.zeros(bs, d, device=DEV), torch.ones(bs, n_tok - d, device=DEV)], 1)
            mc = torch.where(kind == 1, post, mc)
            mc = torch.where(kind == 2, 1 - post, mc)
            mc = torch.where(kind == 3, (u < 0.3).float(), mc)
            mc = torch.where(kind == 4, (u < 0.7).float(), mc)
            sig = (self.SIG_MIN * torch.exp(t * lnratio))[:, None]
            eps = torch.randn_like(v0)
            vin = torch.where(mc > 0, v0, v0 + sig * eps)
            s = self.net(vin, mc, t, self.attn_mask)
            loss = (((sig * s + eps) ** 2) * (1 - mc)).sum() / torch.clamp((1 - mc).sum(), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == max_steps - 1:
                with torch.no_grad():
                    vinv = torch.where(mv > 0, vvr, vvr + sigv * ev)
                    sv = self.net(vinv, mv, tv, self.attn_mask)
                    lv = float(((((sigv * sv + ev) ** 2) * (1 - mv)).sum()
                                / torch.clamp((1 - mv).sum(), min=1.0)))
                if lv < best - 1e-4:
                    best, bad = lv, 0
                    best_state = {k: p.detach().clone() for k, p in self.net.state_dict().items()}
                else:
                    bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    @torch.no_grad()
    def sample_posterior(self, x_std, n, steps=500, seed=0):
        torch.manual_seed(seed)
        d, dx = self.d, self.dx
        mc = torch.zeros(n, d + dx, device=DEV)
        mc[:, d:] = 1.0
        xin = torch.from_numpy(np.broadcast_to(x_std, (n, dx)).astype(np.float32).copy()).to(DEV)
        th = torch.randn(n, d, device=DEV) * self._sigma(self.T_MAX)
        ts = np.linspace(self.T_MAX, self.T_MIN, steps + 1)
        lnr = math.log(self.SIG_MAX / self.SIG_MIN)
        for i in range(steps):
            t0, dt = ts[i], ts[i] - ts[i + 1]
            g2 = 2.0 * self._sigma(t0) ** 2 * lnr
            s = self.net(torch.cat([th, xin], 1), mc, torch.full((n,), float(t0), device=DEV),
                         self.attn_mask)[:, :d]
            th = th + g2 * s * dt + math.sqrt(g2 * dt) * torch.randn_like(th)
        return th.cpu().numpy()

    def _score_div(self, th, xin, mc, tt, use_exact, hutch_k):
        n, d = th.shape
        th = th.detach().requires_grad_(True)
        s = self.net(torch.cat([th, xin], 1), mc, tt, self.attn_mask)[:, :d]
        divv = torch.zeros(n, device=DEV)
        if use_exact:
            for j in range(d):
                divv = divv + torch.autograd.grad(s[:, j].sum(), th, retain_graph=True)[0][:, j]
        else:
            for _ in range(hutch_k):
                r = torch.randint(0, 2, (n, d), device=DEV, dtype=torch.float32) * 2 - 1
                g = torch.autograd.grad((s * r).sum(), th, retain_graph=True)[0]
                divv = divv + (g * r).sum(1)
            divv = divv / hutch_k
        return s.detach(), divv.detach()

    def log_prob(self, eta, x_std, steps=32, div="auto", hutch_k=4, seed=0, batch=1000):
        use_exact = div == "exact" or (div == "auto" and self.d <= 8)
        n_tok = self.d + self.dx
        batch = min(batch, max(32, int(4e8 / (5 * n_tok * n_tok * 4))))
        out = np.empty(len(eta))
        for a in range(0, len(eta), batch):
            sl = slice(a, min(a + batch, len(eta)))
            out[sl] = self._log_prob_batch(eta[sl], x_std[sl], steps, use_exact, hutch_k, seed)
        return out

    def _log_prob_batch(self, eta, x_std, steps, use_exact, hutch_k, seed):
        d, dx, n = self.d, self.dx, len(eta)
        torch.manual_seed(seed)
        mc = torch.zeros(n, d + dx, device=DEV)
        mc[:, d:] = 1.0
        xin = torch.from_numpy(x_std.astype(np.float32)).to(DEV)
        th = torch.from_numpy(eta.astype(np.float32)).to(DEV)
        lnr = math.log(self.SIG_MAX / self.SIG_MIN)
        ts = np.linspace(self.T_MIN, self.T_MAX, steps + 1)
        logp = torch.zeros(n, device=DEV)

        def f(th_, t_):
            g2 = 2.0 * self._sigma(t_) ** 2 * lnr
            s, dv = self._score_div(th_, xin, mc, torch.full((n,), float(t_), device=DEV),
                                    use_exact, hutch_k)
            return -0.5 * g2 * s, -0.5 * g2 * dv

        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]
            h = t1 - t0
            k1, d1 = f(th, t0)
            k2, d2 = f(th + 0.5 * h * k1, t0 + 0.5 * h)
            k3, d3 = f(th + 0.5 * h * k2, t0 + 0.5 * h)
            k4, d4 = f(th + h * k3, t1)
            th = th + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            logp = logp + h / 6.0 * (d1 + 2 * d2 + 2 * d3 + d4)
        sT = self._sigma(self.T_MAX)
        base = -0.5 * (th ** 2).sum(1) / sT ** 2 - d * math.log(sT) - 0.5 * d * math.log(2 * math.pi)
        return (base + logp).cpu().numpy()


def fit_simformer(task, x, theta, seed=0, **fit_kw):
    e_mu, e_sd = theta.mean(0), theta.std(0) + 1e-8
    x_mu, x_sd = x.mean(0), x.std(0) + 1e-8
    model = Simformer(task.d, task.dx, theta_adj=task.theta_adj(), x_adj=task.x_adj(), seed=seed)
    model.fit((theta - e_mu) / e_sd, (x - x_mu) / x_sd, batch=min(512, len(x)),
              seed=seed, **fit_kw)
    return model, (e_mu, e_sd, x_mu, x_sd)


def simformer_log_prob(model, moments, theta, x):
    e_mu, e_sd, x_mu, x_sd = moments
    lp_std = model.log_prob((theta - e_mu) / e_sd, (x - x_mu) / x_sd)
    return lp_std - np.log(e_sd).sum()
