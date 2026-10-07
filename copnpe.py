"""Copula NPE: marginal stage, copula stage, the training pipeline of
Algorithm 1, and evaluation (LPP and the conditional mutual information
diagnostic).

    q(theta | x) = c_psi( F_1(theta_1 | x), ..., F_m(theta_m | x) | x ) prod_j f_j(theta_j | x)

Stage 1 (marginals)  per-parameter univariate Gaussian-mixture MDNs trained
                     by the per-dimension NPE loss; depending on the task the
                     MDN sees all of x, a window of x around the aligned
                     position (one network shared by all positions), or the
                     block of x belonging to a group / subject / site (one
                     network shared by all of them) plus a separate MDN for
                     the hyperparameters on pooled statistics.
Stage 2 (copula)     training pairs are mapped to normal scores
                     z = Phi^-1(F(theta | x)); a conditional Gaussian-mixture
                     model over z is trained by its NLL, each component's
                     precision parameterised by a lower-triangular Cholesky
                     factor whose free entries follow the posterior dependency
                     graph G (task.chol_mask()).  K = 1 is a conditional
                     Gaussian copula.  Sampling pushes copula samples through
                     the marginal quantile functions; the density is analytic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import logsumexp, ndtri
from scipy.stats import norm

LOG2PI = float(np.log(2 * np.pi))


def make_mlp(sizes, seed, out_scale=0.1):
    g = np.random.default_rng(seed)
    layers, n = [], len(sizes) - 1
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        lin = torch.nn.Linear(a, b, dtype=torch.float64)
        sc = np.sqrt(2.0 / a) * (out_scale if i == n - 1 else 1.0)
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(g.normal(0.0, sc, (b, a))))
            lin.bias.zero_()
        layers.append(lin)
        if i < n - 1:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def fit_early_stopping(net, loss, n_rows, seed, *, epochs=250, patience=35,
                       batch=256, lr=1e-3, val_frac=0.15):
    rng = np.random.default_rng(seed)
    n_val = max(1, int(val_frac * n_rows))
    perm = rng.permutation(n_rows)
    vi, ti = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        p = rng.permutation(len(ti))
        for b in range(0, len(ti), batch):
            l = loss(ti[p[b:b + batch]])
            opt.zero_grad()
            l.backward()
            opt.step()
        with torch.no_grad():
            vl = float(loss(vi))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return best


def _t(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


class MarginalMDN:
    def __init__(self, x_dim, d, K=2, hidden=(64, 64), seed=0):
        self.d, self.K = d, K
        self.net = make_mlp([x_dim, *hidden, 3 * K * d], seed)

    def _raw(self, xs):
        raw = self.net(xs).reshape(xs.shape[0], -1, 3, self.K)
        return raw[..., 0, :], raw[..., 1, :], raw[..., 2, :]

    def _nll(self, xs, ts):
        logits, means, logsc = self._raw(xs)
        logw = logits - torch.logsumexp(logits, -1, keepdim=True)
        z = (ts[..., None] - means) * torch.exp(-logsc)
        comp = -0.5 * z ** 2 - logsc - 0.5 * LOG2PI
        return -torch.logsumexp(logw + comp, -1).sum(-1).mean()

    def fit(self, x, theta, seed=0, **kw):
        self.x_mean, self.x_std = x.mean(0), x.std(0) + 1e-8
        self.theta_mean, self.theta_std = theta.mean(0), theta.std(0) + 1e-8
        xs = _t((x - self.x_mean) / self.x_std)
        ts = _t((theta - self.theta_mean) / self.theta_std)
        fit_early_stopping(self.net, lambda idx: self._nll(xs[idx], ts[idx]), len(x), seed, **kw)
        return self

    @torch.no_grad()
    def _params(self, x):
        lead = x.shape[:-1]
        xs = _t(((x - self.x_mean) / self.x_std).reshape(-1, x.shape[-1]))
        logits, means, logsc = (a.numpy() for a in self._raw(xs))
        logsc = np.clip(logsc, -7.0, 5.0)
        logw = logits - logsumexp(logits, axis=-1, keepdims=True)
        shp = (*lead, self.d, self.K)
        return logw.reshape(shp), means.reshape(shp), np.exp(logsc).reshape(shp)

    def cdf(self, x, theta):
        logw, means, scales = self._params(x)
        ts = (theta - self.theta_mean) / self.theta_std
        return np.sum(np.exp(logw) * norm.cdf((ts[..., None] - means) / scales), -1)

    def logpdf(self, x, theta):
        logw, means, scales = self._params(x)
        ts = (theta - self.theta_mean) / self.theta_std
        zc = (ts[..., None] - means) / scales
        pdf = np.sum(np.exp(logw) * np.exp(-0.5 * zc ** 2) / (scales * np.sqrt(2 * np.pi)), -1)
        return np.log(np.maximum(pdf, 1e-300)) - np.log(self.theta_std)

    def ppf(self, x, u, iters=60):
        logw, means, scales = self._params(x)
        w = np.exp(logw)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        lo = (means - 12 * scales).min(-1)
        hi = (means + 12 * scales).max(-1)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            F = np.sum(w * norm.cdf((mid[..., None] - means) / scales), -1)
            lt = F < u
            lo, hi = np.where(lt, mid, lo), np.where(lt, hi, mid)
        return 0.5 * (lo + hi) * self.theta_std + self.theta_mean


class LocalMarginalMDN(MarginalMDN):
    def __init__(self, d, K=2, hidden=(64, 64), window=4, seed=0):
        self.d, self.K, self.window = d, K, window
        self.net = make_mlp([2 * window + 1, *hidden, 3 * K], seed)

    def _windows(self, x):
        B, d, w = x.shape[0], x.shape[1], self.window
        xp = np.pad(x, ((0, 0), (w, w)), mode="edge")
        cols = np.stack([xp[:, t:t + 2 * w + 1] for t in range(d)], axis=1)
        return cols.reshape(B * d, 2 * w + 1)

    def fit(self, x, theta, seed=0, batch=256, **kw):
        B, d = x.shape
        xw = self._windows(x)
        self.x_mean, self.x_std = xw.mean(0), xw.std(0) + 1e-8
        self.theta_mean = np.repeat(theta.mean(), d)
        self.theta_std = np.repeat(theta.std() + 1e-8, d)
        xs = _t((xw - self.x_mean) / self.x_std)
        ts = _t(((theta - self.theta_mean) / self.theta_std).reshape(B * d, 1))
        fit_early_stopping(self.net, lambda idx: self._nll(xs[idx], ts[idx]), B * d, seed,
                           batch=batch * max(1, d // 2), **kw)
        return self

    @torch.no_grad()
    def _params(self, x):
        lead = x.shape[:-1]
        xw = self._windows(x.reshape(-1, self.d))
        xs = _t((xw - self.x_mean) / self.x_std)
        logits, means, logsc = (a.numpy() for a in self._raw(xs))
        logsc = np.clip(logsc, -7.0, 5.0)
        logw = logits - logsumexp(logits, axis=-1, keepdims=True)
        shp = (*lead, self.d, self.K)
        return logw.reshape(shp), means.reshape(shp), np.exp(logsc).reshape(shp)


def ar_group_features(task, x, j):
    return x.reshape(len(x), task.J, task.T)[:, j]


def ar_hyper_features(task, x):
    y = x.reshape(len(x), task.J, task.T)
    return np.concatenate([y.mean(-1), y.std(-1)], axis=1)


def rl_subject_features(task, x, j):
    return x.reshape(len(x), task.J, task.n_sum)[:, j]


def rl_hyper_features(task, x):
    xr = x.reshape(len(x), task.J, task.n_sum)
    return np.concatenate([xr.mean(1), xr.std(1)], axis=1)


def lattice_site_features(task, x):
    B, side = len(x), task.side
    xg = x.reshape(B, side, side)
    acc, cnt = np.zeros((B, side, side)), np.zeros((side, side))
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sr, tr = slice(max(dr, 0), side + min(dr, 0)), slice(max(-dr, 0), side + min(-dr, 0))
        sc, tc = slice(max(dc, 0), side + min(dc, 0)), slice(max(-dc, 0), side + min(-dc, 0))
        acc[:, tr, tc] += xg[:, sr, sc]
        cnt[tr, tc] += 1
    nbm = (acc / cnt).reshape(B, -1)
    return np.stack([x, nbm], axis=2)


class _BlockView:
    def __init__(self, task, K, hidden, seed=0, **fit_kw):
        self.t, self.K, self.hidden, self.seed, self.fit_kw = task, K, hidden, seed, fit_kw
        self.d = task.d

    def _apply(self, fn, x, arr):
        out = np.empty(arr.shape[:-1] + (self.d,))
        for F, cols, head in self._heads(x):
            out[..., cols] = getattr(head, fn)(F, arr[..., cols])
        return out

    def cdf(self, x, theta):
        return self._apply("cdf", x, theta)

    def logpdf(self, x, theta):
        return self._apply("logpdf", x, theta)

    def ppf(self, x, u):
        return self._apply("ppf", x, u)


class PooledGroupView(_BlockView):
    def fit(self, x, theta):
        t, J = self.t, self.t.J
        F = np.concatenate([ar_group_features(t, x, j) for j in range(J)], axis=0)
        Y = np.concatenate([theta[:, 3 * j:3 * j + 3] for j in range(J)], axis=0)
        self.shared = MarginalMDN(F.shape[1], 3, self.K, self.hidden, self.seed) \
            .fit(F, Y, self.seed, **self.fit_kw)
        H = ar_hyper_features(t, x)
        self.hyper = MarginalMDN(H.shape[1], 6, self.K, self.hidden, self.seed + 1) \
            .fit(H, theta[:, 3 * J:], self.seed + 1, **self.fit_kw)
        return self

    def _heads(self, x):
        J = self.t.J
        for j in range(J):
            yield ar_group_features(self.t, x, j), slice(3 * j, 3 * j + 3), self.shared
        yield ar_hyper_features(self.t, x), slice(3 * J, 3 * J + 6), self.hyper


class PooledSubjectView(_BlockView):
    def fit(self, x, theta):
        t, J, nl = self.t, self.t.J, self.t.n_loc
        F = np.concatenate([rl_subject_features(t, x, j) for j in range(J)], axis=0)
        Y = np.concatenate([theta[:, nl * j:nl * j + nl] for j in range(J)], axis=0)
        self.shared = MarginalMDN(F.shape[1], nl, self.K, self.hidden, self.seed) \
            .fit(F, Y, self.seed, **self.fit_kw)
        H = rl_hyper_features(t, x)
        self.hyper = MarginalMDN(H.shape[1], t.n_hyp, self.K, self.hidden, self.seed + 1) \
            .fit(H, theta[:, nl * J:], self.seed + 1, **self.fit_kw)
        return self

    def _heads(self, x):
        J, nl = self.t.J, self.t.n_loc
        for j in range(J):
            yield rl_subject_features(self.t, x, j), slice(nl * j, nl * j + nl), self.shared
        yield rl_hyper_features(self.t, x), slice(nl * J, self.d), self.hyper


class PooledSiteView(_BlockView):
    def fit(self, x, theta):
        F = lattice_site_features(self.t, x)
        Fp = np.concatenate([F[:, j, :] for j in range(self.d)], axis=0)
        Yp = np.concatenate([theta[:, j:j + 1] for j in range(self.d)], axis=0)
        self.shared = MarginalMDN(Fp.shape[1], 1, self.K, self.hidden, self.seed) \
            .fit(Fp, Yp, self.seed, **self.fit_kw)
        return self

    def _heads(self, x):
        F = lattice_site_features(self.t, x)
        for j in range(self.d):
            yield F[:, j, :], slice(j, j + 1), self.shared


def normal_scores(u):
    return ndtri(np.clip(u, 1e-10, 1 - 1e-10))


class GaussianMixtureCopula:
    def __init__(self, x_dim, d, K=1, hidden=(64, 64), chol_mask=None, seed=0):
        self.d, self.K = d, K
        rows, cols = np.tril_indices(d)
        keep = np.ones(len(rows), bool) if chol_mask is None \
            else np.asarray(chol_mask, bool)[rows, cols]
        keep |= rows == cols
        self.rows, self.cols = rows[keep], cols[keep]
        self.M = int(keep.sum())
        self.net = make_mlp([x_dim, *hidden, K + K * d + K * self.M], seed)

    def _split(self, raw):
        B, K, d = raw.shape[0], self.K, self.d
        return (raw[:, :K], raw[:, K:K + K * d].reshape(B, K, d),
                raw[:, K + K * d:].reshape(B, K, self.M))

    def _prec_chol(self, cvec):
        B, K, d = cvec.shape[0], self.K, self.d
        P = cvec.new_zeros(B, K, d, d)
        P[:, :, self.rows, self.cols] = cvec
        r = torch.arange(d)
        diag = torch.nn.functional.softplus(P[:, :, r, r]) + 1e-4
        return P - torch.diag_embed(P[:, :, r, r]) + torch.diag_embed(diag)

    def _log_density(self, xs, z):
        logits, means, cvec = self._split(self.net(xs))
        P = self._prec_chol(cvec)
        diff = z[:, None, :] - means
        q = (P.transpose(-1, -2) @ diff[..., None])[..., 0]
        comp = (torch.log(torch.diagonal(P, dim1=-2, dim2=-1)).sum(-1)
                - 0.5 * (q ** 2).sum(-1) - 0.5 * self.d * LOG2PI)
        logw = logits - torch.logsumexp(logits, -1, keepdim=True)
        return torch.logsumexp(logw + comp, -1).mean()

    def fit(self, x, z, seed=0, **kw):
        self.x_mean, self.x_std = x.mean(0), x.std(0) + 1e-8
        xs, zt = _t((x - self.x_mean) / self.x_std), _t(z)
        fit_early_stopping(self.net, lambda idx: -self._log_density(xs[idx], zt[idx]),
                           len(x), seed, **kw)
        return self

    @torch.no_grad()
    def _params(self, x):
        xs = _t((np.atleast_2d(x) - self.x_mean) / self.x_std)
        logits, means, cvec = self._split(self.net(xs))
        P = self._prec_chol(cvec).numpy()
        Sig = np.linalg.inv(P @ np.swapaxes(P, -1, -2))
        logits = logits.numpy()
        return (logits - logsumexp(logits, axis=-1, keepdims=True), means.numpy(),
                0.5 * (Sig + np.swapaxes(Sig, -1, -2)))

    def sample_u(self, rng, x, n):
        logw, means, Sig = self._params(x)
        B, K, d = means.shape
        w = np.exp(logw)
        chols = np.linalg.cholesky(Sig + 1e-10 * np.eye(d))
        ks = np.stack([rng.choice(K, size=n, p=w[b] / w[b].sum()) for b in range(B)])
        eps = rng.normal(size=(B, n, d))
        mu_k = np.take_along_axis(means, ks[..., None], axis=1)
        L_k = np.take_along_axis(chols, ks[..., None, None], axis=1)
        y = mu_k + np.einsum("bnij,bnj->bni", L_k, eps)
        sig = np.sqrt(np.diagonal(Sig, axis1=-2, axis2=-1))
        zc = (y[:, :, None, :] - means[:, None, :, :]) / sig[:, None, :, :]
        u = np.sum(w[:, None, :, None] * norm.cdf(zc), axis=2)
        return np.clip(u, 1e-10, 1 - 1e-10)

    def log_copula_density(self, x, u, iters=60):
        logw, means, Sig = self._params(x)
        B, K, d = means.shape
        w = np.exp(logw)
        sig = np.sqrt(np.diagonal(Sig, axis1=-2, axis2=-1))
        u = np.clip(u, 1e-10, 1 - 1e-10)
        lo, hi = (means - 12 * sig).min(1), (means + 12 * sig).max(1)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            F = np.sum(w[..., None] * norm.cdf((mid[:, None, :] - means) / sig), axis=1)
            lt = F < u
            lo, hi = np.where(lt, mid, lo), np.where(lt, hi, mid)
        z = 0.5 * (lo + hi)
        chol = np.linalg.cholesky(Sig + 1e-10 * np.eye(d))
        diff = z[:, None, :] - means
        sol = np.linalg.solve(chol, diff[..., None])[..., 0]
        logdet = 2 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(-1)
        comp = -0.5 * (sol ** 2).sum(-1) - 0.5 * logdet - 0.5 * d * LOG2PI
        log_joint = logsumexp(logw + comp, axis=-1)
        zc = (z[:, None, :] - means) / sig
        marg = np.sum(w[..., None] * np.exp(-0.5 * zc ** 2) / (sig * np.sqrt(2 * np.pi)), axis=1)
        return log_joint - np.log(np.maximum(marg, 1e-300)).sum(-1)


@dataclass(frozen=True)
class Config:
    view: str = "global"
    window: int = 0
    L: int = 2
    K: int = 1
    hidden_m: tuple = (64, 64)
    hidden_c: tuple = (64, 64)


CONFIGS = {
    "ssm":      Config("local", window=4, L=4),
    "ssm300":   Config("local", window=4, L=4),
    "minihub":  Config("local", window=0, L=2, K=4, hidden_c=(128, 128)),
    "dor1w62c": Config("global", L=2),
    "cnsmap":   Config("site", L=4),
    "mlarnc20": Config("group", L=6, K=4, hidden_c=(128, 128)),
    "hrl":      Config("subject", L=6, K=4, hidden_c=(128, 128)),
    "rtchain":  Config("local", window=12, L=2, hidden_c=(128, 128)),
}


def build_marginals(task, x, theta, cfg, seed=0, **fit_kw):
    if cfg.view == "global":
        return MarginalMDN(task.dx, task.d, cfg.L, cfg.hidden_m, seed).fit(x, theta, seed, **fit_kw)
    if cfg.view == "local":
        return LocalMarginalMDN(task.d, cfg.L, cfg.hidden_m, cfg.window, seed) \
            .fit(x, theta, seed, **fit_kw)
    view = {"group": PooledGroupView, "subject": PooledSubjectView,
            "site": PooledSiteView}[cfg.view]
    return view(task, cfg.L, cfg.hidden_m, seed, **fit_kw).fit(x, theta)


class CopulaNPE:
    def __init__(self, marginals, copula):
        self.marg, self.cop = marginals, copula

    def log_prob(self, x, theta):
        u = self.marg.cdf(x, theta)
        return self.cop.log_copula_density(x, u) + self.marg.logpdf(x, theta).sum(-1)

    def sample(self, rng, x_obs, n):
        u = self.cop.sample_u(rng, x_obs[None], n)[0]
        return self.marg.ppf(np.broadcast_to(x_obs, (n, len(x_obs))), u)


def fit_copula_npe(task, x, theta, cfg, seed=0, graph="G", **fit_kw):
    marg = build_marginals(task, x, theta, cfg, seed, **fit_kw)
    z = normal_scores(marg.cdf(x, theta))
    mask = {"G": task.chol_mask(), "dense": None}[graph]
    cop = GaussianMixtureCopula(task.dx, task.d, cfg.K, cfg.hidden_c, mask, seed + 9)
    cop.fit(x, z, seed + 9, **fit_kw)
    return CopulaNPE(marg, cop)


def lpp(model, X, TH):
    return float(np.mean(model.log_prob(X, TH)))


def _block_logpdf(z, logw, mu, Sig, idx):
    idx = np.asarray(sorted(idx))
    if len(idx) == 0:
        return np.zeros(z.shape[:2])
    zA, mA = z[:, :, idx], mu[:, :, idx]
    SA = Sig[:, :, idx][:, :, :, idx]
    L = np.linalg.cholesky(SA + 1e-10 * np.eye(len(idx)))
    diff = zA[:, None, :, :] - mA[:, :, None, :]
    sol = np.linalg.solve(L[:, :, None], diff[..., None])[..., 0]
    logdet = 2 * np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1)
    comp = -0.5 * ((sol ** 2).sum(-1) + logdet[:, :, None] + len(idx) * LOG2PI)
    return logsumexp(logw[:, :, None] + comp, axis=1)


def conditional_mi(copula, X, neighbours, n_z=200, seed=0, chunk=25):
    d = copula.d
    rng = np.random.default_rng(seed)
    nodes = [j for j in range(d) if len(neighbours[j]) < d - 1]
    per_x = np.zeros((len(X), len(nodes)))
    for a in range(0, len(X), chunk):
        logw, mu, Sig = copula._params(X[a:a + chunk])
        b, K, _ = mu.shape
        w = np.exp(logw)
        L = np.linalg.cholesky(Sig + 1e-10 * np.eye(d))
        z = np.empty((b, n_z, d))
        for i in range(b):
            k = rng.choice(K, size=n_z, p=w[i] / w[i].sum())
            z[i] = mu[i, k] + np.einsum("nij,nj->ni", L[i, k], rng.normal(size=(n_z, d)))
        lall = _block_logpdf(z, logw, mu, Sig, range(d))
        for c, j in enumerate(nodes):
            ne = set(neighbours[j]) - {j}
            term = (lall - _block_logpdf(z, logw, mu, Sig, [i for i in range(d) if i != j])
                    - _block_logpdf(z, logw, mu, Sig, ne | {j})
                    + _block_logpdf(z, logw, mu, Sig, ne))
            per_x[a:a + chunk, c] = term.mean(1)
    return nodes, per_x.mean(0)


def neighbour_sets(task):
    A = task.theta_adj()
    return [set(np.flatnonzero(A[j]).tolist()) - {j} for j in range(task.d)]
