"""Simulation tasks of the Copula NPE paper and the shared data protocol.

Interface of every task:
    d, dx                 parameter and observation dimension
    name                  task codename
    sample_joint(rng, n)  n pairs (x, theta)
    log_prior(theta)      exact log prior density (None if not available)
    log_posterior(x, th)  exact log posterior density, where closed form
    chol_mask()           lower-triangular boolean (d, d): free entries of the
                          copula's precision-Cholesky factors (the graph G)
    theta_adj()           boolean (d, d): G as the Simformer's theta-theta mask
    x_adj()               boolean (d, dx): the Simformer's theta-x mask
                          (None = full attention)
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, ndtri
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

BUDGETS = (100, 316, 1000, 3162, 10000)
SEEDS = (0, 1, 2, 3, 4)
TRAIN_RNG = 1000
TEST_RNG = 4242
N_TEST = 1000


def training_set(task, N, seed):
    return task.sample_joint(np.random.default_rng(TRAIN_RNG + seed), N)


def test_set(task):
    return task.sample_joint(np.random.default_rng(TEST_RNG), task.n_test)


def anchors(task):
    X, TH = test_set(task)
    out = {"lpp_prior": None, "lpp_true": None}
    if task.log_prior is not None:
        out["lpp_prior"] = float(np.mean(task.log_prior(TH)))
    if hasattr(task, "log_posterior"):
        out["lpp_true"] = float(np.mean(task.log_posterior(X, TH)))
    return out


def band_mask(d, w):
    i, j = np.tril_indices(d)
    m = np.zeros((d, d), dtype=bool)
    m[i, j] = (i - j) <= w
    return m


def chain_adj(d):
    a = np.zeros((d, d), dtype=bool)
    idx = np.arange(d - 1)
    a[idx, idx + 1] = a[idx + 1, idx] = True
    return a


def grid_adj(side):
    d = side * side
    a = np.zeros((d, d), dtype=bool)
    for i in range(d):
        r, c = divmod(i, side)
        if c + 1 < side:
            a[i, i + 1] = a[i + 1, i] = True
        if r + 1 < side:
            a[i, i + side] = a[i + side, i] = True
    return a


def hub_adj(d, n_hub=1):
    a = np.zeros((d, d), dtype=bool)
    for h in range(d - n_hub, d):
        a[h, :h] = a[:h, h] = True
    return a


class StateSpaceChain:
    name = "ssm"

    def __init__(self, d=20, a=0.95, q=0.3, r=0.5, n_test=N_TEST):
        self.d = self.dx = d
        self.a, self.q, self.r = a, q, r
        self.n_test = n_test
        J = np.zeros((d, d))
        J[0, 0] = (1 - a ** 2) / q ** 2
        for t in range(1, d):
            J[t, t] += 1 / q ** 2
            J[t - 1, t - 1] += a ** 2 / q ** 2
            J[t, t - 1] = J[t - 1, t] = -a / q ** 2
        self.J_post = J + np.eye(d) / r ** 2
        _, self._logdet_post = np.linalg.slogdet(self.J_post)

    def sample_joint(self, rng, n):
        d, a, q, r = self.d, self.a, self.q, self.r
        th = np.empty((n, d))
        th[:, 0] = rng.normal(0, q / np.sqrt(1 - a ** 2), size=n)
        for t in range(1, d):
            th[:, t] = a * th[:, t - 1] + q * rng.normal(size=n)
        return th + r * rng.normal(size=(n, d)), th

    def log_prior(self, th):
        th = np.atleast_2d(th)
        lp = norm.logpdf(th[:, 0], 0, self.q / np.sqrt(1 - self.a ** 2))
        return lp + norm.logpdf(th[:, 1:], self.a * th[:, :-1], self.q).sum(-1)

    def posterior_mean(self, x):
        return np.linalg.solve(self.J_post, (np.atleast_2d(x) / self.r ** 2).T).T

    def log_posterior(self, x, th):
        diff = np.atleast_2d(th) - self.posterior_mean(x)
        quad = np.einsum("ni,ij,nj->n", diff, self.J_post, diff)
        return 0.5 * self._logdet_post - 0.5 * self.d * np.log(2 * np.pi) - 0.5 * quad

    def chol_mask(self):
        return band_mask(self.d, 1)

    def theta_adj(self):
        return chain_adj(self.d)

    def x_adj(self):
        return band_mask(self.d, 4) | band_mask(self.d, 4).T


class RandomEffectsHub:
    name = "minihub"

    def __init__(self, J=9, s_mu=3.0, tau=3.0, sig=1.0):
        self.J, self.d, self.dx = J, J + 1, J + 1
        self.s_mu, self.tau, self.sig = s_mu, tau, sig
        self.n_test = N_TEST
        Q = np.zeros((self.d, self.d))
        Q[:J, :J] = np.eye(J) * (1 / tau ** 2 + 1 / sig ** 2)
        Q[:J, J] = Q[J, :J] = -1 / tau ** 2
        Q[J, J] = J / tau ** 2 + 1 / s_mu ** 2
        self.Q_post = Q
        self.Sig_post = np.linalg.inv(Q)
        _, self._logdet_Sig = np.linalg.slogdet(self.Sig_post)

    def sample_joint(self, rng, n):
        mu = rng.normal(0, self.s_mu, size=n)
        th = mu[:, None] + self.tau * rng.normal(size=(n, self.J))
        y = th + self.sig * rng.normal(size=(n, self.J))
        x = np.concatenate([y, y.mean(axis=1, keepdims=True)], axis=1)
        return x, np.concatenate([th, mu[:, None]], axis=1)

    def log_prior(self, th):
        th = np.atleast_2d(th)
        lp = norm.logpdf(th[:, self.J], 0, self.s_mu)
        return lp + norm.logpdf(th[:, :self.J], th[:, [self.J]], self.tau).sum(-1)

    def posterior_mean(self, x):
        y = np.atleast_2d(x)[:, :self.J]
        b = np.concatenate([y / self.sig ** 2, np.zeros((len(y), 1))], axis=1)
        return b @ self.Sig_post.T

    def log_posterior(self, x, th):
        diff = np.atleast_2d(th) - self.posterior_mean(x)
        quad = np.einsum("ni,ij,nj->n", diff, self.Q_post, diff)
        return -0.5 * quad - 0.5 * self._logdet_Sig - 0.5 * self.d * np.log(2 * np.pi)

    def chol_mask(self):
        m = np.eye(self.d, dtype=bool)
        m[-1, :] = True
        return m

    def theta_adj(self):
        return hub_adj(self.d, 1)

    def x_adj(self):
        return np.eye(self.d, dtype=bool)


class BiModalGaussian:
    name = "dor1w62c"
    log_prior = None

    def __init__(self, m=18, seed=2127, w1s=0.5, msc=0.35, wsc=0.8, ssc=0.25,
                 esc=0.8):
        self.d = self.dx = m
        self.n_test = N_TEST
        g = np.random.default_rng(seed)
        H = 32
        self.W1 = g.normal(0, w1s, (m, H))
        self.a1 = g.normal(0, msc, (H, m))
        self.a2 = g.normal(0, msc, (H, m))
        self.cw = g.normal(0, wsc, (H, m))
        self.dw = g.normal(0, ssc, (H, m))
        self.ew = g.normal(0, esc, (H, m - 1))

    def _pars(self, x):
        h = np.tanh(np.atleast_2d(x) @ self.W1)
        mu1 = -1.4 + h @ self.a1
        mu2 = 1.4 + h @ self.a2
        w1 = expit(h @ self.cw)
        s = np.clip(0.45 * np.exp(h @ self.dw), 0.25, 0.9)
        rho = 0.65 * np.tanh(h @ self.ew)
        return mu1, mu2, w1, s, rho

    @staticmethod
    def _F(th, mu1, mu2, w1, s):
        return w1 * norm.cdf((th - mu1) / s) + (1 - w1) * norm.cdf((th - mu2) / s)

    def _F_inv(self, u, mu1, mu2, w1, s):
        lo = np.minimum(mu1, mu2) - 12 * s
        hi = np.maximum(mu1, mu2) + 12 * s
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = self._F(mid, mu1, mu2, w1, s) < u
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def sample_joint(self, rng, n):
        x = rng.standard_normal((n, self.dx))
        mu1, mu2, w1, s, rho = self._pars(x)
        z = np.empty((n, self.d))
        z[:, 0] = rng.standard_normal(n)
        for j in range(1, self.d):
            r = rho[:, j - 1]
            z[:, j] = r * z[:, j - 1] + np.sqrt(1 - r ** 2) * rng.standard_normal(n)
        u = np.clip(norm.cdf(z), 1e-12, 1 - 1e-12)
        return x, self._F_inv(u, mu1, mu2, w1, s)

    def log_posterior(self, x, th):
        mu1, mu2, w1, s, rho = self._pars(x)
        th = np.atleast_2d(th)
        c1 = norm.logpdf(th, mu1, s) + np.log(np.maximum(w1, 1e-300))
        c2 = norm.logpdf(th, mu2, s) + np.log(np.maximum(1 - w1, 1e-300))
        lf = np.logaddexp(c1, c2)
        z = ndtri(np.clip(self._F(th, mu1, mu2, w1, s), 1e-12, 1 - 1e-12))
        lc = np.zeros(len(th))
        for j in range(1, self.d):
            r = rho[:, j - 1]
            lc += (norm.logpdf(z[:, j], r * z[:, j - 1], np.sqrt(1 - r ** 2))
                   - norm.logpdf(z[:, j]))
        return lf.sum(-1) + lc

    def chol_mask(self):
        return band_mask(self.d, 1)

    def theta_adj(self):
        return chain_adj(self.d)

    def x_adj(self):
        return None


class CensoredLattice:
    name = "cnsmap"

    def __init__(self, L=6, tau=1.0, rho=0.95, s=0.5, q=0.40):
        self.L, self.side = L, L
        self.d = self.dx = L * L
        self.n_test = N_TEST
        W = grid_adj(L).astype(float)
        self.Q = tau * (np.diag(W.sum(1)) - rho * W)
        cov = np.linalg.inv(self.Q)
        self._prior_chol = np.linalg.cholesky(cov)
        _, self._logdetQ = np.linalg.slogdet(self.Q)
        self.s = s
        self.c = float(ndtri(q) * np.sqrt(np.mean(np.diag(cov)) + s ** 2))

    def sample_joint(self, rng, n):
        th = rng.normal(size=(n, self.d)) @ self._prior_chol.T
        lat = th + self.s * rng.normal(size=(n, self.d))
        return np.maximum(lat, self.c), th

    def log_prior(self, th):
        th = np.atleast_2d(th)
        quad = np.einsum("ni,ij,nj->n", th, self.Q, th)
        return 0.5 * self._logdetQ - 0.5 * self.d * np.log(2 * np.pi) - 0.5 * quad

    def chol_mask(self):
        return band_mask(self.d, self.L)

    def theta_adj(self):
        return grid_adj(self.L)

    def x_adj(self):
        return grid_adj(self.L) | np.eye(self.d, dtype=bool)


class MultilevelAR1:
    name = "mlarnc20"

    def __init__(self, J=20, T=8):
        self.J, self.T = J, T
        self.d, self.dx = 3 * J + 6, J * T
        self.n_test = N_TEST
        self._hyp_mu = np.array([0.0, -1.2, 1.0, -0.9, -0.7, -1.2])
        self._hyp_sd = np.array([0.5, 0.4, 0.5, 0.4, 0.3, 0.4])

    def sample_joint(self, rng, n):
        J, T = self.J, self.T
        hyp = self._hyp_mu + self._hyp_sd * rng.standard_normal((n, 6))
        z = rng.standard_normal((n, J, 3))
        loc = hyp[:, None, [0, 2, 4]] + np.exp(hyp[:, None, [1, 3, 5]]) * z
        alpha, beta, sig = loc[:, :, 0], 0.95 * np.tanh(loc[:, :, 1]), np.exp(loc[:, :, 2])
        y = np.empty((n, J, T))
        y[:, :, 0] = alpha / (1 - beta) + sig / np.sqrt(1 - beta ** 2) \
            * rng.standard_normal((n, J))
        for t in range(1, T):
            y[:, :, t] = alpha + beta * y[:, :, t - 1] + sig * rng.standard_normal((n, J))
        th = np.concatenate([z.reshape(n, 3 * J), hyp], axis=1)
        return y.reshape(n, J * T), th

    def log_prior(self, th):
        th = np.atleast_2d(th)
        H = 3 * self.J
        return (norm.logpdf(th[:, :H]).sum(-1)
                + norm.logpdf(th[:, H:], self._hyp_mu, self._hyp_sd).sum(-1))

    def chol_mask(self):
        J, H = self.J, 3 * self.J
        m = np.zeros((self.d, self.d), dtype=bool)
        for j in range(J):
            m[3 * j:3 * j + 3, 3 * j:3 * j + 3] = True
        for i, (mcol, ltcol) in enumerate(((0, 1), (2, 3), (4, 5))):
            for j in range(J):
                m[H + mcol, 3 * j + i] = m[H + ltcol, 3 * j + i] = True
        m[H:, H:] = True
        return np.tril(m)

    def theta_adj(self):
        a = self.chol_mask()
        a = a | a.T
        np.fill_diagonal(a, False)
        return a

    def x_adj(self):
        J, T = self.J, self.T
        a = np.zeros((self.d, self.dx), dtype=bool)
        for j in range(J):
            a[3 * j:3 * j + 3, j * T:(j + 1) * T] = True
        a[3 * J:, :] = True
        return a


class HierarchicalRL:
    name = "hrl"

    def __init__(self, J=12, B=4, Tr=20, p=0.8):
        self.J, self.B, self.Tr, self.p = J, B, Tr, p
        self.n_loc, self.n_hyp = 2, 4
        self.n_sum = B + 4
        self.d, self.dx = 2 * J + 4, J * self.n_sum
        self.n_test = N_TEST

    def sample_joint(self, rng, n):
        J, B, Tr = self.J, self.B, self.Tr
        mu_a, lt_a = rng.normal(0.0, 1.0, n), rng.normal(-1.0, 0.5, n)
        mu_b, lt_b = rng.normal(0.5, 0.7, n), rng.normal(-1.0, 0.5, n)
        za, zb = rng.standard_normal((n, J)), rng.standard_normal((n, J))
        alpha = expit(mu_a[:, None] + np.exp(lt_a)[:, None] * za)
        beta = np.exp(np.clip(mu_b[:, None] + np.exp(lt_b)[:, None] * zb, -4.0, 4.0))
        Q = np.zeros((n, J, 2))
        ii = np.arange(n)[:, None], np.arange(J)[None, :]
        corr = np.zeros((n, J, B))
        ws_n, ws_d = np.zeros((n, J)), np.zeros((n, J))
        ls_n, ls_d = np.zeros((n, J)), np.zeros((n, J))
        sw = np.zeros((n, J))
        half = np.zeros((n, J, 2))
        prev_c = prev_r = None
        tt = 0
        for b in range(B):
            good = b % 2
            for _ in range(Tr):
                p1 = expit(beta * (Q[:, :, 1] - Q[:, :, 0]))
                c = (rng.random((n, J)) < p1).astype(int)
                hit = (c == good)
                r = (rng.random((n, J)) < np.where(hit, self.p, 1 - self.p)).astype(float)
                corr[:, :, b] += hit
                half[:, :, 1 if tt >= (B * Tr) // 2 else 0] += hit
                if prev_c is not None:
                    stay = (c == prev_c)
                    ws_d += prev_r
                    ws_n += prev_r * stay
                    ls_d += 1 - prev_r
                    ls_n += (1 - prev_r) * stay
                    sw += 1 - stay
                idx = (ii[0], ii[1], c)
                Q[idx] += alpha * (r - Q[idx])
                prev_c, prev_r = c, r
                tt += 1
        corr /= Tr
        ws = ws_n / np.maximum(ws_d, 1.0)
        ls = ls_n / np.maximum(ls_d, 1.0)
        sw /= max(B * Tr - 1, 1)
        hdiff = (half[:, :, 1] - half[:, :, 0]) / (B * Tr / 2.0)
        x = np.concatenate([corr, ws[..., None], ls[..., None], sw[..., None],
                            hdiff[..., None]], axis=2).reshape(n, self.dx)
        loc = np.stack([za, zb], axis=2).reshape(n, 2 * J)
        th = np.concatenate([loc, mu_a[:, None], lt_a[:, None], mu_b[:, None],
                             lt_b[:, None]], axis=1)
        return x, th

    def log_prior(self, th):
        th = np.atleast_2d(th)
        H = 2 * self.J
        lp = norm.logpdf(th[:, :H]).sum(-1)
        lp += norm.logpdf(th[:, H], 0.0, 1.0) + norm.logpdf(th[:, H + 1], -1.0, 0.5)
        lp += norm.logpdf(th[:, H + 2], 0.5, 0.7) + norm.logpdf(th[:, H + 3], -1.0, 0.5)
        return lp

    def chol_mask(self):
        H = 2 * self.J
        m = np.zeros((self.d, self.d), dtype=bool)
        for j in range(self.J):
            m[2 * j, 2 * j] = True
            m[2 * j + 1, 2 * j:2 * j + 2] = True
        m[H:, :] = True
        return np.tril(m)

    def theta_adj(self):
        a = self.chol_mask()
        a = a | a.T
        np.fill_diagonal(a, False)
        return a

    def x_adj(self):
        H, ns = 2 * self.J, self.n_sum
        a = np.zeros((self.d, self.dx), dtype=bool)
        for j in range(self.J):
            a[2 * j:2 * j + 2, j * ns:(j + 1) * ns] = True
        a[H:, :] = True
        return a


class RenewalChain:
    name = "rtchain"

    def __init__(self, T=30, sig_r=0.07, k=10.0, seed0=20.0, gi_mean=5.0,
                 gi_sd=2.5, K=10):
        self.d = self.dx = T
        self.sig_r, self.k, self.seed0, self.K = sig_r, k, seed0, K
        self.n_test = N_TEST
        sh, sc = (gi_mean / gi_sd) ** 2, gi_sd ** 2 / gi_mean
        w = np.diff(gamma_dist.cdf(np.arange(K + 1, dtype=float), sh, scale=sc))
        self.w = w / w.sum()

    def sample_joint(self, rng, n):
        T, K = self.d, self.K
        l = np.empty((n, T))
        l[:, 0] = np.log(1.1) + 0.2 * rng.standard_normal(n)
        for t in range(1, T):
            l[:, t] = l[:, t - 1] + self.sig_r * rng.standard_normal(n)
        I = np.empty((n, K + T))
        I[:, :K] = rng.poisson(self.seed0, size=(n, K)).astype(float)
        for t in range(T):
            lam = (I[:, t:t + K] * self.w[::-1]).sum(-1)
            muI = np.exp(l[:, t]) * np.maximum(lam, 1e-9)
            I[:, K + t] = rng.poisson(rng.gamma(self.k, muI / self.k)).astype(float)
        return np.log1p(I[:, K:]), l

    def log_prior(self, th):
        th = np.atleast_2d(th)
        lp = norm.logpdf(th[:, 0], np.log(1.1), 0.2)
        return lp + norm.logpdf(th[:, 1:], th[:, :-1], self.sig_r).sum(-1)

    def chol_mask(self):
        return band_mask(self.d, 1)

    def theta_adj(self):
        return chain_adj(self.d)

    def x_adj(self):
        return band_mask(self.d, 12) | band_mask(self.d, 12).T


TASKS = {
    "ssm":      ("State-Space Chain (m=20)",  lambda: StateSpaceChain(d=20)),
    "ssm300":   ("State-Space Chain (m=300)", lambda: StateSpaceChain(d=300, n_test=500)),
    "minihub":  ("Random-Effects Hub (m=10)", lambda: RandomEffectsHub(J=9)),
    "dor1w62c": ("Bi-Modal Gaussian (m=18)",  BiModalGaussian),
    "cnsmap":   ("Censored Lattice (m=36)",   CensoredLattice),
    "mlarnc20": ("Multilevel AR(1) (m=66)",   lambda: MultilevelAR1(J=20, T=8)),
    "hrl":      ("Hierarchical RL (m=28)",    HierarchicalRL),
    "rtchain":  ("Renewal Chain (m=30)",      RenewalChain),
}


def make_task(name):
    return TASKS[name][1]()
