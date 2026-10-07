"""Fit Copula NPE on the State-Space Chain (m = 20) with N = 100 simulations
and print LPP next to the anchors, the conditional mutual information
diagnostic and a sample check."""
import numpy as np

import copnpe as C
import tasks as T

task = T.make_task("ssm")
x, th = T.training_set(task, N=100, seed=0)
model = C.fit_copula_npe(task, x, th, C.CONFIGS["ssm"], seed=0)

X, TH = T.test_set(task)
print(f"LPP {C.lpp(model, X, TH):.3f}   anchors {T.anchors(task)}")
nodes, I = C.conditional_mi(model.cop, X[:200], C.neighbour_sets(task))
print(f"mean conditional MI (1/m) sum_j I_j = {I.sum() / task.d:.2e}")
samples = model.sample(np.random.default_rng(0), X[0], 2000)
print("posterior sample mean vs exact:", np.round(samples.mean(0)[:5], 3),
      np.round(task.posterior_mean(X[:1])[0, :5], 3))
