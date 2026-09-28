# Notebooks

Run them in order; each is self-contained and takes a few minutes.

| Notebook | Key insights |
|---|---|
| `01_linear_regression_variance_reduction.ipynb` | The full replication: three-level hierarchy, MLDA sampling, and the standard vs. variance-reduction estimators. |

## 01 — Linear regression

We replicate and and extend the PyMC3 example
["Variance reduction in MLDA — Linear regression"](https://www.pymc.io/projects/examples/en/2022.01.0/samplers/MLDA_variance_reduction_linear_regression.html)
using the [tinyDA](https://github.com/mikkelbue/tinyDA) MLDA implementation with variance reduction extension.

A three-level hierarchy on the synthetic linear model: `y = 1 + 2x`with Gaussian noise, observed on 100 points. The levels differ only in how much of the data they see — `x[::3]` (34 points), `x[::2]` (50), and the full grid — coarse models are cheaper but biased.

The quantity of interest `Q` is the mean prediction over each level's own x-grid. 

### The main finding: two pairing conventions

The correction term is `Y_l = Q_l(θ) − Q_{l−1}(·)`, and the notebook shows that it matters
*which* coarse sample gets subtracted when the fine level **rejects**:

- **Proposal-paired** — pair with the coarse sample forwarded at that step, accepted or
  not. This is what tinyDA records and what the MLDA papers describe. On a rejection the
  two ends are decorrelated, so `Y` carries the full variance of both chains.
- **State-paired** — on a rejection, reuse the coarse sample that *produced* the current
  fine state. Every pair is then same-θ and `Y` collapses to the model discrepancy.

Measured difference-term variances differ by orders of magnitude between the two, and only
the state-paired version reproduces the gain reported in the original notebook. The
proposal-paired version is shown too and performs overall worse - its performance being tied closely to subchain length udn gengerally worse.

Further exploration still TO DO.

## Requirements

Installed from the repo root; see the top-level README. The notebooks need `tinyDA`
(from the fork pinned in `pixi.toml`), `numpy`, `scipy`, `matplotlib`, `arviz`,
`ray`, and `ipywidgets`.

`ray` is required to import tinyDA at all — the import is unconditional at module level
but ray is not an install requirement, so a plain `pip install tinyda` fails on import.

## TODO

- [ ] Multi-seed (5–10) SE curves with median and IQR band, plus a bias check of the
      state-paired estimator against a proper reference
- [ ] Decide on the Q definition: evaluate on the fine grid at every level, or keep the
      per-level grid, explain the artifact
- [ ] Autocorrelation pre-run to set the subsampling rate properly — pymc3 tutorial
      recommends this
- [ ] Demo run with `randomize_subchain_length` on vs. off, reporting ESS
- [ ] Resolve or document the `AdaptiveMetropolis` anomaly (frozen chains, ESS ≈ 6, while
      `GaussianRandomWalk(adaptive=True)` behaves normally)