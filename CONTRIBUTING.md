# Contributing

Astraeus is a research framework; correctness and honesty about what the model does and
does not capture matter more than features. Please keep that spirit.

## Setup

```bash
git clone https://github.com/vineigenphase/Non-Trivial---Astraeus.git
cd Non-Trivial---Astraeus
python -m pip install -r requirements-dev.txt
make check          # ruff + pytest + 19 selftest invariants
```

## Ground rules

- **Determinism is a contract.** Every episode is a pure function of `(x, seed)`. If a change
  alters trajectories, say so in the PR and regenerate `docs/VALIDATION.md` numbers with
  `make campaign`.
- **Do not hide support mismatch.** Priors whose mass lies outside the simulated support must
  read `n/a`, never a clean `0.000`.
- **Keep provenance.** Proposal (`mc`), adversarial (`cem`) and external simulator rows are
  never mixed into the same importance-weighted estimate.
- **Model changes need a citation or a measurement.** New terramechanics, photometry or
  perception terms should reference a source in `docs/RESEARCH.md` or be labelled as a
  heuristic in `docs/PARAMETERS.md`.
- **Isaac Sim / ROS 2 code you cannot run here must be marked unverified** (see
  `isaac/configs/ros2_topics.yaml`).

## Before opening a PR

```bash
make check                                  # must be green
python scripts/make_renders.py runs/val.npz  # only if the renderer or physics changed
```

CI runs ruff, `compileall`, pytest, the selftest, and boots the FastAPI server to hit the API on
Python 3.10 and 3.12.
