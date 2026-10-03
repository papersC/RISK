# RISK-OCS code and data

Simulator, controllers, experiments and figure scripts for the manuscript
"RISK-OCS: Risk-Budgeted Optical Reconfiguration for Multi-Tenant AI Training Fabrics".

## Layout

| Path | Content |
|---|---|
| `riskocs/sim_core.py` | Compiled (Numba) event-driven fluid core: circuit sharing, max-min packet fallback, outages, greedy b-matching, reactive continuation |
| `riskocs/fabric.py` | State, cloning, causal replay, live run loop, paired counterfactual harm logging |
| `riskocs/workloads.py` | Synthetic DP, PP and MoE workload generators and the uncertainty process |
| `riskocs/controllers.py` | Packet only, Static OCS, Reactive, Cost-aware, Nominal (with hysteresis), RISK-OCS, ablations, Clairvoyant |
| `riskocs/runner.py` | Seeded instance construction, calibration trace, isolated-time reference |
| `tests/test_core.py` | Eight hand-checkable validation cases (`pytest -q`) |
| `experiments.py` | All experiments (`hyst`, `e1` to `e6`, or `all`) |
| `analysis.py` | Bootstrap statistics, LaTeX tables, figures and `figures/numbers.tex` |
| `results/` | Raw per-run summaries (pickle) used in the paper |
| `figures/` | Generated figures and tables |

## Reproduce

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q tests
PYTHONPATH=. python experiments.py all      # about 1.5 h on 2 cores
PYTHONPATH=. python analysis.py
```

Validation seeds are 200-209 (only used to match the hysteresis margin).
Test seeds are 1000-1024. Units: ms, MB, MB/ms (1 MB/ms = 8 Gb/s).

## Experiments

* `hyst` matches the Nominal hysteresis margin to the RISK-OCS switching rate on validation seeds.
* `e1` main comparison (3 mixes, 25 seeds, 50 iterations) and ablations.
* `e2` switching delay x jitter regime map.
* `e3` budget and hysteresis frontiers.
* `e4` decision-time overhead versus scenarios and fabric size (single worker).
* `e5` single-tenant protection.
* `e6` sensitivity (regional outage, packet ratio, ports, holding interval).

Decision times are measured on a Python prototype and are not representative of an optimised controller.
