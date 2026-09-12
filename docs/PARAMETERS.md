# The eight disturbance dimensions

The disturbance vector is, in this fixed order (`astraeus.priors.DIMS`):

| # | Name | Unit | Support | Meaning | Where it acts |
|---|---|---|---|---|---|
| 1 | `k_soil` | × nominal | [0.5, 5.0] | Multiplier on the Bekker sinkage moduli `k_c`, `k_phi`. 1.0 = Lunar Sourcebook nominal regolith; low = soft, high = firm. | `terramechanics.py`: static sinkage `z0`, compaction resistance, entrapment. |
| 2 | `theta_r` | rad | [0.61, 0.87] (35–50°) | Internal friction angle of the regolith. Sets available shear thrust via `W tan φ` and the angle of repose. | `terramechanics.py`: max thrust, slip at demand; Isaac: PhysX friction `tan θ_r`. |
| 3 | `r_terrain` | × nominal | [0.5, 2.0] | Relief-scale multiplier on the fractal heightfield; also scales rms slope used for self-shadowing. | `terrain.py`: height amplitude; `perception.py`: shadow fraction. |
| 4 | `rho_rock` | rocks / 100 m² | [0, 12] | Areal density of rocks with diameter ≥ 0.2 m. Count per episode ~ Poisson(ρ·A/100) over the ≈704 m² corridor. | `terrain.py`: rock field; `rover.py`: detection, avoidance, collision. |
| 5 | `slope_deg` | deg | [0, 15] | Regional slope magnitude along a seed-chosen direction. | `terrain.py`: plane term; `rover.py`: pitch/roll, gravity load on wheels. |
| 6 | `sun_e` | deg | [0.5, 6.0] | Solar elevation. Polar sites live in this band; the OmniLRS/Isaac default of 45° lies **outside** it. | `perception.py`: shadow fraction, glare; `render.py`: light direction; Isaac: distant-light rotation. |
| 7 | `sun_psi` | deg | [0, 360) | Solar azimuth, clockwise from the +x drive axis. | `perception.py`: glare when Δψ to heading < ~35°; `render.py`; Isaac. |
| 8 | `tau_dust` | optical depth | [0, 0.6] | Suspended / lens-deposited dust. Beer–Lambert contrast loss `exp(-2τ)`. | `perception.py`: visibility; `render.py`: haze; Isaac: fog density and light attenuation `exp(-1.3τ)`. |

Bounds are the *simulator support*: samples outside are not clipped silently
by the campaign; `priors.support_coverage` reports the probability mass a
prior puts outside them.

## Two baselines

* `SIM_DEFAULTS` — the OmniLRS/Isaac Sim shipped scene (`sun_e = 45°`,
  `tau_dust = 0`, `theta_r = 1.047` ≈ 60°). Deliberately kept **out of
  support** where it is out of support: it is the object under test for P3.
* `NOMINAL_X` — an in-support, VIPER-like point used as the default for
  `python -m astraeus whatif` and `isaac/export_terrain.py`.

## Candidate priors

All priors are products of independent marginals (the independence assumption
is discussed in RESEARCH.md).

| Prior | Site hypothesis | Notable marginals |
|---|---|---|
| **P1** max ignorance | Widest admissible support | `k_soil` log-uniform; all others uniform on the bounds. The coverage component of `q`. |
| **P2** VIPER-spec south pole | Firmer soil, moderate relief, low sun | `k_soil` log-triangular mode 2.0; `theta_r` U(38°,45°); `sun_e` Tri(0.5, 6, mode 1.5); `tau_dust` Tri(0, 0.3, mode 0.05). |
| **P3** simulator default | Tight Gaussians (sd = 5 %) around `SIM_DEFAULTS` | `sun_e` ≈ N(45°, 2.25°) — essentially zero density on [0.5, 6]. **Unsupported by construction.** |
| **P4** equatorial mare | Flat, sparse rocks, high sun sampled at its low edge | `sun_e` U(4, 6); `slope_deg` Tri(0, 6, mode 1.5). Partially out of `q`'s heavy-density region → small ESS expected. |
| **P5** PSR rim, deep winter | Grazing sun, steep flanks, loose soil, ejecta | `k_soil` log-Tri mode 0.8; `slope_deg` Tri(4, 15, mode 9); `rho_rock` Tri(2, 12, mode 6); `sun_e` Tri(0.5, 2.5, mode 0.8). |
| **P6** post-landing plume | P2 with contaminated optics | `tau_dust` Tri(0.25, 0.6, mode 0.45). |

## Proposal

```
q = 0.40·P1 + 0.15·(P2 + P4 + P5 + P6)
```

Every simulated episode is drawn from `q` (Hesterberg's defensive mixture).
Because each supported prior is a mixture component, its ESS is bounded below
by roughly `α_k·n` instead of collapsing when the prior is narrow. P3 is
intentionally *not* a component: including it would spend 15 % of the budget
at 45° sun, which the rover will never meet at a polar site.

## Importance re-weighting and reliability gates

For prior `P_k`, on proposal rows only:

```
w_i = p_k(x_i) / q(x_i),   w̃_i = w_i / Σ w
p̂_fail = Σ w̃_i · 1[fail_i]
ESS    = (Σ w)² / Σ w²          (Kish)
CI     = weighted bootstrap, 95 %
```

An estimate is flagged **unreliable** (`campaign.MIN_ESS`) when

* ESS < 30, or
* support coverage < 0.5 — i.e. more than half of the prior's own mass lies
  where `q` never samples (P3).

Alongside the flag every report carries `ess_frac = ESS/n`, the largest
normalised weight `max_weight_frac`, and `support_coverage`, so a borderline
estimate can be judged rather than trusted. The internals page (`/internals`)
shows the weight histogram, per-dimension marginal coverage, and the
failure-geometry scatter behind each number.

## Outcomes and severity

| Outcome | Trigger |
|---|---|
| `success` | true position within 1.5 m of the waypoint. |
| `stuck` | wheels immobilised (demand > available thrust) ≥ 8 s, slip–sinkage past 0.5·R, or < 0.30 m progress in 15 s. |
| `tip_over` | \|pitch\| or \|roll\| ≥ 30°. |
| `collision` | body contact with a rock taller than 0.30 m ground clearance. |
| `nav_miss` | the policy *believes* it has arrived (estimate within 0.5 m) and halts, but the true position is outside 1.5 m. Pure perception failure. |
| `timeout` | 180 s of sim time elapsed. |

```
severity = 1[fail] + 0.5·max_tilt/30° + 0.5·max_slip + 0.3·(1 − min_visibility) + final_dist/30 m
```

Severity orders episodes for the elite list and CEM; it is not a probability.

## Rover and controller constants

| Constant | Value | Source / note |
|---|---|---|
| Mass | 430 kg | VIPER-class |
| Wheelbase × track | 1.8 × 1.4 m | rigid footprint used for attitude |
| Wheel | R 0.25 m, width 0.20 m, ×4 | rigid |
| Ground clearance | 0.30 m | |
| Gravity | 1.62 m/s² | |
| Commanded speed / yaw rate | 0.5 m/s / 0.6 rad/s | |
| Sensor | 7 m look-ahead, ±55° FOV, detection rate 1.5 s⁻¹ · V | rocks in shadow ×0.15 |
| VO drift | 0.035 m per m travelled at V = 0, ∝ (1 − V) | `--gt-pose` disables |
| Regolith nominal | `k_c` 1400, `k_phi` 8.2e5, n 1.0, c 170 Pa, K 0.018 m | Lunar Sourcebook ch. 9; Wong 2008 |
