# Research background and next phase

This document has two parts. **Part A** lists the published work Astraeus builds on and states, for
each item, which part of the code uses it. **Part B** is the clearly labelled next-phase research
programme of the repository author (`vineigenphase`); it describes planned and in-progress work and
makes **no claim of published results**.

---

## Part A — Literature

### A.1 Adaptive stress testing (AST)

1. **Lee, R., Kochenderfer, M. J., Mengshoel, O. J., Brat, G. P., Owen, M. P.** (2015).
   *Adaptive stress testing of airborne collision avoidance systems.* IEEE/AIAA 34th Digital Avionics
   Systems Conference (DASC). — The original AST formulation: treat the simulator as a black-box MDP,
   let a search agent choose disturbances, reward it for driving the system towards failure while
   penalising implausible disturbances.
   *Used in:* the framing of the whole project; the severity objective in `campaign.py::cem_search`
   is the AST reward (distance-to-failure plus a log-likelihood plausibility term under the chosen prior).

2. **Lee, R., Mengshoel, O. J., Saksena, A., Gardner, R. W., Genin, D., Silbermann, J., Owen, M.,
   Kochenderfer, M. J.** (2020). *Adaptive stress testing: finding likely failure events with
   reinforcement learning.* Journal of Artificial Intelligence Research 69, 1165–1201. — The
   consolidated AST paper: MCTS and deep RL solvers, the plausibility-weighted reward, and the argument
   that *most likely* failures matter more than *worst possible* ones.
   *Used in:* the decision to rank elites by severity **under a prior** (`Campaign.elites(prior=...)`)
   rather than by raw severity.

3. **Koren, M., Alsaif, S., Lee, R., Kochenderfer, M. J.** (2018). *Adaptive stress testing for
   autonomous vehicles.* IEEE Intelligent Vehicles Symposium (IV). — AST applied to a ground vehicle with
   pedestrians; introduces the "solver sees only the seed" black-box interface.
   *Used in:* the `(x, seed)` interface of `rover.simulate_one`; the simulator is never opened to the
   search.

4. **Corso, A., Moss, R. J., Koren, M., Lee, R., Kochenderfer, M. J.** (2021). *A survey of algorithms
   for black-box safety validation of cyber-physical systems.* Journal of Artificial Intelligence
   Research 72, 377–428. — Taxonomy of falsification, most-likely-failure search and failure-probability
   estimation; positions AST relative to importance sampling.
   *Used in:* the split of Astraeus into (i) failure-probability estimation by importance sampling and
   (ii) most-likely-failure search by CEM, kept separate in `Campaign.source` so neither contaminates
   the other.

5. **Moss, R. J., Corso, A., Caers, J., Kochenderfer, M. J.** (2024). *BetaZero: belief-state planning
   for long-horizon POMDPs using learned approximations* and the associated AST toolbox work
   (`sisl/AdaptiveStressTestingToolbox`). — Modern tooling reference for AST solvers.
   *Used in:* as a design reference for keeping the solver replaceable; CEM is the first solver, not the
   only intended one (see Part B).

### A.2 Failure-probability estimation under uncertain priors

6. **Hesterberg, T.** (1995). *Weighted average importance sampling and defensive mixture
   distributions.* Technometrics 37(2), 185–194. — Sampling from a mixture that includes a wide
   "defensive" component bounds the importance weights and keeps every target's ESS from collapsing.
   *Used in:* `priors.PROPOSAL = 0.4·P1 + 0.15·(P2+P4+P5+P6)`; the docstring of `priors.Mixture`.

7. **Veach, E., Guibas, L. J.** (1995). *Optimally combining sampling techniques for Monte Carlo
   rendering.* SIGGRAPH. — The balance heuristic for multiple importance sampling.
   *Used in:* the mixture density `q(x) = Σ α_k p_k(x)` evaluated exactly in `Mixture.pdf`.

8. **Kong, A.** (1992). *A note on importance sampling using standardized weights.* University of
   Chicago, Dept. of Statistics Tech. Rep. 348; and **Kong, Liu, Wong** (1994), JASA 89(425). — The
   effective-sample-size diagnostic `ESS = (Σw)² / Σw²`.
   *Used in:* `priors.ess`, the `MIN_ESS = 30` reliability gate in `campaign.py`, and the ESS bars on
   `/internals`.

9. **Owen, A. B.** (2013). *Monte Carlo theory, methods and examples*, ch. 9 (Importance sampling).
   — Self-normalised importance sampling, its bias, and why support mismatch is unrecoverable.
   *Used in:* the `support_coverage` check that labels P3 *not estimable* instead of reporting 0.000 as
   a failure rate.

10. **Efron, B., Tibshirani, R. J.** (1993). *An Introduction to the Bootstrap.* — Percentile bootstrap.
    *Used in:* `campaign.bootstrap_ci` (weighted bootstrap of the self-normalised rate, 400 replicates).

### A.3 Rare-event search

11. **Rubinstein, R. Y., Kroese, D. P.** (2004). *The Cross-Entropy Method: A Unified Approach to
    Combinatorial Optimization, Monte-Carlo Simulation and Machine Learning.* Springer. — CEM as an
    adaptive importance-sampling / optimisation method.
    *Used in:* `Campaign.cem_search`: Gaussian proposal in bounded space, elite quantile refit,
    plausibility-penalised severity.

12. **O'Kelly, M., Sinha, A., Namkoong, H., Duchi, J., Tedrake, R.** (2018). *Scalable end-to-end
    autonomous vehicle testing via rare-event simulation.* NeurIPS. — Adaptive importance sampling
    (cross-entropy) to estimate very small failure probabilities for AV stacks.
    *Used in:* the argument for keeping CEM samples out of the estimator (they are biased samples; see
    `Campaign.source == 1`), and the proposal-refit structure.

13. **Kim, Y., Kochenderfer, M. J.** (2016). *Improving aircraft collision risk estimation using the
    cross-entropy method.* Journal of Air Transportation 24(2). — CEM for aviation safety estimation.

### A.4 Distribution shift and robustness

14. **Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., Lawrence, N. D. (eds.)** (2009).
    *Dataset Shift in Machine Learning.* MIT Press. — Covariate shift and importance-weighted
    correction.
    *Used in:* the conceptual basis for re-weighting a fixed campaign under alternative priors.

15. **Rahimian, H., Mehrotra, S.** (2019). *Distributionally robust optimization: a review.*
    arXiv:1908.05659. — Ambiguity sets over the disturbance distribution.
    *Used in:* Part B (planned worst-case-over-priors objective).

16. **Sinha, A., Namkoong, H., Duchi, J.** (2018). *Certifying some distributional robustness with
    principled adversarial training.* ICLR. — Wasserstein ambiguity sets.
    *Used in:* Part B.

### A.5 Lunar terramechanics and rover mobility

17. **Bekker, M. G.** (1969). *Introduction to Terrain–Vehicle Systems.* University of Michigan Press.
    — Pressure–sinkage law `p = (k_c/b + k_φ) zⁿ`.
    *Used in:* `terramechanics.py::sinkage`; `k_soil` scales the sinkage modulus.

18. **Wong, J. Y., Reece, A. R.** (1967). *Prediction of rigid wheel performance based on the analysis
    of soil-wheel stresses.* Journal of Terramechanics 4(1), 81–98. — Stress distribution under a
    driven rigid wheel; slip–sinkage.
    *Used in:* the slip-dependent digging term (`DIG_GAIN·slip²`) and traction saturation.

19. **Janosi, Z., Hanamoto, B.** (1961). *The analytical determination of drawbar pull as a function
    of slip for tracked vehicles in deformable soils.* 1st Int. Conf. ISTVS. — Shear stress–slip law
    `τ = τ_max (1 − e^{−j/K})`.
    *Used in:* `terramechanics.py`: thrust as a function of slip and friction angle `theta_r`.

20. **Ding, L., Gao, H., Deng, Z., Nagatani, K., Yoshida, K.** (2011). *Experimental study and analysis
    on driving wheels' performance for planetary exploration rovers moving in deformable soil.*
    Journal of Terramechanics 48(1), 27–45. — Planetary-wheel slip–sinkage measurements.

21. **Arvidson, R. E. et al.** (2010). *Spirit Mars Rover mission: overview and selected results from
    the northern Home Plate winter haven to the side of Scamander crater.* JGR Planets 115. — The
    Troy entrapment: high-slip sinkage into loose sulfate-rich soil under a crust.
    *Used in:* the `stuck` outcome definition ("Spirit-at-Troy class") in `rover.py`.

22. **Carrier, W. D., Olhoeft, G. R., Mendell, W.** (1991). *Physical properties of the lunar surface.*
    In *Lunar Sourcebook* (Heiken, Vaniman, French, eds.), ch. 9. — Regolith friction angle 35–50°,
    density, cohesion.
    *Used in:* the `theta_r` support 0.61–0.87 rad.

### A.6 Lunar lighting, dust and perception

23. **Mazarico, E., Neumann, G. A., Smith, D. E., Zuber, M. T., Torrence, M. H.** (2011).
    *Illumination conditions of the lunar south pole using high resolution Digital Elevation Models
    from LOLA.* Icarus 211(2), 1066–1081. — Sun elevation at the poles stays within a few degrees.
    *Used in:* the `sun_e` support 0.5–6° and the P5 "deep winter" marginal.

24. **Colaprete, A. et al.** (2021). *The Volatiles Investigating Polar Exploration Rover (VIPER)
    mission.* LPSC / Space Science Reviews. — Landing-site characterisation and mobility
    requirements that motivate P2.

25. **Metzger, P. T., Lane, J. E., Immer, C. D., Clements, S.** (2010). *Cratering and blowing soil
    by rocket engines during lunar landings.* — Plume ejecta and optics contamination that motivate P6.

26. **Hapke, B.** (1981). *Bidirectional reflectance spectroscopy 1. Theory.* JGR 86(B4). — Regolith
    photometry; opposition surge and low-phase-angle contrast.
    *Used in:* qualitatively in `perception.py` (contrast collapses at grazing sun) and `render.py`
    (shading model).

### A.7 Simulation platforms

27. **Richard, A., Kamohara, J., Uno, K., Santra, S., van der Meer, D., Olivares-Mendez, M.,
    Yoshida, K.** (2024). *OmniLRS: An open-source lunar robotics simulator for NVIDIA Isaac Sim.*
    ICRA. — The Isaac Sim lunar environment the ROS 2 bridge targets.
    *Used in:* `isaac/omnilrs_bridge.py`, `isaac/configs/ros2_topics.yaml`.

28. **NVIDIA** (2024). *Isaac Sim documentation — standalone Python workflows, PhysX terrain and
    articulation APIs.* — `isaac/standalone_env.py`.

29. **Allan, M. et al.** (2019). *Planetary rover simulation for lunar exploration missions.*
    IEEE Aerospace Conference. — Survey of what rover simulators can and cannot reproduce
    (terramechanics vs rigid contact).
    *Used in:* the documented gap in `docs/ISAAC_SIM.md`.

---

## Part B — Next phase: the author's research programme

> **Status: planned / in progress. Nothing below has been peer-reviewed or published. This section is
> a statement of intent by the repository author and is kept separate from Part A so that no reader
> confuses it with established results.**

### B.1 Thesis

Mission safety cases for autonomous rovers are usually built against a *single* environmental
prior — most often the simulator's shipped defaults, which are inherited from terrestrial
conventions (e.g. a 45° sun). The author's programme asks: **when the disturbance prior is itself
uncertain, how should a stress-testing campaign be designed, reported and acted on so that the
conclusions survive the prior being wrong?**

Astraeus v1 (the `legacy/` OmniLRS harness, three priors, five dimensions) established the core
empirical observation that motivates this: the simulator-default prior P3 places essentially zero
mass on the lighting the rover will actually meet, so any campaign that trusts P3 is not testing
the mission at all. That finding is now quantified by `support_coverage` and reported for every
prior.

### B.2 Planned contributions

1. **Prior-robust AST objective.** Replace "most likely failure under P_k" with a
   *worst-case-over-priors* objective: `max_x severity(x) · min_k [p_k(x)/q(x)]` over an ambiguity
   set of priors (Part A refs 15–16), so the elites found are failures no candidate prior can dismiss
   as implausible. Implementation target: a second solver next to `cem_search` sharing the same
   `Campaign` state.

2. **Campaign design for bounded ESS across priors.** Choose the mixture weights `α_k` of the
   defensive proposal to maximise the minimum ESS over candidate priors for a fixed budget, and
   adapt them between chunks (sequential design). Baseline: the fixed 0.40/0.15 mixture used now.

3. **Shift-sensitivity certificates.** For each reported failure-mode ranking, compute the smallest
   perturbation of the prior (in a KL or Wasserstein ball) that flips the ranking. A ranking that
   flips under a tiny perturbation should not be used in a safety case; Astraeus currently reports
   only *whether* the ranking shifts between the six discrete priors.

4. **Cross-simulator consistency.** Run identical `(x, seed)` episodes in the NumPy simulator and in
   Isaac Sim / OmniLRS (same procedural terrain via `isaac/export_terrain.py`), ingest both with
   provenance, and quantify where the reduced-order model and the physics engine disagree on the
   *outcome*, not just on trajectories. Rigid PhysX will under-report `stuck`; the size of that gap
   is itself a result.

5. **Perception in the loop.** Replace the parametric visibility/VO-drift model in `perception.py`
   with rendered images from the software renderer (or a Reactor world model) fed to a real
   detector and VO front end, so that `sun_e`, `sun_psi` and `tau_dust` act through pixels rather
   than through fitted curves. The renderer already carries the true sun vector, cast shadows and
   dust extinction for this reason.

6. **Calibrated priors.** Replace the engineering-judgement marginals of P2/P4/P5/P6 with
   distributions fitted to published site characterisations (LOLA illumination, LROC rock counts,
   Apollo/Chang'e soil mechanics), with provenance recorded per marginal.

### B.3 Evaluation plan

- Reproduce the v2 validation campaign (docs/VALIDATION.md) as the fixed baseline.
- For each contribution, report: change in minimum ESS across priors at equal budget; stability of
  the failure-mode ranking under prior perturbation; agreement rate of outcomes between simulators;
  and the number of elite failures that remain plausible under *all* candidate priors.
- All campaigns saved as `.npz` with seeds so every reported elite is replayable.

### B.4 How to contribute to this phase

Open an issue tagged `next-phase` with the contribution number above. Calibrated marginals with a
cited source, a verified OmniLRS topic map, or a terramechanics plugin for Isaac Sim are the
highest-value contributions.
