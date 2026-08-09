# Astraeus — TONIGHT runbook

## 0. Upload these files (clipboard-free route)
Open the Jupyter link on port 8888 -> navigate to /workspace -> New folder
`astraeus` -> Upload button -> drop priors.py, harness.py, analyse.py,
batch_runner.sh. (Jupyter upload works even when the terminal clipboard
doesn't.)

## 1. Fix the TOPICS block  (BLOCKING, 5 min)
Terminal 2:
    cd /workspace/OmniLRS && pixi shell
    ros2 topic list -t
Paste the list to Claude OR edit TOPICS in harness.py yourself (via Jupyter's
editor). Also confirm msg construction in push_sun / push_terrain / teleport
matches the real types.

## 2. Pilot  (BLOCKING, ~20 min)
Sim up in Terminal 1 (pixi run ros2, wait for full load). Terminal 2:
    cd /workspace/astraeus && pixi shell   # or run via: pixi run -e ros2 python
    python harness.py --episodes 5 --k-soil 1.0 --theta-r 1.047 \
        --batch-id pilot --log /workspace/astraeus/pilot.csv
Watch: does the rover drive? does terrain actually regen? does sun move?
does teleport work? Fix before committing the night. Note seconds/episode.

## 3. The overnight batches
k_soil / theta_r are LOAD-TIME yaml params -> 3 batches, 3 sim launches.
Edit cfg/environment/astraeus_base.yaml between launches (or pre-make 3 yamls):
  batch A: k_soil 0.7 (scale force_depth_regression slopes x0.7), theta_r 0.70
  batch B: k_soil 1.0 (stock),                                    theta_r 1.047
  batch C: k_soil 3.0,                                            theta_r 0.61
Launch sim with batch yaml -> then:
    python harness.py --episodes N --k-soil <val> --theta-r <val> \
        --batch-id A --base-seed 20260808 --log /workspace/astraeus/log.csv
(base-seed +1000 per batch to avoid seed overlap.)
N: pick from pilot timing. e.g. 40 s/ep -> ~85 ep/batch fits a 3h slot;
3 batches ~ overnight with sim relaunches. IMPORTANT: run each inside tmux
or nohup so the web terminal dropping doesn't kill the run:
    nohup python harness.py ... >> /workspace/astraeus/batchA.out 2>&1 &

## 4. Before sleep + at morning
Download log.csv via Jupyter (right-click -> Download) BOTH times.
Morning:
    python analyse.py /workspace/astraeus/log.csv
-> summary.txt + two PNGs = the report's results section. STOP THE POD.

## Honest-reporting notes baked into the design
- GT-pose policy => dust + lighting cannot cause failures. Lighting is swept
  and logged to demonstrate the pipeline; report terrain-axis results as the
  empirical finding, dust/vision as implemented-pending-policy.
- P3's sun default (45 deg) has ~zero density on [0.5, 6]: that IS the
  Earth-convention finding, quantified. Do not widen the kernel to hide it.
- 3 discrete (k_soil, theta_r) levels is a coarse grid, not a sampled
  marginal — describe batch-tier params as "conditioned levels" in the report.
- If terrain randomisation cannot change relief scale at runtime, flip
  R_TERRAIN_IS_EPISODE_TIER=False in priors.py and add it to the batch yamls.
