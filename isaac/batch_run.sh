#!/usr/bin/env bash
# Stratified OmniLRS batch campaign.
#
# OmniLRS reads k_soil / theta_r / slope_deg at launch (physics material and
# terrain YAML), so they cannot vary per episode. This runner draws B batch
# vectors for those dims, relaunches the sim per batch with the values written
# into the sim YAML, and lets the bridge vary the live dims within each batch.
#
# Because per-episode rows then are NOT i.i.d. draws from Q, the bridge marks
# them source=2 and the local side should ingest them with
# --not-from-proposal (replay/ranking, cross-simulator comparison), unless
# you re-derive weights for the stratified design yourself.
#
#   isaac/batch_run.sh 8 50            # 8 batches x 50 episodes
set -euo pipefail
B=${1:-8}; N=${2:-50}
REPO=$(cd "$(dirname "$0")/.." && pwd)
TOPICS="$REPO/isaac/configs/ros2_topics.yaml"
SIM_YAML=$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$TOPICS'))['batch_yaml']['path'])")
LAUNCH=${OMNILRS_LAUNCH:-"$REPO/../OmniLRS/run.sh"}   # override with your launch command
OUT="$REPO/isaac/out"; mkdir -p "$OUT"
STAMP=$(date +%Y%m%d-%H%M%S)

# batch-tier draws from Q's marginals for the load-time dims
python3 - "$B" > "$OUT/batches_$STAMP.tsv" <<'PY'
import sys, numpy as np
sys.path.insert(0, ".")
from astraeus import priors as PR
B = int(sys.argv[1]); rng = np.random.default_rng(20260912)
X = PR.sample_proposal(rng, B)
for i, row in enumerate(X):
    d = dict(zip(PR.DIMS, row))
    print(f"{i}\t{d['k_soil']:.4f}\t{d['theta_r']:.4f}\t{d['slope_deg']:.3f}")
PY

while IFS=$'\t' read -r i K TH SL; do
  echo "=== batch $i: k_soil=$K theta_r=$TH slope_deg=$SL"
  python3 - "$SIM_YAML" "$K" "$TH" "$SL" <<'PY'
import sys, yaml
p, k, th, sl = sys.argv[1], *map(float, sys.argv[2:])
cfg = yaml.safe_load(open(p))
def put(d, path, v):
    ks = path.split("."); [d := d.setdefault(x, {}) for x in ks[:-1]]; d[ks[-1]] = v
put(cfg, "physics.ground.stiffness_scale", k)
put(cfg, "physics.ground.friction_angle_rad", th)
put(cfg, "terrain.regional_slope_deg", sl)
yaml.safe_dump(cfg, open(p, "w"))
PY
  "$LAUNCH" &  SIM_PID=$!
  sleep "${SIM_BOOT_S:-90}"
  python3 "$REPO/isaac/discover_topics.py" --config "$TOPICS" --write || { echo "topic map mismatch; aborting"; kill $SIM_PID; exit 1; }
  python3 "$REPO/isaac/omnilrs_bridge.py" --episodes "$N" --seed "$((20260912 + i))" \
      --fixed "k_soil=$K" "theta_r=$TH" "slope_deg=$SL" --batch-id "b$i-$STAMP" \
      --csv "$OUT/log_$STAMP.csv"
  kill $SIM_PID; wait $SIM_PID 2>/dev/null || true
done < "$OUT/batches_$STAMP.tsv"

echo "ingest: python -m astraeus ingest runs/omnilrs_$STAMP.npz $OUT/log_$STAMP.csv --not-from-proposal"
