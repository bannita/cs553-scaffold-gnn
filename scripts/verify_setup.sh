#!/usr/bin/env bash
# Day-one environment check on VERNE. Run this before anything else.
set -u
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "NO GPU VISIBLE"
echo
echo "=== python / torch ==="
python -c "import sys; print('python', sys.version.split()[0])"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -1
echo
echo "=== project libs ==="
for p in rdkit sklearn pandas numpy torch_geometric ogb transformers; do
  python -c "import $p" 2>/dev/null && echo "  OK   $p" || echo "  --   $p (not installed)"
done
echo
echo "=== dataset reachability ==="
python - <<'PY' 2>&1 | tail -3
import os
os.environ.setdefault("SCAFFOLDGEN_DATA", "/home/jovyan/data")
try:
    from ogb.graphproppred import PygGraphPropPredDataset
    d = PygGraphPropPredDataset("ogbg-molhiv", root=os.environ["SCAFFOLDGEN_DATA"])
    print("OGB molhiv OK:", len(d), "graphs")
except Exception as e:
    print("OGB FAILED:", type(e).__name__, e)
PY
echo
echo "=== persistent storage ==="
df -h /home/jovyan 2>/dev/null | tail -1 || echo "/home/jovyan not found"
