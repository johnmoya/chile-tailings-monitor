# chile-tailings-monitor

Monitoring Chilean tailings storage facilities (tranques de relave) with
Sentinel-2 L2A, 2019–2026.

## Research question

Does a U-Net trained on **weak labels** measure the supernatant pond
(laguna) better than an **MNDWI + Otsu** baseline on deposits it has **never
seen**? If it does not beat the baseline, that is reported as is: a negative
result is a valid result.

## v1 scope

- 10–15 large deposits (Coquimbo and centre-north), registry in
  `configs/deposits.geojson` (geometries from Sernageomin's Plataforma
  Pública de Relaves; loaded by hand, not shipped).
- Binary segmentation water / not water inside the deposit polygon.
- Derived metrics: pond area, minimum pond–crest distance, beach fraction.
- Bands B2, B3, B4, B8, B11, B12; 20 m bands resampled to 10 m; 256×256 tiles.
- Split **by deposit** (train / val / test), never by tile. Test deposits
  are scored exactly once with `scripts/evaluate_test.py`.

## Layout

```
src/ctm/
  data/       STAC search + COG windowed reads, deposit registry, tiling, datasets
  baseline/   MNDWI + Otsu inside the polygon
  labels/     weak labels from the baseline + manual-correction hook (interface)
  models/     U-Net (ResNet34, 6 bands), sliding-window inference
  train/      training loop (bf16 AMP, seeds, checkpoints, JSONL metrics)
  metrics/    IoU / F1, scene-level U-Net vs baseline evaluation
  utils/      config, seeding, geometry, single-OOS-run lock
configs/      default.yaml, deposits.geojson
scripts/      check_gpu.py, train_smoke.py, evaluate_test.py
tests/        pytest (markers: network, gpu)
data/         git-ignored: scenes, tiles, runs
reports/      results; reports/locks/ holds the committed test-evaluation lock
```

## Running M0

Requires [uv](https://docs.astral.sh/uv/) on Linux x86_64 (developed under
WSL2 Ubuntu 24.04, RTX 5070).

```bash
uv sync                                      # Python 3.12 + torch 2.14.1+cu130
uv run python scripts/check_gpu.py           # torch/CUDA/GPU/VRAM + bf16 matmul
uv run pytest                                # everything, incl. network + gpu tests
uv run pytest -m "not network and not gpu"   # what CI runs
uv run python scripts/train_smoke.py         # 1 epoch on synthetic tiles, peak VRAM
```

CPU-only install (what CI uses):

```bash
uv sync --no-default-groups --group dev --group cpu
```

Configuration lives in `configs/default.yaml`; scripts accept
`key.path=value` overrides (e.g. `train.device=cpu`).

### Test-set lock

`scripts/evaluate_test.py` refuses to run without
`--i-understand-this-is-the-single-oos-run`. It checks every precondition
(checkpoint, real test deposits, scenes with reference masks) first, then
atomically writes `reports/locks/test_evaluation.lock` **before** reading
test data, and refuses forever after. Commit the lock with the results.
