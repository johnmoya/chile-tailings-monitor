# Pre-registration — chile-tailings-monitor v1

> **Status: DRAFT, not frozen.** Items marked **[OPEN]** must be settled before
> the freeze. Freezing = committing this file with status "FROZEN" and the git
> tag `prereg-v1`, BEFORE any model is trained on real data. Published tags are
> never moved; changes after the freeze go to the "Deviations" section.

## 1. Question

Does a U-Net trained only on **weak labels** (masks produced by MNDWI + Otsu)
segment the supernatant pond of Chilean tailings deposits **better than the
MNDWI + Otsu baseline itself**, on deposits **never seen during training or
model selection**?

A negative or null result is reported with the same prominence as a positive one.

## 2. Data and units of analysis

- Sentinel-2 L2A (Earth Search `sentinel-2-l2a`), 2019-01-01 to 2026-12-31.
  Bands B2, B3, B4, B8, B11, B12; 20 m bands resampled to 10 m.
  DN → reflectance with each asset's own scale/offset (processing baseline ≥ 04.00
  offset handled per scene).
- One fixed MGRS tile per deposit: the tile whose footprint contains the deposit
  bbox and has the most acquisitions in the period.
- Validity mask from SCL: classes 0, 1, 8, 9, 10, 11 always invalid; class 3
  (cloud shadow) invalid only if SCL cloud pixels (8, 9, 10) lie within 1500 m of
  the deposit bbox.
- **Unit of analysis: the deposit.** Scenes are repeated measurements nested in
  deposits.
- **Reference:** manual masks (0 = not water, 1 = water, 255 = ignore) on
  8 scenes per val/test deposit, selected by the fixed procedure in §6.
  Annotators never see MNDWI, baseline or model outputs.

## 3. Split rule

- The split is assigned **per deposit** in `configs/deposits.geojson`
  (train / val / test). Tiles, scenes and labels inherit it. No split by tile
  or by scene, ever.
- The assignment is frozen in the same commit as this document.
- **[OPEN]** Number of test deposits (proposal: 3–4 of 10–15) and the
  stratification criteria (region, pond size, latitude/altitude).
- Test deposits are used for exactly two things before the final run:
  (a) the catalog / mask diagnostics (SCL statistics only), (b) exporting
  imagery for manual annotation. No model or baseline output is computed on them.
- Everything that is tuned (architecture, epochs, learning rate, sigmoid
  threshold, normalisation statistics, augmentation) is chosen on **train/val
  only**. Normalisation statistics come from train deposits only.

## 4. Metrics

Scored pixels for a scene: inside the deposit polygon ∩ valid (SCL rules
above) ∩ reference ≠ 255. Both methods are scored on exactly the same pixels.

**Primary metric — IoU macro by deposit (test split):**
for each test deposit, IoU over the pooled scored pixels of its annotated scenes;
then the unweighted mean over test deposits. Deposits where IoU is undefined
(no water in reference and prediction) are excluded and their count reported.

**Secondary metrics** (always reported, never used to declare H1):
1. IoU micro by pixel (all test pixels pooled).
2. F1 macro by deposit and F1 micro by pixel; precision and recall (micro).
3. Pond area error per scene: absolute error (m²) and absolute percentage error;
   summarised per deposit by the median, then macro-averaged.
4. Minimum pond–crest distance error per scene (m), same summary.
5. Beach fraction error per scene (polygon eroded 30 m inwards minus pond).
6. Per-deposit and per-season tables; number of scenes where Otsu fell back to
   the fixed threshold.
7. Paired bootstrap 95% CI over deposits (10 000 resamples, fixed seed) for the
   macro differences U-Net − baseline.

## 5. Hypotheses and support criteria (fixed before the run)

Let Δ_IoU = IoU_macro(U-Net) − IoU_macro(baseline) on the test deposits.

- **H1 (primary) — segmentation.** The U-Net beats the baseline on unseen
  deposits. **Supported** iff Δ_IoU ≥ **0.05** AND the U-Net's per-deposit IoU is
  ≥ the baseline's in a **strict majority** of test deposits.
  If Δ_IoU ≤ −0.05 the baseline is declared better; anything else is
  "no meaningful difference". **[OPEN]** confirm the 0.05 margin.
- **H2 — pond area.** The U-Net measures pond area more accurately.
  **Supported** iff the macro (over deposits) median absolute percentage error of
  pond area is at least **20 % lower (relative)** for the U-Net AND lower in a
  strict majority of test deposits. **[OPEN]** confirm 20 %.
- **H3 — pond–crest distance.** The U-Net estimates the minimum pond–crest
  distance more accurately. **Supported** iff the macro median absolute
  distance error is at least **10 m (one pixel) lower** for the U-Net AND lower
  in a strict majority of test deposits with a crest line. Scenes where either
  the reference or the prediction has no water are reported separately.
  **[OPEN]** confirm 10 m.

The bootstrap CIs (§4.7) are reported next to each verdict but do not change it:
with 3–4 test deposits they are expected to be wide.

## 6. Annotation protocol

- Per val/test deposit: 8 scenes; seed `20260706`; candidate scenes with SCL
  valid fraction ≥ 0.9 inside the polygon.
- Mandatory first: the best scene of July 2026 and of August 2026, if any exists
  with valid fraction ≥ 0.8.
- Remaining slots: equal quotas per austral season (DJF, MAM, JJA, SON);
  within a season, years filled round-robin (least-used year first); random
  scene inside a (season, year) cell. Shortfalls filled from other seasons.
- Exported: RGB (B4-B3-B2), false colour (B12-B8-B4), empty mask template on the
  same grid. Not exported: MNDWI, baseline/model outputs, SCL.
- Masks must pass `scripts/validate_annotations.py --require-complete`
  (grid, values, ≥ 98 % of polygon pixels labelled) before the run.
- **[OPEN]** Who annotates; double annotation of a subset to measure
  inter-annotator IoU (proposal: 1 scene per deposit, reported as a ceiling).

## 7. v1 restrictions

- **Only conventional tailings dams** (tranques de relave: sand/cycloned-tailings
  dam with a supernatant pond). Thickened, paste and filtered deposits are out of
  scope. **[OPEN]** map this to the Sernageomin "tipo" field.
- **Static crest:** one crest line per deposit for 2019–2026; dam raises are
  ignored, so distances in early years are biased (direction depends on the
  raise geometry). Reported as a limitation, not corrected.
- 10 m resolution: ponds narrower than ~2–3 pixels are not reliably measurable.
- One MGRS tile per deposit; no cross-tile mosaics.
- No SAR, no other sensors, no in-situ data.

## 8. Single out-of-sample run

- The test deposits are scored **once** with
  `scripts/evaluate_test.py --i-understand-this-is-the-single-oos-run`.
- The script checks all preconditions, then atomically writes
  `reports/locks/test_evaluation.lock` **before** reading any test data; it refuses
  forever afterwards. The lock and `reports/test_results.json` are committed.
- The model checkpoint evaluated is fixed in advance (its SHA-256 is recorded in
  the lock).
- If a bug is found after the run, the original numbers stay in the report; any
  corrected rerun is labelled as a deviation and reported alongside, never
  instead.

## 9. Deviations

_None yet._
