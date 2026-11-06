"""Mask diagnostics from the catalog's per-scene SCL class fractions.

Two products:
  * monthly table per deposit: mean fraction of polygon pixels MASKED by each SCL
    class, by calendar month (all years pooled). Class 3 only counts as masked in
    scenes where the shadow rule fired.
  * bright-surface flag: in scenes with LOW tile-level cloud cover, classes 8/9/10/11
    covering more than a threshold of the polygon are suspicious: bright, dry tailings
    (or salt crusts) misread as cloud or snow. A deposit is flagged when the share of
    such scenes among its low-cloud scenes reaches ``flag_min_share``.
"""

from __future__ import annotations

import pandas as pd

BRIGHT = [8, 9, 10, 11]
CLASS_NAMES = {0: "no data", 1: "saturated", 2: "dark area", 3: "cloud shadow",
               4: "vegetation", 5: "not vegetated", 6: "water", 7: "unclassified",
               8: "cloud medium", 9: "cloud high", 10: "thin cirrus", 11: "snow/ice"}


def masked_fractions(catalog: pd.DataFrame, invalid_classes: list[int],
                     shadow_class: int) -> pd.DataFrame:
    """Per scene, fraction of polygon masked by each class (0 for classes kept)."""
    out = catalog[["deposit_id", "date", "scene_cloud_cover"]].copy()
    for k in sorted(set(invalid_classes) | {shadow_class}):
        col = catalog[f"scl_{k}"]
        if k == shadow_class and k not in invalid_classes:
            col = col.where(catalog["shadow_masked"], 0.0)
        out[f"masked_{k}"] = col
    out["masked_total"] = 1.0 - catalog["valid_fraction"]
    if shadow_class not in invalid_classes:
        out["shadow_kept"] = catalog[f"scl_{shadow_class}"].where(~catalog["shadow_masked"], 0.0)
    return out


def monthly_table(masked: pd.DataFrame) -> pd.DataFrame:
    m = masked.assign(month=masked["date"].dt.month)
    cols = [c for c in m.columns if c.startswith("masked_") or c == "shadow_kept"]
    agg = m.groupby(["deposit_id", "month"])[cols].mean()
    agg["n_scenes"] = m.groupby(["deposit_id", "month"]).size()
    return agg.reset_index()


def bright_flags(catalog: pd.DataFrame, low_cloud_max: float, class_threshold: float,
                 flag_min_share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-deposit summary, offending scenes)."""
    low = catalog[catalog["scene_cloud_cover"] <= low_cloud_max].copy()
    bright = low[[f"scl_{k}" for k in BRIGHT]]
    low["max_bright"] = bright.max(axis=1)
    low["bright_total"] = bright.sum(axis=1)
    low["dominant"] = bright.idxmax(axis=1).str.removeprefix("scl_").astype(int)
    low["hit"] = low["max_bright"] > class_threshold
    rows = []
    for did in sorted(catalog["deposit_id"].unique()):
        d = low[low["deposit_id"] == did]
        hits = d[d["hit"]]
        share = len(hits) / len(d) if len(d) else float("nan")
        dom = hits["dominant"].mode()
        rows.append({
            "deposit_id": did, "n_scenes": int((catalog["deposit_id"] == did).sum()),
            "n_low_cloud": len(d), "n_hits": len(hits), "share_hits": share,
            "dominant_class": int(dom.iloc[0]) if len(dom) else None,
            "median_bright_in_hits": float(hits["bright_total"].median()) if len(hits) else None,
            "flagged": bool(len(d) and share >= flag_min_share),
        })
    scenes = low[low["hit"]][["deposit_id", "scene_id", "date", "scene_cloud_cover",
                              "max_bright", "bright_total", "dominant"]]
    return pd.DataFrame(rows), scenes.sort_values(["deposit_id", "date"])


def _pct(x) -> str:
    return "" if x is None or x != x else f"{100 * x:.1f}"


def render_markdown(catalog: pd.DataFrame, cfg: dict) -> str:
    s, d = cfg["stac"], cfg["diagnostics"]
    masked = masked_fractions(catalog, s["scl_invalid_classes"], s["scl_shadow_class"])
    monthly = monthly_table(masked)
    summary, hits = bright_flags(catalog, d["low_cloud_max"], d["bright_class_threshold"],
                                 d["flag_min_share"])
    mcols = [c for c in monthly.columns if c.startswith("masked_") and c != "masked_total"]
    lines = [
        "# M1 — Mask diagnostics",
        "",
        f"Generated from `{cfg['paths']['catalog']}` ({len(catalog)} scenes, "
        f"{catalog['deposit_id'].nunique()} deposits). Fractions are of polygon pixels "
        "(SCL at 20 m), averaged over scenes of each calendar month, all years pooled.",
        f"Always-masked classes: {s['scl_invalid_classes']}; class {s['scl_shadow_class']} "
        f"masked only with clouds {s['scl_shadow_cloud_classes']} within "
        f"{s['shadow_cloud_radius_m']} m of the bbox. `shadow kept` = class 3 left as valid.",
        "",
        "## Bright-surface check (possible tailings/salt read as cloud or snow)",
        "",
        f"Low-cloud scene: tile `eo:cloud_cover` <= {d['low_cloud_max']}%. Hit: any of classes "
        f"8/9/10/11 > {100 * d['bright_class_threshold']:.0f}% of the polygon. Deposit flagged "
        f"when hits >= {100 * d['flag_min_share']:.0f}% of its low-cloud scenes.",
        "",
        "| deposit | scenes | low-cloud | hits | % hits | dominant class | median 8–11 in hits (%) "
        "| FLAG |",
        "|---|---:|---:|---:|---:|---|---:|:---:|",
    ]
    for r in summary.itertuples():
        has_dom = r.dominant_class is not None and r.dominant_class == r.dominant_class
        dom = f"{int(r.dominant_class)} ({CLASS_NAMES[int(r.dominant_class)]})" if has_dom else ""
        lines.append(f"| {r.deposit_id} | {r.n_scenes} | {r.n_low_cloud} | {r.n_hits} | "
                     f"{_pct(r.share_hits)} | {dom} | {_pct(r.median_bright_in_hits)} | "
                     f"{'**YES**' if r.flagged else ''} |")
    lines += ["", "## Monthly masked fraction by class (%)", ""]
    header = ["month", "n"] + [f"{c.removeprefix('masked_')}" for c in mcols] + \
        (["shadow kept"] if "shadow_kept" in monthly else []) + ["total masked"]
    for did, g in monthly.groupby("deposit_id"):
        lines += [f"### {did}", "", "| " + " | ".join(header) + " |",
                  "|" + "---:|" * len(header)]
        for r in g.itertuples(index=False):
            rd = r._asdict()
            vals = [str(rd["month"]), str(rd["n_scenes"])] + [_pct(rd[c]) for c in mcols]
            if "shadow_kept" in rd:
                vals.append(_pct(rd["shadow_kept"]))
            vals.append(_pct(rd["masked_total"]))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    if len(hits):
        lines += ["## Offending low-cloud scenes", "",
                  "| deposit | scene | tile cloud % | max class (%) | 8–11 total (%) | class |",
                  "|---|---|---:|---:|---:|---|"]
        for r in hits.itertuples():
            lines.append(f"| {r.deposit_id} | {r.scene_id} | {r.scene_cloud_cover:.1f} | "
                         f"{_pct(r.max_bright)} | {_pct(r.bright_total)} | {r.dominant} |")
    return "\n".join(lines) + "\n"
