"""Scene selection from the catalog: train sample, time series, annotation set.

All selections are deterministic: ordering rules plus a seeded RNG that is
derived from (seed, deposit_id) so adding a deposit never reshuffles others.
"""

from __future__ import annotations

import zlib
from collections import Counter

import numpy as np
import pandas as pd


def season_of(month: int, seasons: dict[str, list[int]]) -> str:
    for name, months in seasons.items():
        if month in months:
            return name
    raise ValueError(f"month {month} not in any season")


def deposit_rng(seed: int, deposit_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(deposit_id.encode())])


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    """Best first: highest valid fraction, then lowest tile cloud cover, then earliest."""
    return df.sort_values(["valid_fraction", "scene_cloud_cover", "date"],
                          ascending=[False, True, True])


def series_scenes(catalog: pd.DataFrame, min_valid: float) -> pd.DataFrame:
    return catalog[catalog["valid_fraction"] >= min_valid].sort_values(["deposit_id", "date"])


def train_sample(catalog: pd.DataFrame, deposit_ids: list[str], per_month: int,
                 min_valid: float, min_days_apart: int) -> pd.DataFrame:
    """Up to ``per_month`` clean scenes per deposit and calendar month (year-month)."""
    df = catalog[catalog["deposit_id"].isin(deposit_ids)
                 & (catalog["valid_fraction"] >= min_valid)].copy()
    if df.empty:
        return df
    df["ym"] = df["date"].dt.to_period("M")
    picked = []
    for _, grp in df.groupby(["deposit_id", "ym"], sort=True):
        chosen: list[pd.Timestamp] = []
        for idx, row in _rank(grp).iterrows():
            if len(chosen) >= per_month:
                break
            if all(abs((row["date"] - d).days) >= min_days_apart for d in chosen):
                chosen.append(row["date"])
                picked.append(idx)
    return df.loc[picked].drop(columns="ym").sort_values(["deposit_id", "date"])


def annotation_selection(cat_dep: pd.DataFrame, n: int, seed: int, deposit_id: str,
                         seasons: dict[str, list[int]], min_valid: float,
                         mandatory: list[dict], mandatory_min_valid: float) -> pd.DataFrame:
    """``n`` scenes for one deposit, stratified by season and year.

    1. Mandatory windows: best scene (valid >= ``mandatory_min_valid``) in each window.
    2. Seasons get equal quotas (n / #seasons, remainder to the first ones); mandatory
       scenes count towards their season's quota.
    3. Within a season, years are filled round-robin, least-used year first (ties
       broken by the seeded RNG); the scene inside a (season, year) cell is random.
    4. If a season lacks candidates, the leftover slots are filled from any season,
       again preferring the least-used years.
    """
    rng = deposit_rng(seed, deposit_id)
    df = cat_dep.copy()
    df["season"] = df["date"].dt.month.map(lambda m: season_of(m, seasons))
    df["year"] = df["date"].dt.year
    selected: list = []
    reason: dict = {}

    for win in mandatory:
        lo, hi = pd.Timestamp(win["start"]), pd.Timestamp(win["end"])
        cand = df[(df["date"] >= lo) & (df["date"] <= hi)
                  & (df["valid_fraction"] >= mandatory_min_valid)
                  & ~df.index.isin(selected)]
        if len(cand) and len(selected) < n:
            idx = _rank(cand).index[0]
            selected.append(idx)
            reason[idx] = f"mandatory {win['start']}..{win['end']}"

    pool = df[(df["valid_fraction"] >= min_valid)]
    names = list(seasons)
    base, extra = divmod(n, len(names))
    quota = {s: base + (1 if i < extra else 0) for i, s in enumerate(names)}

    def pick(cands: pd.DataFrame, why: str) -> bool:
        cands = cands[~cands.index.isin(selected)]
        if cands.empty:
            return False
        used = Counter(df.loc[selected, "year"])
        years = sorted(cands["year"].unique())
        order = rng.permutation(len(years))
        year = min((years[i] for i in order), key=lambda y: used.get(y, 0))
        cell = cands[cands["year"] == year]
        idx = cell.index[int(rng.integers(len(cell)))]
        selected.append(idx)
        reason[idx] = why
        return True

    for s in names:
        while (df.loc[selected, "season"] == s).sum() < quota[s] and len(selected) < n:
            if not pick(pool[pool["season"] == s], f"stratified {s}"):
                break
    while len(selected) < n and pick(pool, "fill (season short of candidates)"):
        pass

    out = df.loc[selected].copy()
    out["selection_reason"] = [reason[i] for i in selected]
    return out.sort_values("date")
