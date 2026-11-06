"""Deposit registry (configs/deposits.geojson): schema validation and split access.

Anti-leakage rule: the split is a property of the DEPOSIT. Every tile, scene,
weak label and reference mask inherits the split of its deposit; nothing is
ever split by tile. Test deposits are only returned when explicitly requested
with ``allow_test=True``, which only scripts/evaluate_test.py does.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

SPLITS = ("train", "val", "test")
REQUIRED = ("id", "nombre", "region", "split")
_ID_RE = re.compile(r"^[a-z0-9_-]+$")


class DepositSchemaError(ValueError):
    pass


class TestSplitLockedError(PermissionError):
    """Raised when code outside the single OOS evaluation asks for test deposits."""

    __test__ = False  # not a pytest test class despite the name


@dataclass(frozen=True)
class DepositRegistry:
    gdf: gpd.GeoDataFrame

    @property
    def ids(self) -> list[str]:
        return list(self.gdf["id"])

    def ids_for(self, split: str, allow_test: bool = False) -> list[str]:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        if split == "test" and not allow_test:
            raise TestSplitLockedError(
                "test deposits are locked; only scripts/evaluate_test.py may read them"
            )
        return list(self.gdf.loc[self.gdf["split"] == split, "id"])

    def split_of(self, deposit_id: str) -> str:
        row = self.gdf.loc[self.gdf["id"] == deposit_id]
        if row.empty:
            raise KeyError(deposit_id)
        return str(row["split"].iloc[0])

    def geometry(self, deposit_id: str):
        row = self.gdf.loc[self.gdf["id"] == deposit_id]
        if row.empty:
            raise KeyError(deposit_id)
        return row.geometry.iloc[0]

    def real(self) -> DepositRegistry:
        """Registry without TODO placeholders."""
        return DepositRegistry(self.gdf.loc[~self.gdf["todo"]].reset_index(drop=True))


def validate(gdf: gpd.GeoDataFrame, allow_placeholders: bool = True) -> None:
    missing = [c for c in REQUIRED if c not in gdf.columns]
    if missing:
        raise DepositSchemaError(f"missing required properties: {missing}")
    errors: list[str] = []
    dup = gdf["id"][gdf["id"].duplicated()].tolist()
    if dup:
        errors.append(f"duplicate ids: {dup}")
    for i, row in gdf.iterrows():
        rid = row["id"]
        if not isinstance(rid, str) or not _ID_RE.match(rid):
            errors.append(f"row {i}: invalid id {rid!r} (use [a-z0-9_-]+)")
        if row["split"] not in SPLITS:
            errors.append(f"row {i} ({rid}): split must be one of {SPLITS}, got {row['split']!r}")
        for col in ("nombre", "region"):
            if not isinstance(row[col], str) or not row[col].strip():
                errors.append(f"row {i} ({rid}): empty {col}")
        if row["todo"]:
            if not allow_placeholders:
                errors.append(f"row {i} ({rid}): TODO placeholder present")
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            errors.append(f"row {i} ({rid}): missing geometry")
        elif geom.geom_type not in ("Polygon", "MultiPolygon"):
            errors.append(f"row {i} ({rid}): geometry must be (Multi)Polygon, got {geom.geom_type}")
        elif not geom.is_valid:
            errors.append(f"row {i} ({rid}): invalid geometry")
    if errors:
        raise DepositSchemaError("; ".join(errors))


def load_deposits(path: str | Path, allow_placeholders: bool = True) -> DepositRegistry:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        raise DepositSchemaError(f"deposits must be in WGS84, got {gdf.crs}")
    if "todo" not in gdf.columns:
        gdf["todo"] = False
    gdf["todo"] = [bool(v) if v is not None and v == v else False for v in gdf["todo"]]
    validate(gdf, allow_placeholders=allow_placeholders)
    return DepositRegistry(gdf)


def assert_disjoint_splits(deposit_ids_by_split: dict[str, Iterable[str]]) -> None:
    """Fail if any deposit id appears in more than one split."""
    seen: dict[str, str] = {}
    for split, ids in deposit_ids_by_split.items():
        for did in ids:
            if did in seen and seen[did] != split:
                raise AssertionError(f"leakage: deposit {did!r} in both {seen[did]} and {split}")
            seen[did] = split
