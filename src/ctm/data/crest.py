"""Crest (coronamiento) lines: configs/crest_lines.geojson, one LineString per deposit.

v1 assumption: the crest is STATIC over 2019–2026 (dam raises are ignored).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

from ctm.data.deposits import DepositRegistry, DepositSchemaError


@dataclass(frozen=True)
class CrestLines:
    gdf: gpd.GeoDataFrame

    @property
    def deposit_ids(self) -> list[str]:
        return list(self.gdf["deposit_id"])

    def line(self, deposit_id: str):
        row = self.gdf.loc[self.gdf["deposit_id"] == deposit_id]
        if row.empty:
            raise KeyError(f"no crest line for {deposit_id}")
        return row.geometry.iloc[0]


def load_crests(path: str | Path, registry: DepositRegistry | None = None) -> CrestLines:
    gdf = gpd.read_file(path)
    if gdf.empty:
        return CrestLines(gpd.GeoDataFrame({"deposit_id": []}, geometry=[], crs="EPSG:4326"))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        raise DepositSchemaError(f"crest lines must be in WGS84, got {gdf.crs}")
    if "deposit_id" not in gdf.columns:
        raise DepositSchemaError("crest lines need a deposit_id property")
    errors = []
    dup = gdf["deposit_id"][gdf["deposit_id"].duplicated()].tolist()
    if dup:
        errors.append(f"more than one crest feature for {dup} (merge into a MultiLineString)")
    known = set(registry.ids) if registry is not None else None
    for i, row in gdf.iterrows():
        g = row.geometry
        if known is not None and row["deposit_id"] not in known:
            errors.append(f"row {i}: unknown deposit_id {row['deposit_id']!r}")
        if g is None or g.is_empty:
            errors.append(f"row {i}: missing geometry")
        elif g.geom_type not in ("LineString", "MultiLineString"):
            errors.append(f"row {i}: geometry must be (Multi)LineString, got {g.geom_type}")
        elif not g.is_valid:
            errors.append(f"row {i}: invalid geometry")
    if errors:
        raise DepositSchemaError("; ".join(errors))
    return CrestLines(gdf)
