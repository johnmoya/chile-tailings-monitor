"""Build data/catalog.parquet: fixed MGRS tile per deposit + per-scene SCL statistics.

Then write the scene selections (data/samples/train_scenes.parquet and
series_scenes.parquet) and print per-deposit counts.
Usage: ``uv run python scripts/build_catalog.py [--refresh]``
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from ctm.data.catalog import build_catalog
from ctm.data.deposits import load_deposits
from ctm.data.sampling import series_scenes, train_sample
from ctm.utils.config import load_config, resolve_path


def counts_table(catalog: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg["sampling"]
    g = catalog.groupby(["deposit_id", "split", "tile"])
    return pd.DataFrame({
        "total": g.size(),
        f"series_ge_{s['series_min_valid_fraction']}": g["valid_fraction"].apply(
            lambda v: int((v >= s["series_min_valid_fraction"]).sum())),
        f"clean_ge_{s['clean_min_valid_fraction']}": g["valid_fraction"].apply(
            lambda v: int((v >= s["clean_min_valid_fraction"]).sum())),
        "shadow_rule_fired": g["shadow_masked"].sum().astype(int),
        "first": g["date"].min().dt.date, "last": g["date"].max().dt.date,
    }).reset_index()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--refresh", action="store_true", help="ignore cached per-deposit parts")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    reg = load_deposits(resolve_path(cfg, "deposits"), allow_placeholders=False)

    catalog, tiles = build_catalog(reg, cfg, resolve_path(cfg, "catalog_parts"),
                                   refresh=args.refresh)
    out = resolve_path(cfg, "catalog")
    out.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out, index=False)
    tiles.to_parquet(resolve_path(cfg, "catalog_tiles"), index=False)

    s = cfg["sampling"]
    samples = resolve_path(cfg, "samples_dir")
    samples.mkdir(parents=True, exist_ok=True)
    train = train_sample(catalog, reg.ids_for("train"), s["train_per_month"],
                         s["clean_min_valid_fraction"], s["train_min_days_apart"])
    train.to_parquet(samples / "train_scenes.parquet", index=False)
    series = series_scenes(catalog, s["series_min_valid_fraction"])
    series.to_parquet(samples / "series_scenes.parquet", index=False)

    pd.set_option("display.width", 200)
    print(tiles.sort_values(["deposit_id", "n_scenes"], ascending=[True, False]).to_string(
        index=False))
    print()
    table = counts_table(catalog, cfg)
    table["train_sample"] = table["deposit_id"].map(train.groupby("deposit_id").size()).fillna(0)
    print(table.to_string(index=False))
    print(f"\ncatalog: {out} ({len(catalog)} rows)")
    print(f"train sample: {len(train)} scenes; series: {len(series)} scenes -> {samples}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
