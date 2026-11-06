from ctm.data.diagnostics import bright_flags, masked_fractions, monthly_table, render_markdown
from tests.helpers import fake_catalog


def _catalog_with_bright_deposit():
    cat = fake_catalog(["dep-ok", "dep-bright"], "2021-01-01", "2021-12-31", every=5, seed=7)
    bright = (cat["deposit_id"] == "dep-bright") & (cat["scene_cloud_cover"] <= 10)
    # Bright dry tailings read as high-probability cloud in clear scenes.
    cat.loc[bright, "scl_9"] = 0.25
    cat.loc[bright, "scl_5"] = 0.45
    cat.loc[bright, "valid_fraction"] = 0.75
    clear_ok = (cat["deposit_id"] == "dep-ok") & (cat["scene_cloud_cover"] <= 10)
    cat.loc[clear_ok, ["scl_8", "scl_9", "scl_10", "scl_11"]] = 0.0
    return cat


def test_bright_flag_marks_only_the_bright_deposit(cfg):
    d = cfg["diagnostics"]
    summary, hits = bright_flags(_catalog_with_bright_deposit(), d["low_cloud_max"],
                                 d["bright_class_threshold"], d["flag_min_share"])
    s = summary.set_index("deposit_id")
    assert bool(s.loc["dep-bright", "flagged"]) and not bool(s.loc["dep-ok", "flagged"])
    assert s.loc["dep-bright", "dominant_class"] == 9
    assert set(hits["deposit_id"]) == {"dep-bright"}


def test_shadow_counts_as_masked_only_when_rule_fired(cfg):
    cat = fake_catalog(["d"], "2021-01-01", "2021-01-10", seed=1)
    cat["scl_3"] = 0.2
    cat["shadow_masked"] = [True] + [False] * (len(cat) - 1)
    s = cfg["stac"]
    m = masked_fractions(cat, s["scl_invalid_classes"], s["scl_shadow_class"])
    assert m["masked_3"].tolist()[0] == 0.2 and (m["masked_3"].iloc[1:] == 0).all()
    assert (m["shadow_kept"].iloc[1:] == 0.2).all()
    t = monthly_table(m)
    assert t["n_scenes"].sum() == len(cat)


def test_markdown_renders(cfg):
    md = render_markdown(_catalog_with_bright_deposit(), cfg)
    assert md.startswith("# M1")
    assert "dep-bright" in md and "**YES**" in md
    assert "### dep-ok" in md
