"""Where do the distance / speed errors of the existing gradient-boosting runs come from? (no new model)

  python src/distance_error_check.py > runs/distance_error_check.txt
"""
import numpy as np
import pandas as pd

t = pd.read_parquet(r"D:\shipsound\data\clips.parquet")

for name, path, unit in (("DISTANCE", r"D:\shipsound\runs\distance_classical\predictions.parquet", "m"),
                         ("SPEED", r"D:\shipsound\runs\speed_classical_4kn\predictions.parquet", "kn")):
    p = pd.read_parquet(path)
    p = p.join(t[["hydrophone", "interval_mean_distance", "mean_distance", "nav_status", "mean_sog", "length"]], on="row")
    p["res"] = p.reg_m - p.true_m
    p["err"] = p.res.abs()
    print(f"\n=================== {name}: {len(p)} test clips, MAE {p.err.mean():.2f} {unit}")

    # 1. regression to the middle: signed error by true value
    edges = [0, 400, 800, 1200, 1600, 2000, 99999] if unit == "m" else [0, 4, 8, 12, 16, 99]
    p["true_bin"] = pd.cut(p.true_m, edges, right=False)
    print("signed error (pred - true) by true value:")
    print(p.groupby("true_bin", observed=True).agg(clips=("res", "size"), mean_signed=("res", "mean"),
                                                   mae=("err", "mean")).round(1).to_string())

    # 2. do errors stick to ships? share of squared error explained by each ship's average error
    q = p[p.groupby("ship").res.transform("size") >= 3].copy()
    q["ship_mean"] = q.groupby("ship").res.transform("mean")
    share = (q.ship_mean ** 2).sum() / (q.res ** 2).sum()
    print(f"ships with >= 3 test clips: {q.ship.nunique()} ships, {len(q)} clips; "
          f"share of squared error that is the ship's own constant offset: {share:.2f}")
    print(f"  MAE {q.err.mean():.2f} -> {(q.res - q.ship_mean).abs().mean():.2f} {unit} if each ship's offset were known")

    # 3. hydrophone and ship type
    print("MAE by hydrophone:", p.groupby("hydrophone").err.mean().round(2).to_dict())
    print("MAE by ship type:", p.groupby("label").err.mean().round(2).to_dict())

    # 4. labels: how far the clip's distance is from the passage's mean distance; AIS status
    if unit == "m":
        p["label_spread"] = (p.mean_distance - p.interval_mean_distance).abs()
        p["spread_q"] = pd.qcut(p.label_spread, 4)
        print("MAE by |clip distance - interval mean distance| (quartiles):")
        print(p.groupby("spread_q", observed=True).err.mean().round(1).to_string())
    else:
        status = {0: "under way, engine", 5: "moored", 7: "fishing", 15: "not defined", 1: "at anchor", 8: "sailing", 11: "towing"}
        p["status"] = p.nav_status.map(status).fillna("other / missing")
        print("MAE by AIS navigation status:")
        print(p.groupby("status").agg(clips=("err", "size"), mae=("err", "mean")).round(2).to_string())
        print(f"clips under 2 kn (nearly stopped): {(p.true_m < 2).mean():.3f}, MAE there {p.err[p.true_m < 2].mean():.2f}")
    # 5. ship length (how big / how loud the ship is) and the error
    p["len_q"] = pd.qcut(p.length.where(p.length > 0), 4)
    print("signed error by ship length quartile:")
    print(p.groupby("len_q", observed=True).agg(mean_signed=("res", "mean"), mae=("err", "mean")).round(1).to_string())
