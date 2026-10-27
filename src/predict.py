"""New recording -> ship class (or "unknown"), confidence, distance and speed, per 5-second window.

  python src/predict.py recording.wav --class-run runs/class_f1_efficientnet_b0_vessel \
      --distance-run runs/distance_f1_efficientnet_b0_vessel --speed-run runs/speed_f1_efficientnet_b0_vessel

Any sample rate, mono or stereo. Run evaluate.py on the class run first: it stores the temperature and
the "unknown" threshold this script uses.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
import reliability as R
from models import load_model

MIN_SECONDS = 2.5
CHUNK = 32          # windows processed at once; keeps memory small for hour-long recordings


def windows(wave):
    """5-second windows with a 5-second hop; a last piece of >= 2.5 s is padded, a shorter one dropped."""
    n, out = F.CLIP_SAMPLES, []
    for start in range(0, len(wave), n):
        seg = wave[start:start + n]
        if len(seg) < MIN_SECONDS * F.SR:
            break
        out.append((start / F.SR, (start + len(seg)) / F.SR, F.fix_length(seg)))
    return out


def load_calibration(class_run: Path, score: str):
    path = Path(class_run) / "eval" / "calibration.json"
    if not path.exists():
        print(f"warning: {path} not found - run  python src/evaluate.py {class_run}  first. "
              f"Using temperature 1 and an MSP threshold of 0.5 for 'unknown'.")
        return 1.0, "msp", 0.5
    cal = json.loads(path.read_text())
    return cal["temperature"], score, cal[f"{score}_threshold"]


@torch.no_grad()
def _run(model, norm, mels):
    x = (mels - torch.from_numpy(norm[0])[:, None]) / torch.from_numpy(norm[1])[:, None]
    return model(x.unsqueeze(1))


def predict_file(wav_path: Path, class_run: Path, distance_run: Path | None = None,
                 speed_run: Path | None = None, score: str = "energy") -> pd.DataFrame:
    wave, sr, _ = F.decode_wav(str(wav_path))
    wave = F.resample(wave, sr)
    if len(wave) < MIN_SECONDS * F.SR:
        raise ValueError(f"need at least {MIN_SECONDS} s of audio, got {len(wave) / F.SR:.1f} s")
    wins = windows(wave)
    model, info = load_model(class_run)
    T, score, threshold = load_calibration(class_run, score)
    regressors = {head: load_model(run) for head, run in (("distance", distance_run), ("speed", speed_run)) if run}
    logmel, logits, reg = F.LogMel(), [], {head: [] for head in regressors}
    for s in range(0, len(wins), CHUNK):
        mels = logmel(torch.from_numpy(np.stack([w[2] for w in wins[s:s + CHUNK]])))
        logits.append(_run(model, info["norm"], mels)["logits"].numpy())
        for head, (m, inf) in regressors.items():
            reg[head].append(_run(m, inf["norm"], mels)[head].numpy())
    logits = np.concatenate(logits)
    probs = R.softmax(logits, T)
    s = R.energy_score(logits, T) if score == "energy" else R.msp_score(logits, T)
    df = pd.DataFrame({"start_s": [w[0] for w in wins], "end_s": [w[1] for w in wins],
                       "class": [info["known"][i] for i in probs.argmax(1)],
                       "confidence": probs.max(1), "unknown": s > threshold})
    ship = (~df.unknown) & (df["class"] != "background")
    for col, head, scale in (("distance_m", "distance", 1000.0), ("speed_kn", "speed", 10.0)):
        df[col] = np.nan
        if head in reg:
            df.loc[ship, col] = np.concatenate(reg[head])[ship.to_numpy()] * scale

    known = df[~df.unknown]
    overall = {"start_s": "overall", "end_s": df.end_s.max(),
               "class": known["class"].mode().iat[0] if len(known) else "unknown",
               "confidence": float(df.confidence.mean()), "unknown": bool(df.unknown.mean() > 0.5),
               "distance_m": df.distance_m[ship].median() if ship.any() else np.nan,
               "speed_kn": df.speed_kn[ship].median() if ship.any() else np.nan}
    df["start_s"] = df.start_s.astype(object)
    return pd.concat([df, pd.DataFrame([overall])], ignore_index=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", type=Path)
    p.add_argument("--class-run", type=Path, required=True)
    p.add_argument("--distance-run", type=Path)
    p.add_argument("--speed-run", type=Path)
    p.add_argument("--score", choices=["energy", "msp"], default="energy", help="how 'unknown' is decided")
    a = p.parse_args(argv)
    df = predict_file(a.wav, a.class_run, a.distance_run, a.speed_run, a.score)
    with pd.option_context("display.max_rows", 200, "display.width", 120, "display.float_format", "{:.2f}".format):
        print(df.to_string(index=False))
    print("\nNote: distance is learned from loudness on ONC icListen hydrophones; "
          "recordings from other microphones need calibration before trusting it.")


if __name__ == "__main__":
    main()
