"""Run this first on any new machine: is the GPU visible, and was the data copied completely?

  python src/check_setup.py            # check
  python src/check_setup.py --write    # (on the preparation laptop) record checksums of the data files
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR

CHUNK = 64 * 1024 * 1024


def sha256_file(path, start=0, length=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        left = length
        while True:
            block = f.read(CHUNK if left is None else min(CHUNK, left))
            if not block:
                break
            h.update(block)
            if left is not None:
                left -= len(block)
                if left <= 0:
                    break
    return h.hexdigest()


def fingerprint(data_dir: Path) -> dict:
    mel = data_dir / "mel.npy"
    size = mel.stat().st_size
    return {"clips.parquet": sha256_file(data_dir / "clips.parquet"),
            "mel.npy": {"size": size, "sha256_first_64MB": sha256_file(mel, 0, CHUNK),
                        "sha256_last_64MB": sha256_file(mel, max(0, size - CHUNK))}}


def device_smoke_test(dev) -> None:
    """One tiny forward + backward pass: catches a PyTorch build that can't run on this GPU."""
    import torch
    x = torch.randn(2, 1, 32, 32, device=dev, requires_grad=True)
    w = torch.randn(4, 1, 3, 3, device=dev, requires_grad=True)
    torch.nn.functional.conv2d(x, w).relu().mean().backward()
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--write", action="store_true", help="record checksums (preparation laptop only)")
    a = p.parse_args(argv)
    ok = True

    print(f"Python {platform.python_version()} on {platform.system()} {platform.release()}")
    try:
        import torch
        from device import auto_batch_size, gpu_memory_gb, pick_device
        dev = pick_device()
        print(f"PyTorch {torch.__version__} | device: {dev}", end="")
        if dev.type == "cuda":
            print(f" | GPU: {torch.cuda.get_device_name(0)}, {gpu_memory_gb(dev):.1f} GB", end="")
        print(f" | default batch size (efficientnet_b0): {auto_batch_size('efficientnet_b0', dev)}")
        if dev.type == "cpu":
            print("  note: no GPU found - training will work but be much slower (consider Kaggle).")
        try:
            device_smoke_test(dev)
            print(f"test calculation on {dev}: OK")
        except RuntimeError as e:
            print(f"PROBLEM: PyTorch cannot compute on {dev}: {e}\n"
                  f"  See README section 'GPU problems' (usually: install a PyTorch build for an older GPU).")
            ok = False
    except ImportError as e:
        print(f"PROBLEM: {e}. Run the setup script in setup/ first.")
        ok = False

    import data as D
    try:
        mel, table = D.open_features(a.data_dir)
        print(f"data: {len(table)} clips, spectrogram shape {mel.shape[1:]}, folder {a.data_dir}")
    except D.DataError as e:
        print(f"PROBLEM: {e}")
        return 1

    sums_path = a.data_dir / "checksums.json"
    if a.write:
        sums_path.write_text(json.dumps(fingerprint(a.data_dir), indent=2))
        print(f"wrote {sums_path}")
    elif not sums_path.exists():
        print("note: no checksums.json, so the copy can't be fully verified (only its structure was checked).")
    else:
        want, have = json.loads(sums_path.read_text()), fingerprint(a.data_dir)
        for name in ("clips.parquet", "mel.npy"):
            if want[name] != have[name]:
                print(f"PROBLEM: {name} differs from the original (incomplete or damaged copy). Copy {name} again.")
                ok = False
        if ok:
            print("data checksums match the original")
    print("OK - ready to train" if ok else "NOT READY - fix the problems above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
