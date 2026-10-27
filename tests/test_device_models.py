import numpy as np
import pytest
import torch

import device
import models
from labels import CLASSES


def test_device_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(device.shutil, "which", lambda name: None)
    assert device.gpu_temperature() is None
    assert device.TemperatureGuard(check_every=1).step() == 0.0
    assert device.auto_batch_size("efficientnet_b0", torch.device("cpu")) == 16
    assert device.pick_device("cpu") == torch.device("cpu")


def test_auto_batch_size_cuda_table(monkeypatch):
    cuda = torch.device("cuda")
    monkeypatch.setattr(device, "gpu_memory_gb", lambda d: 4.0)
    assert device.auto_batch_size("efficientnet_b0", cuda) == 32
    assert device.auto_batch_size("tiny_cnn", cuda) == 64
    monkeypatch.setattr(device, "gpu_memory_gb", lambda d: 16.0)
    assert device.auto_batch_size("resnet18", cuda) == 192
    assert device.auto_batch_size("efficientnet_b0", torch.device("mps")) == 32


def test_shipnet_tiny_forward():
    net = models.ShipNet("tiny_cnn", n_classes=12, pretrained=False)
    out = net(torch.randn(4, 1, 128, 64))
    assert out["logits"].shape == (4, 12)
    assert out["distance"].shape == (4,) and out["speed"].shape == (4,)
    assert out["features"].shape == (4, models.TinyCNN().num_features)


def test_shipnet_variable_frames():
    net = models.ShipNet("tiny_cnn", n_classes=9, pretrained=False).eval()
    assert net(torch.randn(2, 1, 128, 501))["logits"].shape == (2, 9)


@pytest.mark.parametrize("backbone", models.BACKBONES)
def test_shipnet_every_backbone_offline(backbone):
    net = models.ShipNet(backbone, n_classes=12, pretrained=False).eval()
    out = net(torch.randn(2, 1, 128, 64))
    assert out["logits"].shape == (2, 12) and out["distance"].shape == (2,)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = models.ShipNet("tiny_cnn", n_classes=12, pretrained=False, drop_rate=0.2).eval()
    mean, std = np.zeros(128, np.float32), np.ones(128, np.float32)
    models.save_checkpoint(tmp_path / "best.pt", model=net.state_dict(),
                           config={"backbone": "tiny_cnn", "drop_rate": 0.2, "task": "class"},
                           known=CLASSES, norm={"mean": mean.tolist(), "std": std.tolist()})
    assert not list(tmp_path.glob("*.tmp"))
    loaded, info = models.load_model(tmp_path)
    x = torch.randn(3, 1, 128, 64)
    with torch.no_grad():
        assert torch.allclose(net(x)["logits"], loaded(x)["logits"])
    assert info["known"] == CLASSES and info["config"]["backbone"] == "tiny_cnn"
    assert np.allclose(info["norm"][1], 1.0)


def test_pretrained_download_failure_message(monkeypatch):
    def fail(*args, pretrained=False, **kwargs):
        if pretrained:
            raise OSError("could not reach huggingface.co")
        raise AssertionError("should not get here")
    monkeypatch.setattr(models.timm, "create_model", fail)
    with pytest.raises(models.PretrainedDownloadError, match="--no-pretrained"):
        models.ShipNet("efficientnet_b0", n_classes=12, pretrained=True)


def test_unknown_backbone_rejected():
    with pytest.raises(ValueError, match="backbone"):
        models.ShipNet("vgg_huge", n_classes=12, pretrained=False)


def test_temperature_guard_low_limit_still_pauses(monkeypatch):
    temps = iter([72, 72, 72, 59])
    monkeypatch.setattr(device, "gpu_temperature", lambda: next(temps))
    monkeypatch.setattr(device.time, "sleep", lambda s: None)
    g = device.TemperatureGuard(max_c=70, check_every=1)      # default resume 75 would never wait
    assert g.resume_c == 60
    g.step()
    with pytest.raises(StopIteration):
        next(temps)                                          # it kept polling until the GPU was below 60 °C


def test_pretrained_error_hides_http_traceback(monkeypatch):
    def fail(*args, pretrained=False, **kwargs):
        raise OSError("could not reach huggingface.co")
    monkeypatch.setattr(models.timm, "create_model", fail)
    with pytest.raises(models.PretrainedDownloadError) as e:
        models.ShipNet("efficientnet_b0", n_classes=12, pretrained=True)
    assert e.value.__cause__ is None and e.value.__suppress_context__
