import numpy as np
import torch

import fusion_cnn as FC
import six_classes as S


def test_engineered_columns_drop_loudness_only():
    names = list(S.engineered_features(np.random.default_rng(0).normal(size=S.SR * 5).astype(np.float32)))
    cols = FC.engineered_columns(names)
    assert len(cols) == 139 and not any(c.startswith("level_") for c in cols)


def test_fusion_net_shapes():
    model = FC.FusionNet(n_eng=139, n_classes=5, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 101, 64), torch.randn(2, 139))
    assert out.shape == (2, 5)
