"""SPEPA 工具模块测试。"""
import numpy as np
import pytest

from spepa.utils.config import SPEPAConfig, get_default_config
from spepa.utils.logger import get_logger, LoggedMixin, now_str, ensure_dir
from spepa.utils.metrics import (
    MetricsTracker, kl_divergence, gaussian_kl, prediction_error,
    mse, entropy, cosine_similarity,
)


def test_config_defaults():
    cfg = get_default_config()
    assert cfg.resources.cpu_max_percent == 20.0
    assert cfg.resources.memory_max_percent == 20.0
    assert cfg.resources.storage_keep_top_ratio == 0.30
    assert cfg.exploration.exploration_ratio_min == 0.30
    assert cfg.drives.safety == 1.0
    assert cfg.drives.gain == 1.0
    assert cfg.drives.efficiency == 0.5


def test_config_to_dict():
    cfg = SPEPAConfig()
    d = cfg.to_dict()
    assert "resources" in d
    assert "exploration" in d
    assert "drives" in d


def test_logger():
    log = get_logger("test_spepa")
    assert log is not None
    log.info("test message")


def test_logged_mixin():
    class Dummy(LoggedMixin):
        def __init__(self):
            self._set_log_name("test.dummy")
        def run(self):
            self.log.info("running")
            return 42
    d = Dummy()
    assert d.run() == 42


def test_now_str():
    s = now_str()
    assert isinstance(s, str)
    assert len(s) > 0


def test_ensure_dir(tmp_path):
    p = ensure_dir(str(tmp_path / "subdir"))
    import os
    assert os.path.isdir(p)


def test_kl_divergence():
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert kl_divergence(p, q) < 1e-6
    q2 = np.array([0.1, 0.9])
    assert kl_divergence(p, q2) > 0


def test_gaussian_kl():
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([1.0, 1.0])
    var1 = np.array([1.0, 1.0])
    var2 = np.array([1.0, 1.0])
    kl = gaussian_kl(mu1, mu2, var1, var2)
    assert kl > 0
    # 相同分布 KL=0
    assert gaussian_kl(mu1, mu1, var1, var1) < 1e-6


def test_prediction_error():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0])
    assert prediction_error(a, b) < 1e-6
    c = np.array([2.0, 3.0])
    assert prediction_error(a, c) > 0


def test_mse():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0])
    assert mse(a, b) < 1e-6


def test_entropy():
    p = np.array([0.5, 0.5])
    e = entropy(p)
    assert e > 0
    p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    assert entropy(p_uniform) > entropy(np.array([0.9, 0.1]))


def test_cosine_similarity():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6
    c = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, c)) < 1e-6


def test_metrics_tracker():
    mt = MetricsTracker()
    mt.record("loss", 1.0)
    mt.record("loss", 0.5)
    mt.record("loss", 0.25)
    summary = mt.summary()
    assert "loss" in summary
    assert summary["loss"]["count"] == 3
    assert summary["loss"]["last"] == 0.25
    mt.reset()
    assert mt.summary() == {}
