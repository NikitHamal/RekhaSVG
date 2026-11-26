"""
Pytest verification of Rekha-Net, Rekha-Text, topology-preserving ClDice loss,
and INT8 ONNX export & size/latency budget compliance.
"""
import os
import tempfile
import torch

from rekhasvg.rekha_net import RekhaNet, RekhaNetLoss
from rekhasvg.rekha_text import REKHA_CHARSET, RekhaText, encode_text_targets, greedy_ctc_decode
from scripts.train_rekha_net import run_smoke_test as smoke_net
from scripts.train_rekha_text import run_smoke_test as smoke_txt
from scripts.export_onnx import export_and_verify


def test_rekha_net_smoke_and_cldice():
    res = smoke_net()
    assert res["steps"] == 2
    assert not res["nan_detected"]
    assert 5_000_000 <= res["params"] <= 7_600_000


def test_rekha_text_smoke_and_devanagari_vocab():
    res = smoke_txt()
    assert res["steps"] == 2
    assert not res["nan_detected"]
    assert 3_500_000 <= res["params"] <= 4_600_000
    for ch in ("उ", "त", "ल", "Ω", "θ", "Δ", "₂"):
        assert ch in REKHA_CHARSET


def test_int8_onnx_export_budget_and_latency():
    with tempfile.TemporaryDirectory() as tmpdir:
        status = export_and_verify(out_dir=tmpdir)
        assert status["rekha_net_int8_mb"] <= 7.5
        assert status["rekha_text_int8_mb"] <= 4.5
        assert 10.0 <= status["total_int8_mb"] <= 15.0
        assert status["conv_integer_ops"] == 0
        assert status["nan_count"] == 0
        assert status["cpu_latency_ms"] < 600.0
