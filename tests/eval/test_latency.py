"""Tests for latency measurement tool."""

import inspect

import pytest


class TestLatencyImport:
    """インポートと関数シグネチャのテスト"""

    def test_import_compute_latency(self):
        """compute_latencyがインポートできること"""
        from tools.eval.metrics.latency import compute_latency

        assert callable(compute_latency)

    def test_function_signature(self):
        """関数シグネチャが期待通りであること"""
        from tools.eval.metrics.latency import compute_latency

        sig = inspect.signature(compute_latency)
        params = list(sig.parameters.keys())
        assert "model_path" in params
        assert "audio_path" in params

    def test_optional_parameters(self):
        """オプションパラメータにデフォルト値があること"""
        from tools.eval.metrics.latency import compute_latency

        sig = inspect.signature(compute_latency)
        params = sig.parameters

        assert params["index_path"].default is None
        assert params["f0_method"].default == "rmvpe"
        assert params["f0_up_key"].default == 0
        assert params["device"].default == "cuda"
        assert params["n_runs"].default == 3
        assert params["warmup"].default == 1


class TestLatencyInputValidation:
    """入力検証のテスト"""

    def test_missing_model_file(self, sine_wav):
        """存在しないモデルファイルでFileNotFoundError"""
        from tools.eval.metrics.latency import compute_latency

        audio = sine_wav(freq=440, duration=1.0, filename="test_audio.wav")
        with pytest.raises(FileNotFoundError):
            compute_latency(
                model_path="/nonexistent/model.pth",
                audio_path=audio,
            )

    def test_missing_audio_file(self, tmp_path):
        """存在しない音声ファイルでFileNotFoundError"""
        from tools.eval.metrics.latency import compute_latency

        # ダミーのモデルファイルを作成
        model = tmp_path / "dummy_model.pth"
        model.write_bytes(b"dummy")
        with pytest.raises(FileNotFoundError):
            compute_latency(
                model_path=str(model),
                audio_path="/nonexistent/audio.wav",
            )


class TestLatencyReturnFormat:
    """戻り値の形式テスト（モック使用）"""

    def test_return_dict_structure(self):
        """戻り値dictの必須キーを確認"""
        # compute_latencyの戻り値仕様を確認
        # value, unit, details が必須
        expected_keys = {"value", "unit", "details"}
        expected_detail_keys = {
            "rtf",
            "audio_duration_s",
            "latencies_ms",
            "n_runs",
            "device",
        }
        # これらのキーが仕様に含まれることをdocstringから確認
        from tools.eval.metrics.latency import compute_latency

        doc = compute_latency.__doc__
        for key in expected_keys:
            assert key in doc, f"docstringに'{key}'が記載されていること"
        for key in expected_detail_keys:
            assert key in doc, f"docstringに'{key}'が記載されていること"

    def test_return_dict_with_mock(self, sine_wav, monkeypatch):
        """モックを使って戻り値のdict構造を検証"""
        import sys
        from unittest.mock import MagicMock, patch

        from tools.eval.metrics import latency as latency_mod

        audio = sine_wav(freq=440, duration=1.0, filename="mock_audio.wav")

        # ダミーのモデルファイルパスを用意（isfileをモックするので実在不要）
        model_path = "/fake/model.pth"

        # os.path.isfile を常にTrueにする
        monkeypatch.setattr(
            latency_mod.os.path,
            "isfile",
            lambda p: True,
        )

        # 内部インポートされるモジュール群をモックする
        mock_vc_instance = MagicMock()
        mock_vc_instance.vc_single.return_value = ("Success", None)

        mock_vc_class = MagicMock(return_value=mock_vc_instance)
        mock_config_class = MagicMock()

        mock_librosa = MagicMock()
        mock_librosa.get_duration.return_value = 1.0

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        # sys.modules にモックモジュールを注入して、
        # compute_latency 内部の from ... import が実モジュールを
        # ロードしないようにする
        mock_configs_config = MagicMock()
        mock_configs_config.Config = mock_config_class

        mock_vc_modules = MagicMock()
        mock_vc_modules.VC = mock_vc_class

        mock_dotenv = MagicMock()

        # 注入前の状態を保存（テスト後にmonkeypatchが復元する）
        modules_to_inject = {
            "configs": MagicMock(),
            "configs.config": mock_configs_config,
            "infer": MagicMock(),
            "infer.modules": MagicMock(),
            "infer.modules.vc": MagicMock(),
            "infer.modules.vc.modules": mock_vc_modules,
            "dotenv": mock_dotenv,
        }

        # 既にインポート済みのモジュールを退避して上書き
        saved_modules = {}
        for mod_name, mock_mod in modules_to_inject.items():
            saved_modules[mod_name] = sys.modules.get(mod_name)
            monkeypatch.setitem(sys.modules, mod_name, mock_mod)

        # librosa と torch は関数内 import 文で使われる
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "librosa":
                return mock_librosa
            if name == "torch":
                return mock_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        result = latency_mod.compute_latency(
            model_path=model_path,
            audio_path=audio,
            device="cpu",
            n_runs=2,
            warmup=0,
        )

        # 戻り値の構造を検証
        assert isinstance(result, dict)
        assert "value" in result
        assert "unit" in result
        assert "details" in result
        assert result["unit"] == "ms"
        assert isinstance(result["value"], float)

        details = result["details"]
        assert "rtf" in details
        assert "audio_duration_s" in details
        assert "latencies_ms" in details
        assert "n_runs" in details
        assert "device" in details
        assert details["n_runs"] == 2
        assert details["device"] == "cpu"
        assert len(details["latencies_ms"]) == 2


@pytest.mark.slow
class TestLatencyIntegration:
    """統合テスト（実際のモデルとGPUが必要）"""

    def test_compute_latency_real(self):
        """実際のRVCモデルでレイテンシ計測"""
        pytest.skip("RVCモデルとGPUが必要なためスキップ")
