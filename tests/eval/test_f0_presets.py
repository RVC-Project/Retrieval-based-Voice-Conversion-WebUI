"""Tests for f0 presets."""

from infer.lib.f0_presets import PRESETS, get_preset, get_preset_names


class TestPresets:
    def test_get_preset_names_returns_all(self):
        names = get_preset_names()
        assert "カスタム" in names
        assert "J-POP" in names
        assert "演歌" in names
        assert "アニソン" in names
        assert "話し声" in names

    def test_get_preset_existing(self):
        preset = get_preset("J-POP")
        assert preset["f0_method"] == "rmvpe"
        assert preset["f0_min"] == 65
        assert preset["f0_max"] == 1100
        assert preset["filter_radius"] == 1

    def test_get_preset_custom_has_none_values(self):
        preset = get_preset("カスタム")
        assert preset["f0_method"] is None
        assert preset["filter_radius"] is None
        assert preset["f0_min"] is None
        assert preset["f0_max"] is None

    def test_get_preset_nonexistent_returns_custom(self):
        preset = get_preset("存在しないプリセット")
        assert preset == PRESETS["カスタム"]

    def test_all_presets_have_required_keys(self):
        required_keys = {"f0_method", "filter_radius", "f0_min", "f0_max"}
        for name, preset in PRESETS.items():
            missing = required_keys - set(preset.keys())
            assert not missing, f"Preset '{name}' missing keys: {missing}"

    def test_valid_f0_methods(self):
        valid_methods = {None, "rmvpe", "fcpe", "harvest", "crepe", "pm"}
        for name, preset in PRESETS.items():
            assert preset["f0_method"] in valid_methods, f"Preset '{name}' has invalid f0_method: {preset['f0_method']}"

    def test_anime_preset_uses_fcpe(self):
        """アニソンプリセットはFCPEを使用すべき"""
        preset = get_preset("アニソン")
        assert preset["f0_method"] == "fcpe"

    def test_enka_preset_no_filter(self):
        """演歌プリセットはこぶし保存のためフィルタなし"""
        preset = get_preset("演歌")
        assert preset["filter_radius"] == 0
