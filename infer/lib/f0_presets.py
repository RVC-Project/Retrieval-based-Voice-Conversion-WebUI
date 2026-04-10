"""歌声変換プリセット定義"""

PRESETS = {
    "カスタム": {
        "description": "手動設定（プリセットなし）",
        "f0_method": None,
        "filter_radius": None,
        "f0_min": None,
        "f0_max": None,
    },
    "J-POP": {
        "description": "J-POP向け。ビブラート保存、標準音域",
        "f0_method": "rmvpe",
        "filter_radius": 1,
        "f0_min": 65,
        "f0_max": 1100,
    },
    "演歌": {
        "description": "演歌向け。こぶし保存、フィルタなし",
        "f0_method": "rmvpe",
        "filter_radius": 0,
        "f0_min": 65,
        "f0_max": 900,  # 1100→900: 演歌の音域に適正化
    },
    "アニソン": {
        "description": "アニソン向け。広音域、高速F0追従",
        "f0_method": "fcpe",
        "filter_radius": 1,
        "f0_min": 80,
        "f0_max": 1200,  # 1400→1200: ハーモニクス誤検出リスク低減
    },
    "話し声": {
        "description": "話し声向け。従来互換パラメータ",
        "f0_method": "rmvpe",
        "filter_radius": 3,
        "f0_min": 50,
        "f0_max": 800,
    },
}


def get_preset_names():
    return list(PRESETS.keys())


def get_preset(name):
    return PRESETS.get(name, PRESETS["カスタム"])
