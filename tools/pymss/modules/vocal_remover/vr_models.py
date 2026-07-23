import os


VR_MODEL_METADATA = {
    "10_SP-UVR-2B-32000-1.pth": {"primary_stem": "Instrumental", "secondary_stem": "Vocals", "vr_model_param": "2band_32000"},
    "11_SP-UVR-2B-32000-2.pth": {"primary_stem": "Instrumental", "secondary_stem": "Vocals", "vr_model_param": "2band_32000"},
    "12_SP-UVR-3B-44100.pth": {"primary_stem": "Instrumental", "secondary_stem": "Vocals", "vr_model_param": "3band_44100"},
    "13_SP-UVR-4B-44100-1.pth": {"primary_stem": "Instrumental", "secondary_stem": "Vocals", "vr_model_param": "4band_44100"},
    "14_SP-UVR-4B-44100-2.pth": {"primary_stem": "Instrumental", "secondary_stem": "Vocals", "vr_model_param": "4band_44100"},
    "15_SP-UVR-MID-44100-1.pth": {
        "primary_stem": "Instrumental",
        "secondary_stem": "Vocals",
        "vr_model_param": "3band_44100_mid",
    },
    "16_SP-UVR-MID-44100-2.pth": {
        "primary_stem": "Instrumental",
        "secondary_stem": "Vocals",
        "vr_model_param": "3band_44100_mid",
    },
    "17_HP-Wind_Inst-UVR.pth": {"primary_stem": "No Woodwinds", "secondary_stem": "Woodwinds", "vr_model_param": "4band_v3"},
    "1_HP-UVR.pth": {"primary_stem": "Instrumental", "secondary_stem": "Vocals", "vr_model_param": "4band_44100"},
    "2_HP-UVR.pth": {"primary_stem": "Instrumental", "secondary_stem": "Vocals", "vr_model_param": "4band_v2"},
    "3_HP-Vocal-UVR.pth": {"primary_stem": "Vocals", "secondary_stem": "Instrumental", "vr_model_param": "4band_44100"},
    "4_HP-Vocal-UVR.pth": {"primary_stem": "Vocals", "secondary_stem": "Instrumental", "vr_model_param": "4band_44100"},
    "5_HP-Karaoke-UVR.pth": {
        "primary_stem": "Instrumental",
        "secondary_stem": "Vocals",
        "vr_model_param": "4band_v2_sn",
        "is_karaoke": True,
    },
    "6_HP-Karaoke-UVR.pth": {
        "primary_stem": "Instrumental",
        "secondary_stem": "Vocals",
        "vr_model_param": "3band_44100_msb2",
        "is_karaoke": True,
    },
    "7_HP2-UVR.pth": {"primary_stem": "Instrumental", "secondary_stem": "Vocals", "vr_model_param": "3band_44100_msb2"},
    "8_HP2-UVR.pth": {"primary_stem": "Instrumental", "secondary_stem": "Vocals", "vr_model_param": "4band_44100"},
    "9_HP2-UVR.pth": {"primary_stem": "Instrumental", "secondary_stem": "Vocals", "vr_model_param": "4band_44100"},
    "Harmonic_Noise_Separation_yxlllc.pth": {
        "primary_stem": "No Aspiration",
        "secondary_stem": "Aspiration",
        "vr_model_param": "1band_sr44100_hl1024",
    },
    "MGM_HIGHEND_v4.pth": {
        "primary_stem": "Instrumental",
        "secondary_stem": "Vocals",
        "vr_model_param": "1band_sr44100_hl1024",
    },
    "MGM_LOWEND_A_v4.pth": {
        "primary_stem": "Instrumental",
        "secondary_stem": "Vocals",
        "vr_model_param": "1band_sr32000_hl512",
    },
    "MGM_LOWEND_B_v4.pth": {
        "primary_stem": "Instrumental",
        "secondary_stem": "Vocals",
        "vr_model_param": "1band_sr33075_hl384",
    },
    "MGM_MAIN_v4.pth": {"primary_stem": "Instrumental", "secondary_stem": "Vocals", "vr_model_param": "1band_sr44100_hl512"},
    "UVR-BVE-4B_SN-44100-1.pth": {
        "primary_stem": "Vocals",
        "secondary_stem": "Instrumental",
        "vr_model_param": "4band_v3_sn",
        "is_bv_model": True,
        "is_bv_model_rebalanced": 0.9,
        "nout": 64,
        "nout_lstm": 128,
    },
    "UVR-De-Echo-Aggressive.pth": {
        "primary_stem": "No Echo",
        "secondary_stem": "Echo",
        "vr_model_param": "4band_v3",
        "nout": 48,
        "nout_lstm": 128,
    },
    "UVR-De-Echo-Normal.pth": {
        "primary_stem": "No Echo",
        "secondary_stem": "Echo",
        "vr_model_param": "4band_v3",
        "nout": 48,
        "nout_lstm": 128,
    },
    "UVR-DeEcho-DeReverb.pth": {"primary_stem": "No Reverb", "secondary_stem": "Reverb", "vr_model_param": "4band_v3"},
    "UVR-DeNoise-Lite.pth": {
        "primary_stem": "Noise",
        "secondary_stem": "No Noise",
        "vr_model_param": "1band_sr44100_hl1024",
        "nout": 16,
        "nout_lstm": 128,
    },
    "UVR-DeNoise.pth": {
        "primary_stem": "Noise",
        "secondary_stem": "No Noise",
        "vr_model_param": "4band_v3",
        "nout": 48,
        "nout_lstm": 128,
    },
    "UVR-DeReverb-aufr33-jarredou_4band_v4_ms_fullband.pth": {
        "primary_stem": "Dry",
        "secondary_stem": "Reverb",
        "vr_model_param": "4band_v4_ms_fullband",
        "nout": 32,
        "nout_lstm": 128,
    },
}


def get_vr_model_metadata(model_path):
    model_name = os.path.basename(model_path)
    if model_name not in VR_MODEL_METADATA:
        raise ValueError(f"Unsupported VR model: {model_name}. Only the supported UVR/VR series weights are available.")
    data = dict(VR_MODEL_METADATA[model_name])
    data["model_name"] = model_name
    data["model_class"] = "VR_Models"
    return data
