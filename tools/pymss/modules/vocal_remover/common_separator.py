class CommonSeparator:
    VOCAL_STEM = "Vocals"
    OTHER_STEM = "Other"
    BASS_STEM = "Bass"
    DRUM_STEM = "Drums"
    GUITAR_STEM = "Guitar"
    PIANO_STEM = "Piano"
    SYNTH_STEM = "Synthesizer"
    STRINGS_STEM = "Strings"
    WOODWINDS_STEM = "Woodwinds"
    BRASS_STEM = "Brass"
    WIND_INST_STEM = "Wind Inst"

    NON_ACCOM_STEMS = (
        VOCAL_STEM,
        OTHER_STEM,
        BASS_STEM,
        DRUM_STEM,
        GUITAR_STEM,
        PIANO_STEM,
        SYNTH_STEM,
        STRINGS_STEM,
        WOODWINDS_STEM,
        BRASS_STEM,
        WIND_INST_STEM,
    )

    def __init__(self, config):
        self.logger = config.get("logger")
        self.debug = config.get("debug")
        self.torch_device = config.get("torch_device")
        self.torch_device_cpu = config.get("torch_device_cpu")
        self.torch_device_mps = config.get("torch_device_mps")
        self.model_name = config.get("model_name")
        self.model_path = config.get("model_path")
        self.model_data = config.get("model_data")
        self.sample_rate = config.get("sample_rate")
        self.progress_callback = config.get("progress_callback", None)

        self.primary_stem_name = self.model_data.get("primary_stem", "primary_stem")
        self.secondary_stem_name = self.model_data.get("secondary_stem", "secondary_stem")
        self.is_karaoke = self.model_data.get("is_karaoke", False)
        self.is_bv_model = self.model_data.get("is_bv_model", False)
        self.bv_model_rebalance = self.model_data.get("is_bv_model_rebalanced", 0)

        self.primary_source = None
        self.secondary_source = None

        self.logger.info(f"VR params: model_name={self.model_name}, model_path={self.model_path}")
        self.logger.info(f"VR params: primary_stem={self.primary_stem_name}, secondary_stem={self.secondary_stem_name}")
        self.logger.debug(
            f"VR params: is_karaoke={self.is_karaoke}, is_bv_model={self.is_bv_model}, bv_model_rebalance={self.bv_model_rebalance}"
        )
