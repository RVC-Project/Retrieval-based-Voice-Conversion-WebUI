import json

N_BINS = "n_bins"


def int_keys(d):
    return {int(key) if key.isdigit() else key: value for key, value in d}


class ModelParameters:
    def __init__(self, config_path=""):
        with open(config_path, "r") as f:
            self.param = json.load(f, object_pairs_hook=int_keys)

        for k in ["mid_side", "mid_side_b", "mid_side_b2", "stereo_w", "stereo_n", "reverse"]:
            self.param.setdefault(k, False)

        if N_BINS in self.param:
            self.param["bins"] = self.param[N_BINS]
