from pymss.modules._core_shims import alias_submodules

alias_submodules(
    __name__,
    "pymss_core.modules.vocal_remover.uvr_lib_v5",
    (
        "vr_network",
        "vr_network.layers",
        "vr_network.layers_new",
        "vr_network.model_param_init",
        "vr_network.nets",
        "vr_network.nets_new",
    ),
)
