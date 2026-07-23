from importlib import import_module
import sys

_LOCAL_MODULE_PREFIX = "pymss.modules."
_CORE_MODULE_PREFIX = "pymss_core.modules."


def alias_module(local_name, core_name):
    if not local_name.startswith(_LOCAL_MODULE_PREFIX):
        raise ValueError(f"invalid local module alias: {local_name}")
    if not core_name.startswith(_CORE_MODULE_PREFIX):
        raise ValueError(f"invalid core module alias: {core_name}")
    module = import_module(core_name)
    sys.modules[local_name] = module
    return module


def alias_submodules(local_package, core_package, names):
    for name in names:
        alias_module(f"{local_package}.{name}", f"{core_package}.{name}")
