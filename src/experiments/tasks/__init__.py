import pkgutil
import importlib

def _auto_import_submodules():
    for _, module_name, _ in pkgutil.walk_packages(__path__):
        importlib.import_module(f"{__name__}.{module_name}")

_auto_import_submodules()