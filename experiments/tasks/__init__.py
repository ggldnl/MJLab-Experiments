import pkgutil
import importlib


def _auto_import_submodules():
    for finder, module_name, _ in pkgutil.walk_packages(
        path=__path__,
        prefix=f"{__name__}.",
        onerror=lambda name: print(f"[registry] failed to import {name}"),
    ):
        importlib.import_module(module_name)


_auto_import_submodules()