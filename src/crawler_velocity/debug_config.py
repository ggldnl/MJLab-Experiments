"""Debug script to inspect observation configuration and sensor references."""

from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
import json

cfg = make_velocity_env_cfg()

print("=" * 80)
print("ACTOR OBSERVATIONS")
print("=" * 80)
for term_name, term_cfg in cfg.observations["actor"].terms.items():
    print(f"\nTerm: {term_name}")
    print(f"  Type: {type(term_cfg).__name__}")
    print(f"  Func: {term_cfg.func.__name__ if hasattr(term_cfg, 'func') else 'N/A'}")

    if hasattr(term_cfg, 'params'):
        print(f"  Params:")
        for key, value in term_cfg.params.items():
            print(f"    {key}: {value}")

    # print all attributes
    print(f"  All attributes:")
    for attr in dir(term_cfg):
        if not attr.startswith('_'):
            try:
                val = getattr(term_cfg, attr)
                if not callable(val):
                    print(f"    {attr}: {val}")
            except:
                pass

print("\n" + "=" * 80)
print("CRITIC OBSERVATIONS")
print("=" * 80)
for term_name, term_cfg in cfg.observations["critic"].terms.items():
    print(f"\nTerm: {term_name}")
    print(f"  Type: {type(term_cfg).__name__}")
    print(f"  Func: {term_cfg.func.__name__ if hasattr(term_cfg, 'func') else 'N/A'}")

    if hasattr(term_cfg, 'params'):
        print(f"  Params:")
        for key, value in term_cfg.params.items():
            print(f"    {key}: {value}")

    print(f"  All attributes:")
    for attr in dir(term_cfg):
        if not attr.startswith('_'):
            try:
                val = getattr(term_cfg, attr)
                if not callable(val):
                    print(f"    {attr}: {val}")
            except:
                pass

print("\n" + "=" * 80)
print("SENSORS IN DEFAULT CONFIG")
print("=" * 80)
if cfg.scene.sensors:
    for sensor in cfg.scene.sensors:
        print(f"  {sensor.name}")
else:
    print("  (None)")