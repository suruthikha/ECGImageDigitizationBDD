import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load():
    env = {}
    for line in open(_HERE / "paths.env"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"')
        val = re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), ""), val)
        env[key.strip()] = val
    return env


P = _load()