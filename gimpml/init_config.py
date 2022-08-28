import os,sys,json
from plugins.m_constants import CONFIG_FILE

config_fname = f"{os.path.dirname(os.path.realpath(__file__))}\\{CONFIG_FILE}"
if not os.path.exists(config_fname):
    with open(config_fname, "w") as f:
        json.dump({
            "python_path": f"{sys.executable}",
            "weight_path": f"{os.path.dirname(os.path.realpath(__file__))}\\weights\\"
        }, f, ensure_ascii=True, indent=4, sort_keys=True)