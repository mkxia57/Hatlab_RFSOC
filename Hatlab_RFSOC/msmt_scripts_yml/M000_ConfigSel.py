cfgFileName = "./config_files/example_config.yml"



import yaml
def get_cfg_info():
    yml = yaml.safe_load(open(cfgFileName))
    config, info = yml["config"], yml["info"]
    return config, info