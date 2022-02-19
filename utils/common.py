from configparser import ConfigParser


def parse_config(config_file):
    config = ConfigParser()
    config.read(config_file)
    return config
