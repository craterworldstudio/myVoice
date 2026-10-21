import tomllib
#import types

class ConfigCategoryNode:
    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)

    def __contains__(self, item):
        return hasattr(self, item)


def _to_config_node(val):
    if isinstance(val, dict):
        node = ConfigCategoryNode()
        for k, v in val.items():
            setattr(node, k, _to_config_node(v))
        return node
    return val


class ConfigManager:

    def __init__(self, config_path="./config/config.toml"):
        self.config_path = config_path
        self.reload()

    def reload(self):
        with open(self.config_path, "rb") as f:
            self.rconfig_data = tomllib.load(f)

        for catn, content in self.rconfig_data.items():
            setattr(self, catn, _to_config_node(content))


config = ConfigManager()