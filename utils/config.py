import tomllib
#import types

class ConfigCategoryNode:
    pass

class ConfigManager:

    def __init__(self):
        self.config_data = None

        with open("./config/config.toml", "rb") as f:
            self.rconfig_data = tomllib.load(f)

            #print(self.rconfig_data)
        
        for catn in self.rconfig_data:
            section = ConfigCategoryNode()
            setattr(self, catn, section)
            for item in self.rconfig_data[catn]:
                setattr(section, item, self.rconfig_data[catn][item])


config = ConfigManager()
print(config.audio.sample_rate)