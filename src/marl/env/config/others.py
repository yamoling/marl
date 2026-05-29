from dataclasses import KW_ONLY, dataclass

from marlenv.adapters import SMAC

from .env_config import EnvConfig


@dataclass
class SMACConfig(EnvConfig[SMAC]):
    map_name: str
    _: KW_ONLY
    debug: bool = False

    def make_base_env(self):
        return SMAC(self.map_name, debug=self.debug)
