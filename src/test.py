import marl
from lle import LLE
from marl.utils import serialization
import orjson


def main():
    env = LLE.level(6).build()
    print(env)
    data = orjson.dumps(env, option=orjson.OPT_SERIALIZE_NUMPY)
    env2 = serialization.structure(data, LLE)
    print(env2)


if __name__ == "__main__":
    main()
