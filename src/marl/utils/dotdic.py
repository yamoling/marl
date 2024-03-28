import copy
from typing import Any


class DotDic(dict):  # Source : https://github.com/minqi/learning-to-communicate-pytorch
    def __setattr__(self, name: str, value: Any):
        return self.__setitem__(name, value)

    def __delattr__(self, name: str):
        return self.__delitem__(name)

    def __getattr__(self, __name: str):
        return self.__getitem__(__name)

    def __deepcopy__(self, memo=None):
        return DotDic(copy.deepcopy(dict(self), memo=memo))
