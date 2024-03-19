import copy

class DotDic(dict): # Source : https://github.com/minqi/learning-to-communicate-pytorch
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __deepcopy__(self, memo=None):
		return DotDic(copy.deepcopy(dict(self), memo=memo))