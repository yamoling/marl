class Registry[K, V](dict[K, V]):
    def __init__(self, items: dict[K, V]):
        super().__init__()
        self.update(items)

    def _error_message(self, missing_key: K):
        err_msg = "\n".join([f" - {key}" for key in self.keys()])
        return f"Unsupported configuration: {missing_key}.\nSupported combinations are:\n{err_msg}"

    def __getitem__(self, key: K, /) -> V:
        try:
            return super().__getitem__(key)
        except KeyError:
            raise NotImplementedError(self._error_message(key))
