import logging


class CSVFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self._headers = list[str]()

    def format(self, record):
        data = record.msg
        if not isinstance(data, dict):
            raise TypeError(f"Expected a dict[str, float] but got a {type(data)}")
        res = ""
        if len(self._headers) == 0:
            self._headers = list(data.keys()) + ["timestamp_sec"]
            res += ",".join(self._headers) + "\n"
        data["timestamp_sec"] = record.created
        res += ",".join([str(data.get(key, None)) for key in self._headers])
        return res
