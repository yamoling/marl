import json
from dataclasses import is_dataclass, fields
from typing import Any
from abc import ABC

class Serializable(ABC):
    """
    A dataclass that can be serialized to json and back to a dataclass.
    
    Only the fields declared in the dataclass are serialized.
    """
    def __init__(self):
        assert is_dataclass(self), "The class must be a dataclass to be serializable"

    def as_dict(self):
        res: dict[str, Any] = {"__type__": self.__class__.__name__ }
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Serializable):
                value = value.as_dict()
            res[field.name] = value
        return res
    
    def as_json(self) -> str:
        return json.dumps(self.as_dict())
    

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)

    @classmethod
    def from_json(cls, json_data: str):
        return cls.from_dict(json.loads(json_data))