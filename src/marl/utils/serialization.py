import os
from typing import Type, overload

import numpy as np
import orjson
import torch
from cattrs import Converter
from lle import World
import logging


def default_serialization(obj):
    """Default behaviour for orjson serialization"""
    match obj:
        case torch.nn.Parameter() as param:
            return f"Parameter(shape={param.shape}, dtype={param.dtype})"
        case set():
            return list(obj)
        case torch.optim.Optimizer() as optim:
            return {
                "name": optim.__class__.__name__,
                "param_groups": [{k: v for k, v in group.items()} for group in optim.param_groups],
            }
        case np.ndarray():
            return obj.tolist()
        case np.signedinteger():
            return int(obj)
        case np.floating():
            return float(obj)
        case World():
            return obj.world_string
    raise TypeError(f"Type {type(obj)} is not serializable")


def get_subclass_map(base_class: Type) -> dict[str, Type]:
    """
    Recursively finds all subclasses and maps their names to the class object.
    """
    mapping = {}
    for subclass in base_class.__subclasses__():
        mapping[subclass.__name__] = subclass
        # Recurse in case there are subclasses of subclasses
        mapping.update(get_subclass_map(subclass))
    return mapping


@overload
def structure[T](filepath: str, to: Type[T], /) -> T:
    """Create an instance of `T` from the given file path."""


@overload
def structure[T](json: str | bytes | bytearray, to: Type[T], /) -> T:
    """Create an instance of `T` from the given file JSON."""


def structure[T](path_or_json: str | bytes | bytearray | memoryview, to: Type[T]):
    _CONVERTER.initialize_hooks()
    match path_or_json:
        case str(path) if os.path.exists(path):
            with open(path, "rb") as f:
                json = f.read()
        case str() | bytes() | bytearray() | memoryview():
            json = path_or_json
        case other:
            raise NotImplementedError(f"Unsupported deserialization argument of type {type(other)}!")
    json = orjson.loads(json)
    return _CONVERTER.structure(json, to)


class _InitConverter(Converter):
    def __init__(self):
        super().__init__()
        self._is_initialized = False

    def initialize_hooks(self):
        if self._is_initialized:
            return
        from marl.models import Trainer, ReplayMemory, IRModule, Mixer, NN, Agent

        self.register_marlenv()
        self.register(Trainer)
        self.register(ReplayMemory)
        self.register(IRModule)
        self.register(Mixer)
        self.register(NN)
        self.register(Agent)
        self._is_initialized = True

    def register(self, base_class: Type):
        """
        Creates a structure hook that automatically dispatches to the correct subclass.
        """
        # Build the map once
        subclass_map = get_subclass_map(base_class)
        logging.info(f"{len(subclass_map)} subclasses found for {base_class.__name__}")

        def structure_hook(data: dict, _: Type):
            # Defaulting to a 'type' field in the JSON
            type_name = data.get("name")
            assert isinstance(type_name, str)
            target_class = subclass_map.get(type_name)
            if not target_class:
                raise ValueError(f"Unsupported type '{type_name}'. Must be one of: {list(subclass_map.keys())}")

            return self.structure(data, target_class)

        self.register_structure_hook(base_class, structure_hook)

    def register_marlenv(self):
        """
        Creates a structure hook that automatically dispatches to the correct subclass.
        """
        from marlenv import MARLEnv

        # Build the map once
        subclass_map = get_subclass_map(MARLEnv)
        logging.info(f"{len(subclass_map)} subclasses found for {MARLEnv.__name__}")

        def structure_hook(data: dict, _: Type):
            # Defaulting to a 'type' field in the JSON
            type_name = data.get("name")
            assert isinstance(type_name, str)
            target_class = subclass_map.get(type_name)
            if not target_class:
                raise ValueError(f"Unsupported type '{type_name}'. Must be one of: {list(subclass_map.keys())}")
            return self.structure(data, target_class)

        self.register_structure_hook(MARLEnv, structure_hook)


_CONVERTER = _InitConverter()
