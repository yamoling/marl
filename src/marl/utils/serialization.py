from dataclasses import Field, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Self, Type, Union, get_args, get_origin

import orjson

# Use a hyphen (-) in the discriminator such that no attribute ever
# deserializes to that key.
DISCRIMINATOR_KEY = "class-name"


def default_serialization(obj):
    """Default behaviour for orjson serialization"""
    match obj:
        case set():
            return list(obj)
        case Path():
            return obj.as_posix()
    raise TypeError(f"Type {type(obj)} is not serializable")


def get_subclass_map(base_class: Type):
    """
    Recursively finds all subclasses and maps their names to the class object.
    """
    mapping = {base_class.__name__: base_class}
    for subclass in base_class.__subclasses__():
        mapping[subclass.__name__] = subclass
        # Recurse in case there are subclasses of subclasses
        mapping.update(get_subclass_map(subclass))
    return mapping


def get_subclass_from_name(base_class: Type, class_name: str) -> Type | None:
    """
    Retrieve the subclass whose name is `class_name`, if if exist.

    **Note:** the class provided as argument is not considered to be a subclass of itself.
    """
    for subclass in base_class.__subclasses__():
        if subclass.__name__ == class_name:
            return subclass
        # Recurse in case there are subclasses of subclasses
        result = get_subclass_from_name(subclass, class_name)
        if result is not None:
            return result
    return None


def deserialize_field(f: Field, value: Any):
    """
    Deserialize a field json-deserialized value to its actual value according to the following ordered rules:
        - datetimes are deserialized from ISO format
        - non-dict values are left unchanged
        - fields that inherit from `Serializable` return their deserialied value

    **Notes:**
        - These rules assume that there is no complex types such as `list[SomeComplexClass]` and currently
    fail at deserialising such values.
        - Union types other than `T | None` are not yet supported.
    """
    if f.type is datetime:
        return datetime.fromisoformat(value)
    if not isinstance(value, dict):
        return value
    actual_type = f.type
    # For union types like `X | None`, get the `X` type
    if get_origin(actual_type) is Union:
        # Filter out NoneType to find the actual class
        union_types = [a for a in get_args(actual_type) if a is not type(None)]
        if len(union_types) > 1:
            raise NotImplementedError(f"Union types other than `T | None` are not yet supported. Got type: {actual_type}.")
        actual_type = union_types[0]
    # If the resulting type is a Serializable subclass, then deserialize it
    if isinstance(actual_type, type) and issubclass(actual_type, Serializable):
        return actual_type.from_dict(value)
    return value


@dataclass
class Serializable:
    def __post_init__(self):
        return

    def to_dict(self):
        res = {}
        for field in fields(self):
            if not field.init:
                continue
            value = self.__dict__[field.name]
            if isinstance(value, Serializable):
                res[field.name] = value.to_dict()
            else:
                res[field.name] = value
        res[DISCRIMINATOR_KEY] = self.__class__.__name__
        return res

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Recursively build the configuration from a dictionary.
        Child objects that are serializable are deserialized thanks to their `from_dict` method.
        """
        # If the DISCRIMINATOR_KEY field is no longer there, it means that a parent class has already
        # tasked a subclass (current cls) to handle deserialisation.
        class_name = d.pop(DISCRIMINATOR_KEY, cls.__name__)
        if class_name != cls.__name__:
            subtype = get_subclass_from_name(cls, class_name)
            if subtype is None:
                raise KeyError(f"Unknown subclass {class_name} for {cls.__name__}")
            return subtype.from_dict(d)

        # Iterate on all fields to identify complex ones that require deserialization
        for f in fields(cls):
            d[f.name] = deserialize_field(f, d[f.name])
        return cls(**d)

    @classmethod
    def from_json(cls, data: bytes) -> Self:
        """Build the configuration from a JSON file."""
        d = orjson.loads(data)
        return cls.from_dict(d)

    def to_json(self, *, beautify: bool = False):
        option = None
        if beautify:
            option = orjson.OPT_INDENT_2
        return orjson.dumps(self.to_dict(), option=option, default=default_serialization)

    @classmethod
    def from_file(cls, path: Path | str):
        with open(path, "rb") as f:
            return cls.from_json(f.read())

    def to_file(self, path: Path | str, beautify: bool = False):
        if not isinstance(path, Path):
            path = Path(path)
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.to_json(beautify=beautify))
