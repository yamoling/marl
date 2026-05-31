from dataclasses import MISSING, Field, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Self, TypeVar, get_origin

import orjson

from marl.utils.reflection import get_subclass_from_name, unwrap_optional

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


@dataclass
class Serializable:
    def __post_init__(self):
        return

    @property
    def name(self):
        return self.__class__.__name__

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
        res["name"] = self.name
        return res

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, exact_type: bool = False) -> Self:
        """
        Recursively build the configuration from a dictionary.
        Child objects that are serializable are deserialized thanks to their `from_dict` method.
        """
        # If the DISCRIMINATOR_KEY field is no longer there, it means that a parent class has already
        # tasked a subclass (current cls) to handle deserialization.
        class_name = d.pop(DISCRIMINATOR_KEY, cls.__name__)
        if class_name != cls.__name__ and not exact_type:
            subtype = get_subclass_from_name(cls, class_name)
            if subtype is None:
                raise KeyError(f"Unknown subclass {class_name} for {cls.__name__}")
            return subtype.from_dict(d)

        # Iterate on all fields to identify complex ones that require deserialization
        init_dict = {}
        for f in fields(cls):
            if not f.init:
                continue
            if f.name not in d:
                if f.default is not MISSING:
                    init_dict[f.name] = f.default
                else:
                    raise KeyError(f"Missing value for required field {f.name} of class {cls.__name__}")
            else:
                field_value = cls.deserialize_field(f, d[f.name])
                init_dict[f.name] = field_value
        return cls(**init_dict)

    @classmethod
    def from_json(cls, data: bytes, *, exact_type: bool = False) -> Self:
        """Build the configuration from a JSON file."""
        d = orjson.loads(data)
        return cls.from_dict(d, exact_type=exact_type)

    def to_json(self, *, beautify: bool = False):
        option = None
        if beautify:
            option = orjson.OPT_INDENT_2
        return orjson.dumps(self.to_dict(), option=option, default=default_serialization)

    @classmethod
    def from_file(cls, path: Path | str, *, exact_type: bool = False):
        with open(path, "rb") as f:
            return cls.from_json(f.read(), exact_type=exact_type)

    def to_file(self, path: Path | str, *, beautify: bool = False):
        if not isinstance(path, Path):
            path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            f.write(self.to_json(beautify=beautify))

    @classmethod
    def deserialize_field(cls, f: Field, value: Any):
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
        field_type = resolve_type(f.type)
        assert issubclass(field_type, Serializable), (
            f"Attribute {f.name} of class {cls} is of type {field_type}, which is not Serializable !"
        )
        return field_type.from_dict(value)


def resolve_type(field_type):
    """
    Resolve a field type annotation to its corresponding class object.

    Rules (applied in order):

    1. Plain types (e.g. ``x: SomeClass``) → the class itself.
    2. Optional types (e.g. ``x: SomeClass | None``) → the non-``None`` type,
       via :func:`~marl.utils.reflection.unwrap_optional`.
    3. Constrained TypeVars (e.g. ``x: T`` where ``T: SomeClass``) → their
       bound.
    4. Generic types (e.g. ``x: SomeGenericType[T]``) → their origin class.
    """
    if isinstance(field_type, type):
        return field_type
    # Resolve optional types (e.g. `SomeClass | None` → `SomeClass`)
    unwrapped = unwrap_optional(field_type)
    if unwrapped is not field_type:
        return resolve_type(unwrapped)
    # Resolve `x: T` where `T: SomeClass` → `SomeClass`
    if isinstance(field_type, TypeVar):
        if field_type.__bound__ is None:
            raise TypeError(f"Generic type variable {field_type} is not constrained. Only constrained can be deserialized.")
        return resolve_type(field_type.__bound__)
    # Resolve `x: SomeGenericType[T]` → `SomeGenericType`
    origin = get_origin(field_type)
    if origin is not None:
        return resolve_type(origin)
    raise TypeError(f"Unsupported field type {field_type} for deserialization.")
