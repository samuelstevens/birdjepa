"""Configuration loading and merging utilities."""

import dataclasses
import logging
import pathlib
import types
import typing as tp

import beartype

import birdjepa.augment
import birdjepa.data
import birdjepa.nn.objectives
import birdjepa.nn.transformer

T = tp.TypeVar("T")

logger = logging.getLogger(__name__)

# Registry for union type resolution via __class__ marker
TYPE_REGISTRY: dict[str, type] = {
    # Data configs
    "XenoCanto": birdjepa.data.XenoCanto,
    "Cifar100": birdjepa.data.Cifar100,
    # Model configs
    "TransformerConfig": birdjepa.nn.transformer.Transformer,
    "Transformer": birdjepa.nn.transformer.Transformer,
    "DebugConfig": birdjepa.nn.transformer.Debug,
    "Debug": birdjepa.nn.transformer.Debug,
    # Objective configs
    "SupervisedConfig": birdjepa.nn.objectives.SupervisedConfig,
    "LeJEPAConfig": birdjepa.nn.objectives.LeJEPAConfig,
    "PixioConfig": birdjepa.nn.objectives.PixioConfig,
    # Augmentation configs
    "RandomCrop": birdjepa.augment.RandomCrop,
    "HorizontalFlip": birdjepa.augment.HorizontalFlip,
    "GaussianNoise": birdjepa.augment.GaussianNoise,
    "FreqMask": birdjepa.augment.FreqMask,
    "TimeMask": birdjepa.augment.TimeMask,
    "RandomResizedCrop": birdjepa.augment.RandomResizedCrop,
    "ColorJitter": birdjepa.augment.ColorJitter,
    "GaussianBlur": birdjepa.augment.GaussianBlur,
    "Solarize": birdjepa.augment.Solarize,
}


@beartype.beartype
def load_sweep(sweep_fpath: pathlib.Path) -> list[dict]:
    """Load a sweep file and return the list of config dicts.

    Args:
        sweep_fpath: Path to a Python file with a `make_cfgs()` function.

    Returns:
        List of config dictionaries from `make_cfgs()`.
    """
    namespace: dict = {}
    exec(sweep_fpath.read_text(), namespace)
    result = namespace["make_cfgs"]()
    assert isinstance(result, list), (
        f"make_cfgs() returned {type(result)}, expected list"
    )
    return result


@beartype.beartype
def get_non_default_values(obj: T, default_obj: T) -> dict:
    """Recursively find fields that differ from defaults."""
    assert dataclasses.is_dataclass(obj) and not isinstance(obj, type)
    assert dataclasses.is_dataclass(default_obj) and not isinstance(default_obj, type)

    diff: dict = {}
    for field in dataclasses.fields(obj):
        obj_value = getattr(obj, field.name)
        default_value = getattr(default_obj, field.name)

        if obj_value == default_value:
            continue

        if (
            dataclasses.is_dataclass(obj_value)
            and dataclasses.is_dataclass(default_value)
            and type(obj_value) is type(default_value)
        ):
            nested_diff = get_non_default_values(obj_value, default_value)
            if nested_diff:
                diff[field.name] = nested_diff
        else:
            diff[field.name] = obj_value

    return diff


@beartype.beartype
def _filter_overridden_fields(
    dct: dict[str, object], overridden: dict[str, object], override_obj=None
) -> dict[str, object]:
    """Remove fields from `dct` that are present in `overridden`.

    Args:
        dct: Sweep dict to filter.
        overridden: Dict of non-default values from CLI.
        override_obj: The actual CLI override object (for type checking).
    """
    result: dict = {}
    for key, value in dct.items():
        if key not in overridden:
            result[key] = value
            continue

        if isinstance(value, dict):
            overridden_value = overridden.get(key)

            # If sweep has __class__ and CLI specifies a different type, CLI wins
            if "__class__" in value and override_obj is not None:
                cli_obj = getattr(override_obj, key, None)
                if dataclasses.is_dataclass(cli_obj):
                    sweep_type = value["__class__"]
                    cli_type = type(cli_obj).__name__
                    if sweep_type != cli_type:
                        # Different types - CLI override wins, skip sweep value
                        continue

            if isinstance(overridden_value, dict):
                nested_obj = getattr(override_obj, key, None) if override_obj else None
                filtered = _filter_overridden_fields(
                    value, overridden_value, nested_obj
                )
                if filtered:
                    result[key] = filtered
                continue

            if dataclasses.is_dataclass(overridden_value) and not isinstance(
                overridden_value, type
            ):
                try:
                    default_value = type(overridden_value)()
                except TypeError:
                    continue

                nested_overrides = get_non_default_values(
                    overridden_value, default_value
                )
                nested_obj = getattr(override_obj, key, None) if override_obj else None
                filtered = _filter_overridden_fields(
                    value, nested_overrides, nested_obj
                )
                if filtered:
                    result[key] = filtered
                continue

    return result


@beartype.beartype
def _convert_value(value: object, field_type: object) -> object:
    """Convert a value to the correct type based on field_type."""
    origin = tp.get_origin(field_type)
    args = tp.get_args(field_type)

    # Handle __class__ marker for union types
    if isinstance(value, dict) and "__class__" in value:
        type_name = value["__class__"]
        assert type_name in TYPE_REGISTRY, f"Unknown type: {type_name}"
        cls = TYPE_REGISTRY[type_name]
        remaining = {k: v for k, v in value.items() if k != "__class__"}
        if not remaining:
            return cls()
        # Recursively convert remaining fields using the target class's field types
        return dict_to_dataclass(remaining, cls)

    # Handle lists (e.g., list of augmentations)
    if origin is list and isinstance(value, list):
        item_type = args[0] if args else None
        return [_convert_value(item, item_type) for item in value]

    if origin is tuple and args:
        return tuple(value) if isinstance(value, list) else value
    elif dataclasses.is_dataclass(field_type):
        return dict_to_dataclass(value, field_type)
    elif field_type is pathlib.Path:
        return pathlib.Path(value) if value is not None else value
    elif origin is tp.Union and pathlib.Path in args:
        return pathlib.Path(value) if value is not None else value
    elif origin is types.UnionType and pathlib.Path in args:
        return pathlib.Path(value) if value is not None else value
    else:
        return value


@beartype.beartype
def _recursive_dataclass_update(obj, updates: dict[str, object]):
    """Recursively update nested dataclasses."""
    if not dataclasses.is_dataclass(obj):
        return updates

    result: dict = {}
    for key, value in updates.items():
        if not hasattr(obj, key):
            result[key] = value
            continue

        attr = getattr(obj, key)
        field_type = type(obj).__dataclass_fields__[key].type

        # Check for __class__ marker first (for union types)
        if isinstance(value, dict) and "__class__" in value:
            result[key] = _convert_value(value, field_type)
        elif dataclasses.is_dataclass(attr) and isinstance(value, dict):
            nested_updates = _recursive_dataclass_update(attr, value)
            result[key] = dataclasses.replace(attr, **nested_updates)
        else:
            result[key] = _convert_value(value, field_type)

    return result


@beartype.beartype
def dict_to_dataclass(data: dict, cls: type[T]) -> T:
    """Recursively convert a dictionary to a dataclass instance."""
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs: dict = {}

    for field_name, field_type in field_types.items():
        if field_name not in data:
            continue

        value = data[field_name]
        kwargs[field_name] = _convert_value(value, field_type)

    return cls(**kwargs)


@beartype.beartype
def load_cfgs(
    override: T, *, default: T, sweep_dcts: list[dict]
) -> tuple[list[T], list[str]]:
    """Load a list of configs from CLI override and sweep dicts.

    Args:
        override: Command-line overridden values.
        default: The default values for a config.
        sweep_dcts: A list of dictionaries from sweep files.

    Returns:
        A list of configs and a list of errors.
    """
    assert dataclasses.is_dataclass(override) and not isinstance(override, type)
    assert dataclasses.is_dataclass(default) and not isinstance(default, type)

    if not sweep_dcts:
        return [override], []

    overridden_fields = get_non_default_values(override, default)

    cfgs: list[T] = []
    errs: list[str] = []

    for sweep_dct in sweep_dcts:
        filtered_dct = _filter_overridden_fields(sweep_dct, overridden_fields, override)

        if not filtered_dct:
            cfgs.append(override)
            continue

        try:
            updates = _recursive_dataclass_update(override, filtered_dct)
            cfgs.append(dataclasses.replace(override, **updates))
        except Exception as err:
            errs.append(str(err))

    return cfgs, errs
