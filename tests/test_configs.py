"""Tests for birdjepa.configs module."""

import dataclasses
import pathlib

import beartype
import pytest

import birdjepa.augment
import birdjepa.configs
import birdjepa.data
import birdjepa.nn.objectives
import birdjepa.nn.transformer
import birdjepa.pretrain


@pytest.fixture
def sweep_path(tmp_path):
    """Create a temporary sweep file."""
    sweep_file = tmp_path / "sweep.py"
    sweep_file.write_text("""
def make_cfgs():
    return [
        {
            "train_data": {"__class__": "Cifar100"},
            "model": {
                "input_h": 32,
                "input_w": 32,
                "patch_h": 4,
                "patch_w": 4,
                "embed_dim": 128,
                "depth": 6,
                "n_heads": 4,
            },
            "batch_size": 128,
            "epochs": 100,
        }
    ]
""")
    return sweep_file


def test_load_sweep(sweep_path):
    """Test loading sweep file returns list of dicts."""
    dcts = birdjepa.configs.load_sweep(sweep_path)
    assert len(dcts) == 1
    assert dcts[0]["batch_size"] == 128
    assert dcts[0]["train_data"]["__class__"] == "Cifar100"


def test_load_cfgs_basic(sweep_path):
    """Test basic config loading from sweep."""
    sweep_dcts = birdjepa.configs.load_sweep(sweep_path)
    cfgs, errs = birdjepa.configs.load_cfgs(
        birdjepa.pretrain.Config(),
        default=birdjepa.pretrain.Config(),
        sweep_dcts=sweep_dcts,
    )
    assert len(errs) == 0
    assert len(cfgs) == 1

    cfg = cfgs[0]
    assert isinstance(cfg.train_data, birdjepa.data.Cifar100)
    assert cfg.model.input_h == 32
    assert cfg.model.embed_dim == 128
    assert cfg.batch_size == 128
    assert cfg.epochs == 100


def test_cli_override_simple_field(sweep_path):
    """Test CLI override of simple field takes priority over sweep."""
    sweep_dcts = birdjepa.configs.load_sweep(sweep_path)

    # CLI sets epochs=5, sweep has epochs=100
    cli_cfg = birdjepa.pretrain.Config(epochs=5)
    cfgs, errs = birdjepa.configs.load_cfgs(
        cli_cfg,
        default=birdjepa.pretrain.Config(),
        sweep_dcts=sweep_dcts,
    )

    assert len(errs) == 0
    cfg = cfgs[0]
    assert cfg.epochs == 5  # CLI wins


def test_cli_override_nested_field(sweep_path):
    """Test CLI override of nested model field takes priority."""
    sweep_dcts = birdjepa.configs.load_sweep(sweep_path)

    # CLI sets embed_dim=256, sweep has embed_dim=128
    cli_cfg = birdjepa.pretrain.Config(
        model=birdjepa.nn.transformer.Config(embed_dim=256)
    )
    cfgs, errs = birdjepa.configs.load_cfgs(
        cli_cfg,
        default=birdjepa.pretrain.Config(),
        sweep_dcts=sweep_dcts,
    )

    assert len(errs) == 0
    cfg = cfgs[0]
    assert cfg.model.embed_dim == 256  # CLI wins
    # But sweep's other model fields should still apply
    assert cfg.model.input_h == 32
    assert cfg.model.depth == 6


def test_cli_override_different_data_type(sweep_path):
    """Test CLI override with different data type takes priority."""
    sweep_dcts = birdjepa.configs.load_sweep(sweep_path)

    # CLI sets train_data=XenoCanto, sweep has train_data=Cifar100
    cli_cfg = birdjepa.pretrain.Config(train_data=birdjepa.data.XenoCanto(subset="XCL"))
    cfgs, errs = birdjepa.configs.load_cfgs(
        cli_cfg,
        default=birdjepa.pretrain.Config(),
        sweep_dcts=sweep_dcts,
    )

    assert len(errs) == 0
    cfg = cfgs[0]
    # CLI type wins over sweep type
    assert isinstance(cfg.train_data, birdjepa.data.XenoCanto)
    assert cfg.train_data.subset == "XCL"


def test_get_non_default_values():
    """Test finding fields that differ from defaults."""
    default = birdjepa.pretrain.Config()
    modified = birdjepa.pretrain.Config(epochs=5, batch_size=32)

    diff = birdjepa.configs.get_non_default_values(modified, default)
    assert diff == {"epochs": 5, "batch_size": 32}


def test_get_non_default_values_nested():
    """Test finding nested fields that differ from defaults."""
    default = birdjepa.pretrain.Config()
    modified = birdjepa.pretrain.Config(
        model=birdjepa.nn.transformer.Config(embed_dim=256)
    )

    diff = birdjepa.configs.get_non_default_values(modified, default)
    assert "model" in diff
    assert diff["model"] == {"embed_dim": 256}


def test_empty_sweep_returns_override():
    """Test empty sweep list returns just the override config."""
    override = birdjepa.pretrain.Config(epochs=5)
    cfgs, errs = birdjepa.configs.load_cfgs(
        override,
        default=birdjepa.pretrain.Config(),
        sweep_dcts=[],
    )

    assert len(errs) == 0
    assert len(cfgs) == 1
    assert cfgs[0].epochs == 5


def test_sweep_with_multiple_cli_overrides(sweep_path):
    """Test multiple CLI overrides applied to sweep config."""
    sweep_dcts = birdjepa.configs.load_sweep(sweep_path)

    # CLI overrides epochs and batch_size, sweep has epochs=100, batch_size=128
    cli_cfg = birdjepa.pretrain.Config(epochs=3, batch_size=32)
    cfgs, errs = birdjepa.configs.load_cfgs(
        cli_cfg,
        default=birdjepa.pretrain.Config(),
        sweep_dcts=sweep_dcts,
    )

    assert len(errs) == 0
    cfg = cfgs[0]
    # CLI overrides win
    assert cfg.epochs == 3
    assert cfg.batch_size == 32
    # Sweep values still apply for non-overridden fields
    assert isinstance(cfg.train_data, birdjepa.data.Cifar100)
    assert cfg.model.input_h == 32
    assert cfg.model.embed_dim == 128


# -----------------------------------------------------------------------------
# Tests for _convert_value
# -----------------------------------------------------------------------------


def test_convert_value_class_marker():
    """Test __class__ marker resolution from TYPE_REGISTRY."""
    value = {"__class__": "RandomCrop", "padding": 8}
    result = birdjepa.configs._convert_value(value, birdjepa.augment.Config)
    assert isinstance(result, birdjepa.augment.RandomCrop)
    assert result.padding == 8


def test_convert_value_class_marker_defaults():
    """Test __class__ marker with default values."""
    value = {"__class__": "HorizontalFlip"}
    result = birdjepa.configs._convert_value(value, birdjepa.augment.Config)
    assert isinstance(result, birdjepa.augment.HorizontalFlip)
    assert result.p == 0.5  # default


def test_convert_value_list_of_class_markers():
    """Test list of items with __class__ markers."""
    value = [
        {"__class__": "RandomCrop", "padding": 4},
        {"__class__": "HorizontalFlip", "p": 0.3},
    ]
    result = birdjepa.configs._convert_value(value, list[birdjepa.augment.Config])
    assert len(result) == 2
    assert isinstance(result[0], birdjepa.augment.RandomCrop)
    assert result[0].padding == 4
    assert isinstance(result[1], birdjepa.augment.HorizontalFlip)
    assert result[1].p == 0.3


def test_convert_value_tuple():
    """Test tuple conversion from list."""
    value = ["a", "b", "c"]
    result = birdjepa.configs._convert_value(value, tuple[str, ...])
    assert result == ("a", "b", "c")


def test_convert_value_path():
    """Test pathlib.Path conversion."""
    value = "/tmp/test"
    result = birdjepa.configs._convert_value(value, pathlib.Path)
    assert result == pathlib.Path("/tmp/test")
    assert isinstance(result, pathlib.Path)


def test_convert_value_optional_path():
    """Test optional path conversion."""
    result_some = birdjepa.configs._convert_value("/tmp/test", pathlib.Path | None)
    assert result_some == pathlib.Path("/tmp/test")

    result_none = birdjepa.configs._convert_value(None, pathlib.Path | None)
    assert result_none is None


def test_convert_value_passthrough():
    """Test that unknown types pass through unchanged."""
    value = 42
    result = birdjepa.configs._convert_value(value, int)
    assert result == 42


# -----------------------------------------------------------------------------
# Tests for dict_to_dataclass
# -----------------------------------------------------------------------------


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class SimpleTestConfig:
    value: int = 1


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class NestedInnerTestConfig:
    x: int = 5
    y: int = 10


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class NestedTestConfig:
    inner: NestedInnerTestConfig = dataclasses.field(
        default_factory=NestedInnerTestConfig
    )
    outer_value: int = 100


def test_dict_to_dataclass_simple():
    """Test simple dict to dataclass conversion."""
    data = {"value": 42}
    result = birdjepa.configs.dict_to_dataclass(data, SimpleTestConfig)
    assert result.value == 42


def test_dict_to_dataclass_missing_fields():
    """Missing fields use defaults."""
    data = {}
    result = birdjepa.configs.dict_to_dataclass(data, SimpleTestConfig)
    assert result.value == 1  # default


def test_dict_to_dataclass_non_dataclass():
    """Test with non-dataclass returns input unchanged."""
    data = {"a": 1, "b": 2}
    result = birdjepa.configs.dict_to_dataclass(data, dict)
    assert result == {"a": 1, "b": 2}


# -----------------------------------------------------------------------------
# Tests for augmentations in data configs
# -----------------------------------------------------------------------------


def test_augmentations_in_data_config():
    """Test augmentations list with __class__ markers in data config."""
    sweep_dcts = [
        {
            "train_data": {
                "__class__": "Cifar100",
                "augmentations": [
                    {"__class__": "RandomCrop", "padding": 4},
                    {"__class__": "HorizontalFlip", "p": 0.5},
                ],
            }
        }
    ]
    cfgs, errs = birdjepa.configs.load_cfgs(
        birdjepa.pretrain.Config(),
        default=birdjepa.pretrain.Config(),
        sweep_dcts=sweep_dcts,
    )
    assert len(errs) == 0, f"Errors: {errs}"
    assert len(cfgs) == 1
    assert isinstance(cfgs[0].train_data, birdjepa.data.Cifar100)
    assert len(cfgs[0].train_data.augmentations) == 2
    assert isinstance(cfgs[0].train_data.augmentations[0], birdjepa.augment.RandomCrop)
    assert cfgs[0].train_data.augmentations[0].padding == 4
    assert isinstance(
        cfgs[0].train_data.augmentations[1], birdjepa.augment.HorizontalFlip
    )


def test_objective_config_class_marker():
    """Test __class__ marker for objective configs."""
    sweep_dcts = [{"objective": {"__class__": "LeJEPAConfig", "proj_dim": 32}}]
    cfgs, errs = birdjepa.configs.load_cfgs(
        birdjepa.pretrain.Config(),
        default=birdjepa.pretrain.Config(),
        sweep_dcts=sweep_dcts,
    )
    assert len(errs) == 0
    assert len(cfgs) == 1
    assert isinstance(cfgs[0].objective, birdjepa.nn.objectives.LeJEPAConfig)
    assert cfgs[0].objective.proj_dim == 32


def test_full_pretrain_config():
    """Integration test with realistic pretrain config sweep."""
    sweep_dcts = [
        {
            "train_data": {
                "__class__": "Cifar100",
                "augmentations": [
                    {"__class__": "RandomCrop", "padding": 4},
                    {"__class__": "HorizontalFlip"},
                ],
            },
            "test_data": {"__class__": "Cifar100", "split": "test"},
            "model": {
                "input_h": 32,
                "input_w": 32,
                "patch_h": 4,
                "patch_w": 4,
                "embed_dim": 384,
                "depth": 12,
                "n_heads": 6,
            },
            "objective": {"__class__": "LeJEPAConfig", "proj_dim": 16},
            "batch_size": 256,
            "lr": 2e-3,
            "epochs": 100,
        }
    ]
    cfgs, errs = birdjepa.configs.load_cfgs(
        birdjepa.pretrain.Config(),
        default=birdjepa.pretrain.Config(),
        sweep_dcts=sweep_dcts,
    )
    assert len(errs) == 0, f"Errors: {errs}"
    assert len(cfgs) == 1
    cfg = cfgs[0]

    # Check data
    assert isinstance(cfg.train_data, birdjepa.data.Cifar100)
    assert cfg.train_data.split == "train"  # default
    assert len(cfg.train_data.augmentations) == 2
    assert isinstance(cfg.test_data, birdjepa.data.Cifar100)
    assert cfg.test_data.split == "test"

    # Check model
    assert cfg.model.input_h == 32
    assert cfg.model.embed_dim == 384
    assert cfg.model.n_heads == 6

    # Check objective
    assert isinstance(cfg.objective, birdjepa.nn.objectives.LeJEPAConfig)
    assert cfg.objective.proj_dim == 16

    # Check training params
    assert cfg.batch_size == 256
    assert cfg.lr == 2e-3
    assert cfg.epochs == 100


def test_multiple_sweep_configs():
    """Test sweep with multiple configurations."""
    sweep_dcts = [
        {"batch_size": 64, "lr": 1e-3},
        {"batch_size": 128, "lr": 2e-3},
        {"batch_size": 256, "lr": 4e-3},
    ]
    cfgs, errs = birdjepa.configs.load_cfgs(
        birdjepa.pretrain.Config(),
        default=birdjepa.pretrain.Config(),
        sweep_dcts=sweep_dcts,
    )
    assert len(errs) == 0
    assert len(cfgs) == 3
    assert cfgs[0].batch_size == 64
    assert cfgs[0].lr == 1e-3
    assert cfgs[1].batch_size == 128
    assert cfgs[1].lr == 2e-3
    assert cfgs[2].batch_size == 256
    assert cfgs[2].lr == 4e-3


# -----------------------------------------------------------------------------
# Tests for uncovered lines
# -----------------------------------------------------------------------------


def test_sweep_fully_filtered_by_override():
    """Test when CLI override filters all sweep fields (lines 257-258)."""
    # CLI overrides the same field that sweep sets
    override = birdjepa.pretrain.Config(batch_size=999)
    default = birdjepa.pretrain.Config()
    sweep_dcts = [{"batch_size": 64}, {"batch_size": 128}]

    cfgs, errs = birdjepa.configs.load_cfgs(
        override, default=default, sweep_dcts=sweep_dcts
    )

    # All sweep values filtered, so we get override config for each
    assert len(cfgs) == 2
    assert all(cfg.batch_size == 999 for cfg in cfgs)


def test_filter_overridden_with_dataclass_override():
    """Test _filter_overridden_fields with dataclass override (lines 124-141)."""
    # CLI sets nested dataclass, sweep tries to override same nested field
    override = birdjepa.pretrain.Config(
        model=birdjepa.nn.transformer.Config(embed_dim=512, depth=6)
    )
    default = birdjepa.pretrain.Config()
    sweep_dcts = [
        {"model": {"embed_dim": 256}, "batch_size": 64},
        {"model": {"embed_dim": 384}, "batch_size": 128},
    ]

    cfgs, errs = birdjepa.configs.load_cfgs(
        override, default=default, sweep_dcts=sweep_dcts
    )

    # CLI model.embed_dim and model.depth should win
    assert len(cfgs) == 2
    assert all(cfg.model.embed_dim == 512 for cfg in cfgs)
    assert all(cfg.model.depth == 6 for cfg in cfgs)
    # But batch_size from sweep should apply
    assert cfgs[0].batch_size == 64
    assert cfgs[1].batch_size == 128


def test_convert_value_plain_dataclass():
    """Test _convert_value with plain dataclass field type (line 171)."""

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Inner:
        x: int = 0

    value = {"x": 42}
    result = birdjepa.configs._convert_value(value, Inner)
    assert isinstance(result, Inner)
    assert result.x == 42


def test_convert_value_typing_union_path():
    """Test _convert_value with typing.Union style path (line 175)."""
    import typing

    # Python 3.9 style Union
    result = birdjepa.configs._convert_value(
        "/tmp/test", typing.Union[pathlib.Path, None]
    )
    assert result == pathlib.Path("/tmp/test")


def test_recursive_dataclass_update_non_dataclass():
    """Test _recursive_dataclass_update with non-dataclass (line 186)."""
    result = birdjepa.configs._recursive_dataclass_update(
        "not a dataclass", {"key": "value"}
    )
    assert result == {"key": "value"}


def test_recursive_dataclass_update_missing_attr():
    """Test _recursive_dataclass_update with missing attribute (lines 191-192)."""

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Small:
        x: int = 1

    obj = Small()
    # Try to update with a field that doesn't exist
    result = birdjepa.configs._recursive_dataclass_update(
        obj, {"nonexistent": 42, "x": 10}
    )
    # nonexistent should be passed through, x should be converted
    assert result["nonexistent"] == 42
    assert result["x"] == 10


def test_load_cfgs_error_handling():
    """Test error handling in load_cfgs (lines 263-264)."""

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class StrictConfig:
        value: int = 1

    override = StrictConfig()
    default = StrictConfig()
    # Pass invalid type that will fail dataclass construction
    sweep_dcts = [{"value": "not_an_int"}]

    cfgs, errs = birdjepa.configs.load_cfgs(
        override, default=default, sweep_dcts=sweep_dcts
    )

    # Should have error
    assert len(errs) > 0


def test_filter_with_dataclass_no_default_constructor():
    """Test _filter_overridden_fields when dataclass has no default (line 129-130)."""

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class RequiredFields:
        required: int  # No default!

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Outer:
        inner: RequiredFields = dataclasses.field(
            default_factory=lambda: RequiredFields(required=0)
        )
        value: int = 1

    override = Outer(inner=RequiredFields(required=99))
    default = Outer()

    # This tests line 129-130 where type(overridden_value)() raises TypeError
    diff = birdjepa.configs.get_non_default_values(override, default)
    assert "inner" in diff


def test_nested_override_partially_filters_sweep():
    """Test nested override only filters matching nested fields."""
    # CLI overrides only model.embed_dim
    override = birdjepa.pretrain.Config(
        model=birdjepa.nn.transformer.Config(embed_dim=512)
    )
    default = birdjepa.pretrain.Config()
    sweep_dcts = [
        {"model": {"embed_dim": 256, "depth": 6}},
        {"model": {"embed_dim": 384, "depth": 12}},
    ]

    cfgs, errs = birdjepa.configs.load_cfgs(
        override, default=default, sweep_dcts=sweep_dcts
    )

    # embed_dim from CLI, but depth from sweep
    assert len(cfgs) == 2
    assert all(cfg.model.embed_dim == 512 for cfg in cfgs)
    assert cfgs[0].model.depth == 6
    assert cfgs[1].model.depth == 12


# -----------------------------------------------------------------------------
# Tests for free wins sweeps
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "sweep_path",
        "optimizer",
        "use_rope",
        "use_qk_norm",
        "use_swiglu",
        "use_layerscale",
    ),
    [
        (
            pathlib.Path("sweeps/001_freewins/adamw_vits_xcl.py"),
            "adamw",
            False,
            False,
            False,
            False,
        ),
        (
            pathlib.Path("sweeps/001_freewins/muon_vits_xcl.py"),
            "muon",
            True,
            True,
            True,
            True,
        ),
    ],
)
def test_free_wins_sweep_loading(
    sweep_path: pathlib.Path,
    optimizer: str,
    use_rope: bool,
    use_qk_norm: bool,
    use_swiglu: bool,
    use_layerscale: bool,
):
    """Free wins sweep config loads correctly."""
    if not sweep_path.exists():
        pytest.skip("Sweep file not found")

    sweep = birdjepa.configs.load_sweep(sweep_path)

    assert len(sweep) > 0
    for cfg_dict in sweep:
        assert cfg_dict["model"]["use_rope"] is use_rope
        assert cfg_dict["model"]["use_qk_norm"] is use_qk_norm
        assert cfg_dict["model"]["use_swiglu"] is use_swiglu
        assert cfg_dict["model"]["use_layerscale"] is use_layerscale
        assert cfg_dict["optimizer"] == optimizer


@pytest.mark.parametrize(
    (
        "sweep_path",
        "optimizer",
        "use_rope",
        "use_qk_norm",
        "use_swiglu",
        "use_layerscale",
    ),
    [
        (
            pathlib.Path("sweeps/001_freewins/adamw_vits_xcl.py"),
            "adamw",
            False,
            False,
            False,
            False,
        ),
        (
            pathlib.Path("sweeps/001_freewins/muon_vits_xcl.py"),
            "muon",
            True,
            True,
            True,
            True,
        ),
    ],
)
def test_free_wins_config_instantiation_from_sweep(
    sweep_path: pathlib.Path,
    optimizer: str,
    use_rope: bool,
    use_qk_norm: bool,
    use_swiglu: bool,
    use_layerscale: bool,
):
    """Sweep dicts can be converted to Config objects."""
    if not sweep_path.exists():
        pytest.skip("Sweep file not found")

    sweep = birdjepa.configs.load_sweep(sweep_path)
    cfgs, errs = birdjepa.configs.load_cfgs(
        birdjepa.pretrain.Config(), default=birdjepa.pretrain.Config(), sweep_dcts=sweep
    )

    assert len(errs) == 0
    assert len(cfgs) == len(sweep)

    for cfg in cfgs:
        assert cfg.model.use_rope is use_rope
        assert cfg.model.use_qk_norm is use_qk_norm
        assert cfg.model.use_swiglu is use_swiglu
        assert cfg.model.use_layerscale is use_layerscale
        assert cfg.optimizer == optimizer
