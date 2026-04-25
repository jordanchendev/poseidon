"""Tests for Model Engine — Phase 3.

Tests cover: BaseModel ABC, registry, lifecycle state machine,
artifact storage, ModelManager, and XGBoost implementation.
"""

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from poseidon.ml.base import BaseModel
from poseidon.ml.lifecycle import (
    ALL_STATES,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    InvalidTransitionError,
    validate_transition,
)
from poseidon.ml.registry import _registry, get_model, list_models, register_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class DummyModel(BaseModel):
    """Minimal concrete model for testing the ABC."""

    name = "dummy"
    description = "Test model"

    def __init__(self):
        self._trained = False
        self._data = None

    def train(self, features, targets, params):
        self._trained = True
        self._data = {"n": len(features)}
        return {"accuracy": 0.8, "n_samples": len(features)}

    def predict(self, features):
        return pd.DataFrame(
            {"prediction": ["hold"] * len(features), "confidence": [0.5] * len(features)},
            index=features.index,
        )

    def validate(self, features, targets):
        return {"accuracy": 0.75}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "dummy.json").write_text(json.dumps(self._data or {}))

    @classmethod
    def load(cls, path):
        instance = cls()
        data_file = Path(path) / "dummy.json"
        if data_file.exists():
            instance._data = json.loads(data_file.read_text())
            instance._trained = True
        return instance

    def get_default_params(self):
        return {"n_estimators": 10}

    def get_feature_list(self):
        return ["sma_20", "rsi_14"]


@pytest.fixture
def sample_features():
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "sma_20": np.random.randn(n),
            "rsi_14": np.random.rand(n) * 100,
            "close": 100 + np.cumsum(np.random.randn(n)),
        }
    )


@pytest.fixture
def sample_targets():
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1, 2], 50))


# ---------------------------------------------------------------------------
# BaseModel ABC Tests
# ---------------------------------------------------------------------------


class TestBaseModelABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseModel()

    def test_concrete_model_instantiates(self):
        m = DummyModel()
        assert m.name == "dummy"

    def test_train_returns_metrics(self, sample_features, sample_targets):
        m = DummyModel()
        metrics = m.train(sample_features, sample_targets, {})
        assert "accuracy" in metrics
        assert "n_samples" in metrics

    def test_predict_returns_correct_columns(self, sample_features):
        m = DummyModel()
        result = m.predict(sample_features)
        assert "prediction" in result.columns
        assert "confidence" in result.columns
        assert len(result) == len(sample_features)

    def test_validate_returns_metrics(self, sample_features, sample_targets):
        m = DummyModel()
        metrics = m.validate(sample_features, sample_targets)
        assert isinstance(metrics, dict)

    def test_save_and_load(self, tmp_path, sample_features, sample_targets):
        m = DummyModel()
        m.train(sample_features, sample_targets, {})
        m.save(tmp_path / "model")

        loaded = DummyModel.load(tmp_path / "model")
        assert loaded._trained is True
        assert loaded._data == {"n": len(sample_features)}

    def test_get_default_params(self):
        m = DummyModel()
        params = m.get_default_params()
        assert isinstance(params, dict)

    def test_get_feature_list(self):
        m = DummyModel()
        features = m.get_feature_list()
        assert isinstance(features, list)
        assert len(features) > 0


# ---------------------------------------------------------------------------
# Registry Tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_model_decorator(self):
        # DummyModel is not registered via decorator, register it manually
        @register_model
        class TestModel(BaseModel):
            name = "test_registry_model"
            description = "test"

            def train(self, f, t, p):
                return {}

            def predict(self, f):
                return pd.DataFrame()

            def validate(self, f, t):
                return {}

            def save(self, p):
                pass

            @classmethod
            def load(cls, p):
                return cls()

            def get_default_params(self):
                return {}

            def get_feature_list(self):
                return []

        assert "test_registry_model" in _registry
        assert get_model("test_registry_model") is TestModel

        # Cleanup
        del _registry["test_registry_model"]

    def test_get_unknown_model_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent_model")

    def test_register_without_name_raises(self):
        with pytest.raises(ValueError, match="must define a 'name'"):

            @register_model
            class BadModel(BaseModel):
                name = ""

                def train(self, f, t, p):
                    return {}

                def predict(self, f):
                    return pd.DataFrame()

                def validate(self, f, t):
                    return {}

                def save(self, p):
                    pass

                @classmethod
                def load(cls, p):
                    return cls()

                def get_default_params(self):
                    return {}

                def get_feature_list(self):
                    return []

    def test_xgboost_registered(self):
        """XGBoost model should be auto-registered via implementations import."""
        assert "xgboost" in list_models()

    def test_list_models(self):
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0


# ---------------------------------------------------------------------------
# Lifecycle Tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_all_states_defined(self):
        assert {"training", "ready", "shadow", "active", "failed", "retired"} == ALL_STATES

    def test_terminal_states(self):
        assert {"failed", "retired"} == TERMINAL_STATES

    def test_valid_transitions(self):
        # training -> ready
        validate_transition("training", "ready")
        # training -> failed
        validate_transition("training", "failed")
        # ready -> shadow
        validate_transition("ready", "shadow")
        # ready -> active (skip shadow)
        validate_transition("ready", "active")
        # shadow -> active
        validate_transition("shadow", "active")
        # active -> retired
        validate_transition("active", "retired")

    def test_invalid_transition_raises(self):
        with pytest.raises(InvalidTransitionError):
            validate_transition("training", "active")

    def test_failed_is_terminal(self):
        with pytest.raises(InvalidTransitionError):
            validate_transition("failed", "ready")

    def test_retired_is_terminal(self):
        with pytest.raises(InvalidTransitionError):
            validate_transition("retired", "active")

    def test_unknown_state_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown state"):
            validate_transition("unknown", "ready")
        with pytest.raises(ValueError, match="Unknown state"):
            validate_transition("training", "unknown")

    def test_cannot_go_backwards(self):
        with pytest.raises(InvalidTransitionError):
            validate_transition("active", "shadow")
        with pytest.raises(InvalidTransitionError):
            validate_transition("active", "ready")
        with pytest.raises(InvalidTransitionError):
            validate_transition("shadow", "ready")

    def test_all_transitions_covered(self):
        """Every state has an entry in VALID_TRANSITIONS."""
        for state in ALL_STATES:
            assert state in VALID_TRANSITIONS


# ---------------------------------------------------------------------------
# Artifacts Tests
# ---------------------------------------------------------------------------


class TestArtifacts:
    def test_get_version_dir(self, tmp_path):
        with patch("poseidon.ml.artifacts.settings") as mock_settings:
            mock_settings.model_artifact_dir = str(tmp_path)
            from poseidon.ml.artifacts import get_version_dir

            path = get_version_dir("my_model", 3)
            assert path == tmp_path / "my_model" / "v3"

    def test_ensure_version_dir_creates(self, tmp_path):
        with patch("poseidon.ml.artifacts.settings") as mock_settings:
            mock_settings.model_artifact_dir = str(tmp_path)
            from poseidon.ml.artifacts import ensure_version_dir

            path = ensure_version_dir("my_model", 1)
            assert path.exists()
            assert path.is_dir()

    def test_save_and_load_metadata(self, tmp_path):
        with patch("poseidon.ml.artifacts.settings") as mock_settings:
            mock_settings.model_artifact_dir = str(tmp_path)
            from poseidon.ml.artifacts import ensure_version_dir, load_metadata, save_metadata

            ensure_version_dir("my_model", 1)
            save_metadata("my_model", 1, {"sharpe": 1.5, "win_rate": 0.6})
            meta = load_metadata("my_model", 1)
            assert meta["sharpe"] == 1.5
            assert meta["win_rate"] == 0.6

    def test_load_metadata_missing_returns_empty(self, tmp_path):
        with patch("poseidon.ml.artifacts.settings") as mock_settings:
            mock_settings.model_artifact_dir = str(tmp_path)
            from poseidon.ml.artifacts import load_metadata

            meta = load_metadata("nonexistent", 99)
            assert meta == {}

    def test_active_symlink(self, tmp_path):
        with patch("poseidon.ml.artifacts.settings") as mock_settings:
            mock_settings.model_artifact_dir = str(tmp_path)
            from poseidon.ml.artifacts import (
                ensure_version_dir,
                get_active_version,
                set_active_symlink,
            )

            ensure_version_dir("my_model", 1)
            ensure_version_dir("my_model", 2)

            set_active_symlink("my_model", 1)
            assert get_active_version("my_model") == 1

            # Switch to v2
            set_active_symlink("my_model", 2)
            assert get_active_version("my_model") == 2

    def test_active_symlink_nonexistent_raises(self, tmp_path):
        with patch("poseidon.ml.artifacts.settings") as mock_settings:
            mock_settings.model_artifact_dir = str(tmp_path)
            from poseidon.ml.artifacts import set_active_symlink

            with pytest.raises(FileNotFoundError):
                set_active_symlink("my_model", 99)

    def test_get_active_version_no_symlink(self, tmp_path):
        with patch("poseidon.ml.artifacts.settings") as mock_settings:
            mock_settings.model_artifact_dir = str(tmp_path)
            from poseidon.ml.artifacts import get_active_version

            assert get_active_version("no_model") is None


# ---------------------------------------------------------------------------
# ModelManager Tests (with mocked DB session)
# ---------------------------------------------------------------------------


class TestModelManager:
    def _make_manager(self):
        """Create a ModelManager with a mocked session."""
        from poseidon.ml.manager import ModelManager

        session = MagicMock()
        return ModelManager(session), session

    def test_create_version(self, tmp_path):
        from poseidon.ml.manager import ModelManager

        session = MagicMock()
        # Mock: no existing versions
        session.execute.return_value.scalar.return_value = None

        with patch("poseidon.ml.manager.artifacts.ensure_version_dir") as mock_ensure:
            mock_ensure.return_value = tmp_path / "test_model" / "v1"
            mgr = ModelManager(session)
            mgr.create_version("test_model", {"lr": 0.1}, ["sma_20"])

        session.add.assert_called_once()
        session.commit.assert_called_once()
        added = session.add.call_args[0][0]
        assert added.name == "test_model"
        assert added.version == 1
        assert added.status == "training"

    def test_create_version_increments(self, tmp_path):
        from poseidon.ml.manager import ModelManager

        session = MagicMock()
        # Mock: existing max version = 3
        session.execute.return_value.scalar.return_value = 3

        with patch("poseidon.ml.manager.artifacts.ensure_version_dir") as mock_ensure:
            mock_ensure.return_value = tmp_path / "test_model" / "v4"
            mgr = ModelManager(session)
            mgr.create_version("test_model", {}, [])

        added = session.add.call_args[0][0]
        assert added.version == 4

    def test_transition_valid(self):
        from poseidon.ml.manager import ModelManager
        from poseidon.models.model_version import ModelVersion

        session = MagicMock()
        mv = ModelVersion()
        mv.id = uuid.uuid4()
        mv.name = "test"
        mv.version = 1
        mv.status = "training"
        session.get.return_value = mv

        mgr = ModelManager(session)
        result = mgr.transition(mv.id, "ready", metrics={"accuracy": 0.9})
        assert result.status == "ready"
        assert result.metrics == {"accuracy": 0.9}

    def test_transition_invalid_raises(self):
        from poseidon.ml.manager import ModelManager
        from poseidon.models.model_version import ModelVersion

        session = MagicMock()
        mv = ModelVersion()
        mv.id = uuid.uuid4()
        mv.name = "test"
        mv.version = 1
        mv.status = "training"
        session.get.return_value = mv

        mgr = ModelManager(session)
        with pytest.raises(InvalidTransitionError):
            mgr.transition(mv.id, "active")

    def test_transition_not_found_raises(self):
        from poseidon.ml.manager import ModelManager

        session = MagicMock()
        session.get.return_value = None

        mgr = ModelManager(session)
        with pytest.raises(ValueError, match="not found"):
            mgr.transition(uuid.uuid4(), "ready")

    def test_transition_to_active_retires_previous(self):
        from poseidon.ml.manager import ModelManager
        from poseidon.models.model_version import ModelVersion

        session = MagicMock()

        # The version being activated
        mv = ModelVersion()
        mv.id = uuid.uuid4()
        mv.name = "test"
        mv.version = 2
        mv.status = "ready"
        session.get.return_value = mv

        # Previously active version
        old_active = ModelVersion()
        old_active.id = uuid.uuid4()
        old_active.name = "test"
        old_active.version = 1
        old_active.status = "active"
        session.execute.return_value.scalars.return_value.all.return_value = [old_active]

        with patch("poseidon.ml.manager.artifacts.set_active_symlink"):
            mgr = ModelManager(session)
            mgr.transition(mv.id, "active")

        assert mv.status == "active"
        assert old_active.status == "retired"

    def test_transition_to_failed_with_error(self):
        from poseidon.ml.manager import ModelManager
        from poseidon.models.model_version import ModelVersion

        session = MagicMock()
        mv = ModelVersion()
        mv.id = uuid.uuid4()
        mv.name = "test"
        mv.version = 1
        mv.status = "training"
        session.get.return_value = mv

        mgr = ModelManager(session)
        result = mgr.transition(mv.id, "failed", error_message="OOM during training")
        assert result.status == "failed"
        assert result.error_message == "OOM during training"
