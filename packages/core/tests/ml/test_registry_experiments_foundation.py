import json

from quant_os_core.ml.experiments import JsonlExperimentTracker
from quant_os_core.ml.foundation import foundation_model_catalog
from quant_os_core.ml.registry import ModelFamily, ModelRegistry


def test_registry_versions_increment():
    registry = ModelRegistry()
    one = registry.register("alpha", ModelFamily.LINEAR)
    two = registry.register("alpha", ModelFamily.BOOSTING)
    assert one.version == 1
    assert two.version == 2
    assert registry.latest("alpha") == two
    assert len(registry.versions("alpha")) == 2


def test_jsonl_tracker(tmp_path):
    path = tmp_path / "experiments.jsonl"
    tracker = JsonlExperimentTracker(path)
    tracker.log("exp-1", "ridge", metrics={"mse": 0.1}, parameters={"alpha": 1})
    parsed = json.loads(path.read_text().strip())
    assert parsed["experiment_id"] == "exp-1"
    assert parsed["metrics"]["mse"] == 0.1


def test_foundation_catalog_is_lazy_and_descriptive():
    specs = foundation_model_catalog()
    ids = {item.model_id for item in specs}
    assert {"amazon_chronos", "google_timesfm", "kronos_financial"} <= ids
    assert all(item.notes for item in specs)
