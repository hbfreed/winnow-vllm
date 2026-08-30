import winnow_vllm


def test_registers_only_winnow_architectures(monkeypatch):
    registered = []

    def record(name, target):
        registered.append((name, target))

    monkeypatch.setattr(winnow_vllm, "version", lambda _name: "0.25.0")
    monkeypatch.setattr("vllm.ModelRegistry.register_model", record)
    winnow_vllm.register()
    assert {name for name, _target in registered} == {
        "WinnowOlmoeForCausalLM",
        "WinnowQwen3_5MoeForCausalLM",
        "WinnowAfmoeForCausalLM",
    }


def test_installed_entry_point_registers_real_model_classes():
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    config = AutoConfig.for_model(
        "winnow_olmoe",
        expert_widths=[[128, 128]],
        expert_indices=[[0, 1]],
    )
    assert config.__class__.__name__ == "WinnowOlmoeConfig"
    model_class = ModelRegistry.models["WinnowOlmoeForCausalLM"].load_model_cls()
    assert model_class.__module__ == "winnow_vllm.olmoe"
