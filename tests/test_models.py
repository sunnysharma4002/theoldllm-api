import pytest

from theoldllm.exceptions import UnknownModelError
from theoldllm.models import MODELS, VERIFIED_ONLY, Models, Provider


def test_catalogue_matches_the_ids_the_site_ships():
    assert [m.id for m in MODELS] == [
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-6-astra",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5-thinking",
        "gpt-5-mini-aichat",
        "claude-opus-4-7",
        "gemini-3.1-pro",
        "gemini-3.8-flash",
        "grok-4.7-latest",
        "grok-4.5-latest",
        "grok-4.3-latest",
    ]


def test_verified_only_matches_the_requires_verified_account_flag():
    assert VERIFIED_ONLY == {
        "gpt-6-sol",
        "gpt-6-astra",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "claude-opus-4-7",
        "gemini-3.1-pro",
        "grok-4.7-latest",
    }


def test_anonymous_models_are_the_ones_verified_without_an_account():
    anonymous = {m.id for m in Models.anonymous()}
    assert "gpt-6-luna" in anonymous
    assert "gpt-6-sol" not in anonymous
    assert "gpt-6-astra" not in anonymous  # force-disabled upstream


def test_picker_hides_hidden_and_disabled_models():
    picker = {m.id for m in Models.picker()}
    assert "gpt-5-mini-aichat" not in picker
    assert "gpt-6-astra" not in picker
    assert "gpt-6-luna" in picker


def test_by_provider_groups_models():
    assert {m.id for m in Models.by_provider(Provider.XAI)} == {
        "grok-4.7-latest",
        "grok-4.5-latest",
        "grok-4.3-latest",
    }
    assert {m.id for m in Models.by_provider("google")} == {"gemini-3.1-pro", "gemini-3.8-flash"}


def test_unknown_model_raises_by_default_and_can_be_optional():
    with pytest.raises(UnknownModelError):
        Models.by_id("gpt-4o")
    assert Models.by_id("gpt-4o", strict=False) is None


def test_deprecated_ids_resolve_with_a_warning():
    with pytest.warns(DeprecationWarning) as record:
        model = Models.by_id("claude-sonnet-5")
    assert model is not None and model.id == "claude-opus-4-7"
    assert "claude-opus-4-7" in str(record[0].message)


def test_deprecated_alias_chain_is_not_infinite():
    with pytest.warns(DeprecationWarning):
        assert Models.by_id("grok-4.2-latest").id == "grok-4.3-latest"
