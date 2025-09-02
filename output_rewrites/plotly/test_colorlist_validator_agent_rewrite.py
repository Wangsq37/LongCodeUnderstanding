import pytest
import numpy as np

from _plotly_utils.basevalidators import ColorlistValidator


# Fixtures
# --------
@pytest.fixture()
def validator():
    return ColorlistValidator("prop", "parent")


# Rejection
# ---------
@pytest.mark.parametrize("val", [set(), 23, 0.5, {}, "redd"])
def test_rejection_value(validator, val):
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


@pytest.mark.parametrize("val", [[set()], [23, 0.5], [{}, "red"], ["blue", "redd"]])
def test_rejection_element(validator, val):
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)


# Acceptance
# ----------
@pytest.mark.parametrize(
    "val",
    [
        ["black", "white", "yellow"],  # multiple valid color strings
        ["rgb(0,0,0)", "rgba(255,255,255,0.7)", "rgb(127,127,127)"],  # rgb and rgba formats
        np.array(["hsl(200,50%,50%)", "hsla(210,60%,60%,0.5)", "hsv(20,100%,80%)"]),  # hsl, hsla, hsv as array
        ["hsv(360, 100%, 100%)", "hsva(180, 60%, 80%, 0.8)"],  # full gamut hsv and hsva
        ["rgb(0,0,0)", "hsl(0, 0%, 0%)", "blue"],  # mixture of different valid formats
        np.array(["red", "blue", "green", "yellow"]),  # multiple color strings array
        [],  # empty list (edge case: does ColorlistValidator accept this as valid?)
    ],
)
def test_acceptance_aok(val, validator):
    coerce_val = validator.validate_coerce(val)
    assert isinstance(coerce_val, list)
    assert validator.present(coerce_val) == tuple(val)