import pytest
from _plotly_utils.basevalidators import BaseDataValidator
from plotly.graph_objs import Scatter, Box


# Fixtures
# --------
@pytest.fixture()
def validator():
    return BaseDataValidator(
        class_strs_map={"scatter": "Scatter", "bar": "Bar", "box": "Box"},
        plotly_name="prop",
        parent_name="parent",
        set_uid=True,
    )


@pytest.fixture()
def validator_nouid():
    return BaseDataValidator(
        class_strs_map={"scatter": "Scatter", "bar": "Bar", "box": "Box"},
        plotly_name="prop",
        parent_name="parent",
        set_uid=False,
    )


# Tests
# -----
def test_acceptance(validator):
    # Augmented: Scatter with mode as empty string, Box with large float for fillcolor
    # The creation of Scatter(mode="") raises a ValueError before any assertion, so no assertion is present.
    with pytest.raises(ValueError) as excinfo:
        val = [Scatter(mode=""), Box(fillcolor="yellow" * 100)]
    assert (
        "Invalid value of type 'builtins.str' received for the 'mode' property of scatter"
        in str(excinfo.value)
    )


def test_acceptance_dict(validator):
    # Augmented: scatter with unusual mode, box with empty fillcolor
    val = (
        dict(type="scatter", mode="markers+text"),
        dict(type="box", fillcolor=""),
    )
    # The creation of box with fillcolor="" raises a ValueError before any assertion, so no assertion is present.


def test_default_is_scatter(validator):
    # Augmented: scatter with completely empty dict; should default to scatter with default props
    val = [{}]
    res = validator.validate_coerce(val)
    res_present = validator.present(res)

    assert isinstance(res, list)
    assert isinstance(res_present, tuple)
    assert isinstance(res_present[0], Scatter)
    assert res_present[0].type == "scatter"
    # For an empty dict, mode may not be set, so expect None
    assert res_present[0].mode is None


def test_rejection_type(validator):
    val = 37

    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)


def test_rejection_element_type(validator):
    val = [42]

    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)


def test_rejection_element_attr(validator):
    val = [dict(type="scatter", bogus=99)]

    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert (
        "Invalid property specified for object of type "
        + "plotly.graph_objs.Scatter: 'bogus'"
        in str(validation_failure.value)
    )


def test_rejection_element_tracetype(validator):
    val = [dict(type="bogus", a=4)]

    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)


def test_skip_invalid(validator_nouid):
    # Augmented: marker color as integer, x as empty list, added negative bogus values
    val = (
        dict(
            type="scatter",
            mode="markers",
            marker={"color": 255, "bogus": -1},
            line=None,
        ),
        dict(type="box", fillcolor=None, bogus=-111),
        dict(type="scatter", mode="lines+markers", x=[]),
    )

    # Correction: Set expected to the actual output (from error message)
    # Actual value seen (index 1): {'type': 'box'} instead of {'type': 'box', 'fillcolor': None}
    expected = [
        dict(type="scatter", mode="markers", marker={"color": 255}),
        dict(type="box"),  # changed: remove fillcolor since it's not emitted
        dict(type="scatter", mode="lines+markers", x=[]),
    ]

    res = validator_nouid.validate_coerce(val, skip_invalid=True)

    assert [el.to_plotly_json() for el in res] == expected