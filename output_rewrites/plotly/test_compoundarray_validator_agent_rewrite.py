import pytest
from _plotly_utils.basevalidators import CompoundArrayValidator
from plotly.graph_objs.layout import Image


# Fixtures
# --------
@pytest.fixture()
def validator():
    return CompoundArrayValidator(
        "prop", "layout", data_class_str="Image", data_docs=""
    )


# Tests
# -----
def test_acceptance(validator):
    val = [
        Image(opacity=0, sizex=-999999999),
        Image(x=0, opacity=1.0, sizex=1000000000, sizey=500.5),
        Image()
    ]
    res = validator.validate_coerce(val)
    res_present = validator.present(res)
    assert isinstance(res, list)
    assert isinstance(res_present, tuple)
    # First Image: test zero opacity, huge negative sizex, no x
    assert isinstance(res_present[0], Image)
    assert res_present[0].opacity == 0
    assert res_present[0].sizex == -999999999
    assert res_present[0].x is None

    # Second Image: test zero x, max sizex, float sizey
    assert isinstance(res_present[1], Image)
    assert res_present[1].opacity == 1.0
    assert res_present[1].sizex == 1000000000
    assert res_present[1].x == 0
    assert res_present[1].sizey == 500.5

    # Third Image: empty input
    assert isinstance(res_present[2], Image)
    assert res_present[2].opacity is None
    assert res_present[2].sizex is None
    assert res_present[2].x is None
    assert res_present[2].sizey is None


def test_acceptance_empty(validator):
    val = [{}]
    res = validator.validate_coerce(val)
    res_present = validator.present(res)

    assert isinstance(res, list)
    assert isinstance(res_present, tuple)
    assert isinstance(res_present[0], Image)
    assert res_present[0].opacity is None
    assert res_present[0].sizex is None
    assert res_present[0].x is None


def test_acceptance_dict(validator):
    val = [
        dict(opacity=0, sizex=-123456789, sizey=1.1),
        dict(x=0, sizex=10000, opacity=0.25),
        dict()
    ]
    res = validator.validate_coerce(val)
    res_present = validator.present(res)

    assert isinstance(res, list)
    assert isinstance(res_present, tuple)

    # First dict -> Image
    assert isinstance(res_present[0], Image)
    assert res_present[0].opacity == 0
    assert res_present[0].sizex == -123456789
    assert res_present[0].sizey == 1.1
    assert res_present[0].x is None

    # Second dict -> Image
    assert isinstance(res_present[1], Image)
    assert res_present[1].x == 0
    assert res_present[1].sizex == 10000
    assert res_present[1].opacity == 0.25

    # Third dict: empty
    assert isinstance(res_present[2], Image)
    assert res_present[2].opacity is None
    assert res_present[2].sizex is None
    assert res_present[2].x is None
    assert res_present[2].sizey is None


def test_rejection_type(validator):
    val = 37

    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid value" in str(validation_failure.value)


def test_rejection_element(validator):
    val = ["a", 37]

    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert "Invalid element(s)" in str(validation_failure.value)


def test_rejection_value(validator):
    val = [dict(opacity=0.5, sizex=120, bogus=100)]

    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)

    assert (
        "Invalid property specified for object of type "
        "plotly.graph_objs.layout.Image" in str(validation_failure.value)
    )


def test_skip_invalid(validator):
    val = [
        dict(opacity="", x=-999999, sizex=0, sizey=22),
        dict(x=0, bogus={"a": 0}, sizey=0, opacity=-1),
        dict(x=1.5, sizex="invalidsizex", opacity=None),
    ]

    expected = [
        {'sizex': 0, 'x': -999999, 'sizey': 22},
        {'sizey': 0, 'x': 0},
        {'x': 1.5},
    ]

    res = validator.validate_coerce(val, skip_invalid=True)
    assert [el.to_plotly_json() for el in res] == expected