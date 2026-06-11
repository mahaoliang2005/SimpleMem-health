import pytest
from health.validators import (
    BloodPressureValidator,
    HeartRateValidator,
    ValidationError,
    WeightValidator,
    get_validator,
)


class TestBloodPressureValidator:
    def test_valid_bp(self):
        v = BloodPressureValidator()
        result = v.validate({"systolic": 120, "diastolic": 80})
        assert result["systolic"] == 120.0
        assert result["diastolic"] == 80.0
        assert result["unit"] == "mmHg"

    def test_invalid_high_systolic(self):
        v = BloodPressureValidator()
        with pytest.raises(ValidationError, match=r"Systolic pressure 300\.?0? out of range"):
            v.validate({"systolic": 300, "diastolic": 80})

    def test_invalid_low_systolic(self):
        v = BloodPressureValidator()
        with pytest.raises(ValidationError, match=r"Systolic pressure 30\.?0? out of range"):
            v.validate({"systolic": 30, "diastolic": 80})

    def test_systolic_less_than_diastolic(self):
        v = BloodPressureValidator()
        with pytest.raises(ValidationError, match="must be greater than diastolic"):
            v.validate({"systolic": 80, "diastolic": 90})

    def test_missing_values(self):
        v = BloodPressureValidator()
        with pytest.raises(ValidationError, match="requires both"):
            v.validate({"systolic": 120})


class TestHeartRateValidator:
    def test_valid_hr(self):
        v = HeartRateValidator()
        result = v.validate({"value": 72})
        assert result["value"] == 72.0

    def test_invalid_high(self):
        v = HeartRateValidator()
        with pytest.raises(ValidationError, match="out of range"):
            v.validate({"value": 300})


class TestWeightValidator:
    def test_valid_weight(self):
        v = WeightValidator()
        result = v.validate({"value": 70.5})
        assert result["value"] == 70.5

    def test_negative_weight(self):
        v = WeightValidator()
        with pytest.raises(ValidationError, match="out of range"):
            v.validate({"value": -10})


class TestGetValidator:
    def test_known_type(self):
        v = get_validator("blood_pressure")
        assert isinstance(v, BloodPressureValidator)

    def test_unknown_type(self):
        with pytest.raises(ValidationError, match="Unknown vital sign type"):
            get_validator("unknown_type")
