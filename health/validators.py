from abc import ABC, abstractmethod
from typing import Any, Dict


class ValidationError(Exception):
    pass


class Validator(ABC):
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize data. Raises ValidationError on failure."""
        ...


class BloodPressureValidator(Validator):
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        systolic = data.get("systolic")
        diastolic = data.get("diastolic")
        if systolic is None or diastolic is None:
            raise ValidationError("Blood pressure requires both systolic and diastolic values")
        try:
            sys_val = float(systolic)
            dia_val = float(diastolic)
        except (TypeError, ValueError):
            raise ValidationError("Blood pressure values must be numeric")
        if not (60 <= sys_val <= 250):
            raise ValidationError(f"Systolic pressure {sys_val} out of range (60-250)")
        if not (40 <= dia_val <= 150):
            raise ValidationError(f"Diastolic pressure {dia_val} out of range (40-150)")
        if sys_val <= dia_val:
            raise ValidationError(f"Systolic ({sys_val}) must be greater than diastolic ({dia_val})")
        return {
            "systolic": sys_val,
            "diastolic": dia_val,
            "unit": data.get("unit", "mmHg")
        }


class HeartRateValidator(Validator):
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        value = data.get("value")
        if value is None:
            raise ValidationError("Heart rate requires a value")
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise ValidationError("Heart rate must be numeric")
        if not (30 <= val <= 220):
            raise ValidationError(f"Heart rate {val} out of range (30-220)")
        return {"value": val, "unit": data.get("unit", "bpm")}


class WeightValidator(Validator):
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        value = data.get("value")
        if value is None:
            raise ValidationError("Weight requires a value")
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise ValidationError("Weight must be numeric")
        if not (0 < val <= 500):
            raise ValidationError(f"Weight {val} out of range (0-500)")
        return {"value": val, "unit": data.get("unit", "kg")}


class SleepDurationValidator(Validator):
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        value = data.get("value")
        if value is None:
            raise ValidationError("Sleep duration requires a value")
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise ValidationError("Sleep duration must be numeric")
        if not (0 <= val <= 24):
            raise ValidationError(f"Sleep duration {val} out of range (0-24)")
        return {"value": val, "unit": data.get("unit", "hours")}


VALIDATOR_REGISTRY: Dict[str, Validator] = {
    "blood_pressure": BloodPressureValidator(),
    "heart_rate": HeartRateValidator(),
    "weight": WeightValidator(),
    "sleep_duration": SleepDurationValidator(),
}


def get_validator(sign_type: str) -> Validator:
    validator = VALIDATOR_REGISTRY.get(sign_type)
    if validator is None:
        raise ValidationError(f"Unknown vital sign type: {sign_type}")
    return validator
