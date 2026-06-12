from typing import Any, Dict, Optional

from health.validators import ValidationError, get_validator
from main import SimpleMemSystem


class HealthDataInput:
    def __init__(self, simplemem: SimpleMemSystem):
        self.simplemem = simplemem

    async def record(
        self,
        user_id: str,
        sign_type: str,
        data: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> dict:
        """
        Validate structured health data, convert to natural language,
        and store via SimpleMem.
        """
        try:
            validator = get_validator(sign_type)
            validated = validator.validate(data)
        except ValidationError as e:
            return {"status": "error", "message": str(e)}

        # Naturalize to text
        natural_text = self._naturalize(sign_type, validated, user_id)

        # Store via SimpleMem
        self.simplemem.add_dialogue(
            speaker="user",
            content=natural_text,
            timestamp=timestamp
        )

        return {"status": "ok", "sign_type": sign_type}

    def _naturalize(self, sign_type: str, data: Dict[str, Any], user_id: str) -> str:
        """Convert validated data to a natural language sentence."""
        if sign_type == "blood_pressure":
            return (
                f"User {user_id} recorded blood pressure: "
                f"{data['systolic']}/{data['diastolic']} {data.get('unit', 'mmHg')}."
            )
        elif sign_type == "heart_rate":
            return (
                f"User {user_id} recorded heart rate: "
                f"{data['value']} {data.get('unit', 'bpm')}."
            )
        elif sign_type == "weight":
            return (
                f"User {user_id} recorded weight: "
                f"{data['value']} {data.get('unit', 'kg')}."
            )
        elif sign_type == "sleep_duration":
            return (
                f"User {user_id} recorded sleep duration: "
                f"{data['value']} {data.get('unit', 'hours')}."
            )
        else:
            return f"User {user_id} recorded {sign_type}: {data}."
