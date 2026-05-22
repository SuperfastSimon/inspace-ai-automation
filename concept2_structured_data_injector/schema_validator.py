from __future__ import annotations
from dataclasses import dataclass, field
import json
import requests
from logger import get_logger

log = get_logger("schema_validator")

REQUIRED_PROPS = {
    "Article": ["headline", "author", "datePublished"],
    "Product": ["name"],
    "FAQPage": ["mainEntity"],
}

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SchemaValidator:
    def validate(self, schema: dict) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        schema_type = schema.get("@type", "")

        if not schema.get("@context"):
            errors.append("Missing @context")
        if not schema_type:
            errors.append("Missing @type")

        required = REQUIRED_PROPS.get(schema_type, [])
        for prop in required:
            if prop not in schema:
                warnings.append(f"Recommended property '{prop}' missing for {schema_type}")

        if errors:
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        try:
            resp = requests.post(
                "https://validator.schema.org/validate",
                data={"html": f'<script type="application/ld+json">{json.dumps(schema)}</script>'},
                timeout=10,
            )
            if resp.status_code == 200:
                result = resp.json()
                remote_errors = [e["message"] for e in result.get("errors", [])]
                errors.extend(remote_errors)
        except Exception as exc:
            log.warning("Remote validation unavailable: %s", exc)

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
