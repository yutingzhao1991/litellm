from typing import Any

import litellm


def is_deterministic_tool_schema_error(error: Exception) -> bool:
    """Return whether a provider rejected the request's function-tool schema.

    These 400 responses are determined entirely by the submitted schema. Retrying
    the same deployment or falling back to another model can only duplicate work
    or hide a client compatibility bug; the original error should be returned.
    """

    if not isinstance(error, litellm.BadRequestError):
        return False

    candidates = [getattr(error, "message", None), str(error)]
    body: Any = getattr(error, "body", None)
    if isinstance(body, dict):
        candidates.extend(
            [
                body.get("code"),
                body.get("message"),
                body.get("error"),
            ]
        )
    normalized = "\n".join(str(value).lower() for value in candidates if value)
    if "invalid_function_parameters" in normalized:
        return True
    if "invalid schema for function" in normalized:
        return True
    return (
        "function_declarations" in normalized
        and "parameters" in normalized
        and "property is not defined" in normalized
    )
