import re
from typing import Any

# Matches api_key, OPENAI_API_KEY, hf_token, client_secret, password; not max_tokens.
SECRET_KEY_PATTERN = re.compile(r"(api[_-]?key|secret|password|(^|[_-])token)$", re.IGNORECASE)
REDACTED = "<redacted>"


def redact_secrets(obj: Any) -> Any:
    """Return a copy of obj with values under secret-looking keys replaced by REDACTED.

    Used before writing run arguments/configs (e.g., args.json) to disk so API keys are never
    persisted in plaintext.
    """
    if isinstance(obj, dict):
        return {
            k: (REDACTED if isinstance(k, str) and SECRET_KEY_PATTERN.search(k) and v is not None
                else redact_secrets(v))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_secrets(v) for v in obj)
    return obj
