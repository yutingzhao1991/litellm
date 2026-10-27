"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```

When the remote fetch fails, the last successfully fetched map (cached on disk) is used
before falling back to the backup bundled with the package. The bundled backup only knows
the models that existed at build time, so falling back to it silently prices any model
added later at 0. Cache location can be overridden with
`LITELLM_MODEL_COST_MAP_CACHE_PATH` (exact file) or `LITELLM_MODEL_COST_MAP_CACHE_DIR`
(directory; the file name is derived from the URL), defaulting to the system temp dir.
"""

import hashlib
import json
import os
import tempfile
from importlib.resources import files
from typing import Optional

import httpx

from litellm import verbose_logger
from litellm.constants import (
    MODEL_COST_MAP_MAX_SHRINK_RATIO,
    MODEL_COST_MAP_MIN_MODEL_COUNT,
)

COST_MAP_CACHE_PATH_ENV = "LITELLM_MODEL_COST_MAP_CACHE_PATH"
COST_MAP_CACHE_DIR_ENV = "LITELLM_MODEL_COST_MAP_CACHE_DIR"
COST_MAP_CACHE_FILE_PREFIX = "litellm_model_cost_map_last_good"


class GetModelCostMap:
    """
    Handles fetching, validating, and loading the model cost map.

    Only the backup model *count* is cached (a single int). The full
    backup dict is never held in memory — it is only parsed when it
    needs to be *returned* as a fallback.
    """

    _backup_model_count: int = -1  # -1 = not yet loaded

    @staticmethod
    def load_local_model_cost_map() -> dict:
        """Load the local backup model cost map bundled with the package."""
        content = json.loads(
            files("litellm")
            .joinpath("model_prices_and_context_window_backup.json")
            .read_text(encoding="utf-8")
        )
        return content

    @classmethod
    def _get_backup_model_count(cls) -> int:
        """Return the number of models in the local backup (cached int)."""
        if cls._backup_model_count < 0:
            backup = cls.load_local_model_cost_map()
            cls._backup_model_count = len(backup)
        return cls._backup_model_count

    @staticmethod
    def _cache_path(url: Optional[str]) -> str:
        """Resolve the path used to persist the last successful remote map."""
        explicit_path = os.getenv(COST_MAP_CACHE_PATH_ENV)
        if explicit_path:
            return explicit_path

        cache_dir = os.getenv(COST_MAP_CACHE_DIR_ENV) or tempfile.gettempdir()
        digest = hashlib.sha256((url or "").encode("utf-8")).hexdigest()[:12]
        return os.path.join(
            cache_dir, f"{COST_MAP_CACHE_FILE_PREFIX}_{digest}.json"
        )

    @classmethod
    def _write_last_good_cache(cls, url: Optional[str], content: dict) -> None:
        """
        Persist a successfully fetched + validated map (best effort).

        Never raises: a read-only filesystem must not break startup or a reload.
        """
        path = cls._cache_path(url)
        try:
            directory = os.path.dirname(path) or "."
            os.makedirs(directory, exist_ok=True)
            file_descriptor, tmp_path = tempfile.mkstemp(
                dir=directory, prefix=".model_cost_map_", suffix=".json"
            )
            try:
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                    json.dump(content, handle)
                os.replace(tmp_path, path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            verbose_logger.warning(
                "LiteLLM: Failed to write model cost map cache at %s: %s. "
                "Falling back to the bundled backup will be less accurate.",
                path,
                str(e),
            )

    @classmethod
    def _load_last_good_cache(cls, url: Optional[str]) -> Optional[dict]:
        """Read the persisted last known good map, or None if unusable."""
        path = cls._cache_path(url)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                content = json.load(handle)
        except FileNotFoundError:
            return None
        except Exception as e:
            verbose_logger.error(
                "LiteLLM: Failed to read model cost map cache at %s: %s. Ignoring cache.",
                path,
                str(e),
            )
            return None

        if not cls._check_is_valid_dict(content):
            return None

        return content

    @classmethod
    def _load_fallback_map(cls, url: Optional[str], reason: str) -> dict:
        """
        Build the fallback map when the remote fetch failed.

        Prefers the last known good remote snapshot over the bundled backup:
        the bundled backup predates any model added while the gateway is running,
        so it silently prices those models at 0.
        """
        cached = cls._load_last_good_cache(url)
        if cached is not None and cls.validate_model_cost_map(
            fetched_map=cached,
            backup_model_count=cls._get_backup_model_count(),
        ):
            verbose_logger.error(
                "LiteLLM: Remote model cost map unavailable (%s). "
                "Using last known good cached map with %d models from %s. "
                "Prices stay stable until the next successful reload.",
                reason,
                len(cached),
                cls._cache_path(url),
            )
            _cost_map_source_info.source = "cache"
            _cost_map_source_info.fallback_reason = (
                f"{reason}; using last known good cached map"
            )
            return cached

        verbose_logger.error(
            "LiteLLM: Remote model cost map unavailable (%s) and no usable cache at %s. "
            "Falling back to the bundled backup, which prices models added after the "
            "image build at 0. Reload the cost map once the remote source recovers.",
            reason,
            cls._cache_path(url),
        )
        _cost_map_source_info.source = "local"
        _cost_map_source_info.fallback_reason = reason
        return cls.load_local_model_cost_map()

    @staticmethod
    def _check_is_valid_dict(fetched_map: dict) -> bool:
        """Check 1: fetched map is a non-empty dict."""
        if not isinstance(fetched_map, dict):
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map is not a dict (type=%s). "
                "Falling back to local backup.",
                type(fetched_map).__name__,
            )
            return False

        if len(fetched_map) == 0:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map is empty. "
                "Falling back to local backup.",
            )
            return False

        return True

    @classmethod
    def _check_model_count_not_reduced(
        cls,
        fetched_map: dict,
        backup_model_count: int,
        min_model_count: int = MODEL_COST_MAP_MIN_MODEL_COUNT,
        max_shrink_ratio: float = MODEL_COST_MAP_MAX_SHRINK_RATIO,
    ) -> bool:
        """Check 2: model count has not reduced significantly vs backup."""
        fetched_count = len(fetched_map)

        if fetched_count < min_model_count:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map has only %d models (minimum=%d). "
                "This may indicate a corrupted upstream file. "
                "Falling back to local backup.",
                fetched_count,
                min_model_count,
            )
            return False

        if (
            backup_model_count > 0
            and fetched_count < backup_model_count * max_shrink_ratio
        ):
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map shrank significantly "
                "(fetched=%d, backup=%d, threshold=%.0f%%). "
                "This may indicate a corrupted upstream file. "
                "Falling back to local backup.",
                fetched_count,
                backup_model_count,
                max_shrink_ratio * 100,
            )
            return False

        return True

    @classmethod
    def validate_model_cost_map(
        cls,
        fetched_map: dict,
        backup_model_count: int,
        min_model_count: int = MODEL_COST_MAP_MIN_MODEL_COUNT,
        max_shrink_ratio: float = MODEL_COST_MAP_MAX_SHRINK_RATIO,
    ) -> bool:
        """
        Validate the integrity of a fetched model cost map.

        Runs each check in order and returns False on the first failure.

        Checks:
        1. ``_check_is_valid_dict`` -- fetched map is a non-empty dict.
        2. ``_check_model_count_not_reduced`` -- model count meets minimum
           and has not shrunk >``max_shrink_ratio`` vs backup.

        Returns True if all checks pass, False otherwise.
        """
        if not cls._check_is_valid_dict(fetched_map):
            return False

        if not cls._check_model_count_not_reduced(
            fetched_map=fetched_map,
            backup_model_count=backup_model_count,
            min_model_count=min_model_count,
            max_shrink_ratio=max_shrink_ratio,
        ):
            return False

        return True

    @staticmethod
    def fetch_remote_model_cost_map(url: str, timeout: int = 5) -> dict:
        """
        Fetch the model cost map from a remote URL.

        Returns the parsed JSON dict. Raises on network/parse errors
        (caller is expected to handle).
        """
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()


class ModelCostMapSourceInfo:
    """Tracks the source of the currently loaded model cost map."""

    source: str = "local"  # "local", "cache" or "remote"
    url: Optional[str] = None
    is_env_forced: bool = False
    fallback_reason: Optional[str] = None
    cache_path: Optional[str] = None


# Module-level singleton tracking the source of the current cost map
_cost_map_source_info = ModelCostMapSourceInfo()


def get_model_cost_map_source_info() -> dict:
    """
    Return metadata about where the current model cost map was loaded from.

    Returns a dict with:
    - source: "remote", "cache" or "local"
    - url: the remote URL attempted (or None for local-only)
    - is_env_forced: True if LITELLM_LOCAL_MODEL_COST_MAP=True forced local usage
    - fallback_reason: human-readable reason if the remote fetch failed
    - cache_path: where the last known good map is persisted
    """
    return {
        "source": _cost_map_source_info.source,
        "url": _cost_map_source_info.url,
        "is_env_forced": _cost_map_source_info.is_env_forced,
        "fallback_reason": _cost_map_source_info.fallback_reason,
        "cache_path": _cost_map_source_info.cache_path,
    }


def get_model_cost_map(url: str) -> dict:
    """
    Public entry point — returns the model cost map dict.

    1. If ``LITELLM_LOCAL_MODEL_COST_MAP`` is set, uses the local backup only.
    2. Otherwise fetches from ``url``, validates integrity, persists the map as the
       last known good cache, and on any failure falls back to that cache and only
       then to the bundled backup.

    Only the backup model count is cached (a single int) for validation.
    The full backup dict is only parsed when it must be *returned* as a
    fallback — it is never held in memory long-term.
    """
    # Note: can't use get_secret_bool here — this runs during litellm.__init__
    # before litellm._key_management_settings is set.
    if os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true":
        _cost_map_source_info.source = "local"
        _cost_map_source_info.url = None
        _cost_map_source_info.is_env_forced = True
        _cost_map_source_info.fallback_reason = None
        _cost_map_source_info.cache_path = None
        return GetModelCostMap.load_local_model_cost_map()

    _cost_map_source_info.url = url
    _cost_map_source_info.is_env_forced = False
    _cost_map_source_info.cache_path = GetModelCostMap._cache_path(url)

    try:
        content = GetModelCostMap.fetch_remote_model_cost_map(url)
    except Exception as e:
        return GetModelCostMap._load_fallback_map(url, f"Remote fetch failed: {e}")

    # Validate using cached count (cheap int comparison, no file I/O)
    if not GetModelCostMap.validate_model_cost_map(
        fetched_map=content,
        backup_model_count=GetModelCostMap._get_backup_model_count(),
    ):
        return GetModelCostMap._load_fallback_map(
            url, "Remote data failed integrity validation"
        )

    GetModelCostMap._write_last_good_cache(url, content)
    _cost_map_source_info.source = "remote"
    _cost_map_source_info.fallback_reason = None
    return content
