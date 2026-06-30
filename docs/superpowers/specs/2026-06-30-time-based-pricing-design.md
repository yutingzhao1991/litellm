# Time-Based Model Pricing Design

## Context

LiteLLM stores model pricing in `model_prices_and_context_window.json` and calculates spend through the shared cost calculation path. Today the pricing map supports static per-token, per-second, cache, modality, service-tier, and context-threshold prices. It does not support time-of-day pricing.

DeepSeek is expected to introduce peak pricing where all billable items are charged at 2x during Beijing time windows:

- 09:00 to 12:00
- 14:00 to 18:00

This design adds a generic model-cost-map feature so DeepSeek can be configured without provider-specific hardcoding.

## Goals

- Allow time-of-day price adjustments to be configured in `model_prices_and_context_window.json`.
- Keep the feature provider-agnostic and usable by any model/provider.
- Preserve existing behavior for models that do not configure time-based pricing.
- Apply the multiplier consistently to all token cost components calculated by the generic token cost path, including input, output, cache read, cache write, reasoning, audio, and image token components.
- Use request start time for logged requests so cost tracking is stable even for long responses.
- Keep invalid or incomplete time pricing config from breaking model calls.

## Non-Goals

- Do not implement provider-specific DeepSeek branching.
- Do not add a UI editor for the new pricing field in this design.
- Do not support calendar-date-specific holiday pricing in the first version.
- Do not change provider-reported costs from response headers; if a provider explicitly returns a cost, LiteLLM should continue to trust that value.
- Do not apply this first implementation to every non-token cost calculator. The initial scope is the shared token path used by DeepSeek and most chat/completion models.

## Configuration Schema

Add an optional `time_based_pricing` object to `ModelInfo`.

```json
{
  "time_based_pricing": {
    "timezone": "Asia/Shanghai",
    "rules": [
      {
        "name": "peak_morning",
        "start_time": "09:00",
        "end_time": "12:00",
        "multiplier": 2.0
      },
      {
        "name": "peak_afternoon",
        "start_time": "14:00",
        "end_time": "18:00",
        "multiplier": 2.0
      }
    ]
  }
}
```

Field semantics:

- `timezone`: required when `time_based_pricing` is present. Uses an IANA timezone string, for example `Asia/Shanghai`.
- `rules`: required list. Empty lists are treated as no time-based pricing.
- `name`: optional human-readable rule name for logs and cost breakdowns.
- `start_time`: required local wall-clock time in `HH:MM` format. Start is inclusive.
- `end_time`: required local wall-clock time in `HH:MM` format. End is exclusive.
- `multiplier`: required positive number. `2.0` doubles all applicable costs.
- `days`: optional list of weekday names. Values are case-insensitive and support `mon`, `tue`, `wed`, `thu`, `fri`, `sat`, `sun`. If omitted, the rule applies every day.

Rules should not overlap. If multiple rules match, LiteLLM uses the first matching rule in config order. This keeps behavior deterministic and easy to reason about.

## DeepSeek Example

```json
{
  "deepseek/deepseek-chat": {
    "cache_creation_input_token_cost": 0.0,
    "cache_read_input_token_cost": 2.8e-08,
    "input_cost_per_token": 2.8e-07,
    "input_cost_per_token_cache_hit": 2.8e-08,
    "litellm_provider": "deepseek",
    "max_input_tokens": 131072,
    "max_output_tokens": 8192,
    "max_tokens": 8192,
    "mode": "chat",
    "output_cost_per_token": 4.2e-07,
    "time_based_pricing": {
      "timezone": "Asia/Shanghai",
      "rules": [
        {
          "name": "deepseek_peak_morning",
          "start_time": "09:00",
          "end_time": "12:00",
          "multiplier": 2.0
        },
        {
          "name": "deepseek_peak_afternoon",
          "start_time": "14:00",
          "end_time": "18:00",
          "multiplier": 2.0
        }
      ]
    }
  }
}
```

The static prices remain the non-peak prices. During either configured peak window, the final prompt and completion token costs are multiplied by `2.0`.

## Architecture

Add a focused helper module:

`litellm/litellm_core_utils/llm_cost_calc/time_based_pricing.py`

Responsibilities:

- Parse `time_based_pricing` from a model info dictionary.
- Convert a supplied `datetime` into the configured local timezone using `zoneinfo.ZoneInfo`.
- Match local weekday and wall-clock time against configured rules.
- Return a small result object containing the multiplier and optional matched rule metadata.

Suggested public API:

```python
class TimeBasedPricingResult(TypedDict, total=False):
    multiplier: float
    rule_name: Optional[str]
    timezone: Optional[str]


def get_time_based_pricing_result(
    model_info: ModelInfo,
    pricing_datetime: Optional[datetime] = None,
) -> TimeBasedPricingResult:
    ...
```

The default result is `{"multiplier": 1.0}`.

## Cost Calculation Flow

Extend the shared token cost path:

1. `Logging._response_cost_calculator()` passes request start time into `response_cost_calculator()`.
2. `response_cost_calculator()` passes it into `completion_cost()`.
3. `completion_cost()` passes it into `cost_per_token()`.
4. `cost_per_token()` lets the selected provider token calculator calculate prompt and completion cost exactly as it does today.
5. Before returning from token-like pricing branches, `cost_per_token()` looks up a time-based pricing result from `model_info`.
6. If the multiplier is not `1.0`, multiply both `prompt_cost` and `completion_cost`.

This preserves all existing token accounting logic and applies the multiplier after provider-specific token adjustments. Applying the multiplier after provider calculators avoids subtle conflicts with existing provider wrappers such as Anthropic geo/speed multipliers.

Direct calls to `litellm.cost_per_token()` and `litellm.completion_cost()` should accept an optional `pricing_datetime`. If omitted, they use the current time. This makes the public helper usable and testable without relying on monkeypatching global time.

Explicit `custom_cost_per_token` values passed directly to a request keep their current behavior and are not modified by `time_based_pricing`. The feature is driven by model info from the cost map or deployment-level `model_info`.

## Request Time Semantics

For logged LiteLLM and Proxy calls, pricing should be based on request start time, not response end time. This matters for requests that begin before a peak boundary and finish after it.

The logging layer already tracks start and end times. The implementation should pass the existing `start_time` value into the cost calculator. If that value is missing or not a usable `datetime`, the helper falls back to current time.

## Error Handling

Invalid time-based pricing config should not fail model calls. The helper should return the default multiplier and emit debug-level logging for:

- Missing or invalid `timezone`
- Missing or invalid `rules`
- Invalid `start_time` or `end_time`
- Non-positive or non-numeric `multiplier`
- Unknown weekday values

The cost map remains community-maintained, so resilience is more important than strict runtime failure.

## Cost Breakdown Metadata

When a multiplier is applied, store optional metadata in the cost breakdown if a logging object is available:

```json
{
  "time_based_pricing": {
    "multiplier": 2.0,
    "rule_name": "deepseek_peak_morning",
    "timezone": "Asia/Shanghai"
  }
}
```

This is useful for debugging spend logs. It should be additive and must not change the existing `input_cost`, `output_cost`, or `total_cost` meanings.

## Type Updates

Update type definitions so the new field is discoverable:

- `ModelInfoBase` in `litellm/types/utils.py`
- `CustomPricingLiteLLMParams` if deployment-level custom pricing should support the same field
- Any schema helpers used by model info JSON schema tests

The field should be optional and typed permissively enough to remain compatible with Pydantic v1 and v2.

## Documentation

Update:

- `docs/my-website/docs/provider_registration/add_model_pricing.md`
- `docs/my-website/docs/proxy/custom_pricing.md`

Document:

- The new `time_based_pricing` field.
- Timezone requirements.
- Inclusive start and exclusive end semantics.
- DeepSeek peak pricing example.
- That static cost fields represent the base price.

## Tests

Add focused unit tests in the cost calculator test suite.

Core helper tests:

- No `time_based_pricing` returns multiplier `1.0`.
- `Asia/Shanghai` 09:00 matches the morning peak rule.
- `Asia/Shanghai` 11:59 matches the morning peak rule.
- `Asia/Shanghai` 12:00 does not match the morning peak rule.
- `Asia/Shanghai` 14:00 matches the afternoon peak rule.
- `Asia/Shanghai` 18:00 does not match the afternoon peak rule.
- `days` filters are honored.
- Invalid timezone and invalid times return multiplier `1.0`.

Integration-style cost tests:

- A model configured with DeepSeek-like pricing returns normal cost outside peak.
- The same model returns exactly 2x cost inside peak.
- Prompt cache read and output costs are both multiplied inside peak.
- Existing models without `time_based_pricing` produce unchanged costs.

Logging test:

- When logging start time is inside a peak window and end time is outside it, cost uses the start time.

## Compatibility

The feature is backward compatible:

- Existing cost map entries do not change behavior.
- Existing custom pricing fields keep their semantics.
- Provider-specific cost calculators that do not use `generic_cost_per_token()` are unaffected in the first version.
- Provider response costs from hidden params continue to take precedence.

## Future Extensions

The schema can be extended later without breaking this design:

- Rule-specific `cost_overrides` for providers that publish completely different peak prices.
- Date ranges for temporary promotions or holidays.
- A global proxy-level default time pricing policy.
- Applying the same helper to image, rerank, OCR, and other non-token calculators where useful.
