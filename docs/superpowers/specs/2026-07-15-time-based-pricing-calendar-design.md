# Time-Based Pricing Calendar Design

> Status: implemented in this fork. The schema below is what
> `litellm/litellm_core_utils/llm_cost_calc/time_based_pricing.py` accepts, and the
> desktop client mirrors the same semantics in
> `apps/desktop/src/utils/timeBasedPricing.ts`.

## Context

`time_based_pricing` (see `2026-06-30-time-based-pricing-design.md`) supports `timezone`, `start_time` / `end_time`, `multiplier` and an optional weekday filter `days`. It has no calendar-date dimension: the original design explicitly listed "calendar-date-specific holiday pricing" as a non-goal.

DeepSeek's published peak/off-peak policy now makes that dimension load-bearing. Per DeepSeek's 2026-09-19 peak/off-peak notice, two classes of days are priced differently:

| Day | Billing |
| --- | --- |
| Workday that is not a statutory holiday | peak window / off-peak window |
| Saturday, Sunday (including weekends the holiday schedule turns into workdays, 调休) | off-peak all day |
| Statutory holiday (including a multi-day National Day block) | off-peak all day |

Two consequences follow:

1. A holiday exclusion is mandatory. Without it, statutory holidays that fall on a weekday are billed at peak, which is exactly what the provider does not charge.
2. A weekday filter alone is not enough either: `days: [mon..fri]` happens to align with this particular policy, but it cannot express the holiday exclusion, and it silently becomes wrong for any provider that does charge peak on an adjusted weekend. Date classes express the policy directly.

The cost map therefore needs an explicit holiday calendar. This design adds it as configuration data inside `time_based_pricing` itself: no external loader, no new file format, no code path that can silently fail to find a calendar file.

`calendar.workdays` exists as a general override for providers whose schedule does charge peak on an adjusted weekend. DeepSeek is not one of them: every Saturday and Sunday is off-peak all day, so a DeepSeek entry should leave `calendar.workdays` unset.

## Goals

- Let a rule declare which calendar dates it applies to, including statutory holidays, ordinary weekends, and (for providers that need it) weekends their schedule turns into workdays.
- Keep the entire calendar inside `time_based_pricing`, so one model entry is self-describing and reviewable in a diff.
- Be fully backward compatible: existing configs (including the current deepseek entries, which use only time windows) keep their exact behavior.
- Degrade to today's behavior when date config is absent or malformed. Never raise, never fail a model call.
- Keep matching deterministic and cheap, using the request start time and the configured timezone.

## Non-Goals

- Do not hardcode Chinese holidays in Python.
- Do not add an external calendar file loader or a network calendar fetch.
- Do not support recurring rules such as "the nth Monday of a month", or lunar-calendar derivation. Holidays are published yearly and are configured explicitly.
- Do not change static price semantics or provider-reported costs.
- Do not add a proxy UI editor for the new fields.

## Schema

Everything is optional and lives under `time_based_pricing`.

```yaml
time_based_pricing:
  timezone: Asia/Shanghai

  # Calendar data. Lists of "YYYY-MM-DD" or "YYYY-MM-DD..YYYY-MM-DD" ranges.
  calendar:
    holidays: ["2026-09-25..2026-09-27", "2026-10-01..2026-10-07"]

  rules:
    # Peak windows apply on workdays only. Weekends and the holidays listed above
    # fall through to the static price, which is the off-peak price.
    - name: deepseek_peak_morning
      start_time: "09:00"
      end_time: "12:00"
      multiplier: 1.6
      dates: [workday]

    - name: deepseek_peak_afternoon
      start_time: "14:00"
      end_time: "18:00"
      multiplier: 1.6
      dates: [workday]
```

### `calendar`

- `holidays`: dates that are off but are not ordinary weekends. Used by the `holiday` date class.
- `workdays`: optional override for providers that do charge peak on a weekend the holiday schedule turned into a workday; such dates become `workday` instead of `weekend`. Leave it unset when every Saturday and Sunday is off-peak.
- Both accept `"YYYY-MM-DD"` and inclusive `"YYYY-MM-DD..YYYY-MM-DD"` range strings, or a raw `{start, end}` object in JSON cost-map entries.
- `workdays` wins over `holidays` and over the natural weekend rule if a date appears in both. That is the safe direction: an explicit "people work this day" instruction should not be cancelled by a sloppy range.
- An empty or absent `calendar` is valid and means "no calendar data".

### Rule date classes

Each rule may declare `dates`, a list of classes evaluated in the rule's timezone:

- `workday`: not a Saturday/Sunday, and not in `calendar.holidays`, or explicitly listed in `calendar.workdays`.
- `weekend`: Saturday/Sunday and not listed in `calendar.workdays`. Every adjusted weekend is off-peak unless the provider's calendar says otherwise, so this is the correct class for a DeepSeek-style policy.
- `holiday`: listed in `calendar.holidays`.
- `all`: every date. This is the default when `dates` is omitted.
- `"YYYY-MM-DD"` or `"YYYY-MM-DD..YYYY-MM-DD"`: explicit literal date or inclusive range, allowed inline so one-off promotions do not need a calendar.

An unknown entry invalidates the whole `dates` list, so the rule is skipped with a debug log. Validation covers every entry before any matching: a rule written as `dates: ["workday", "hoilday"]` is skipped entirely rather than silently applying on workdays. This matches how an unknown `days` value already behaves.

### Interaction with `days`

- `dates` omitted: current behavior exactly. `days`, when present, filters by `local_datetime.weekday()`.
- `dates` present: the date class decides. `days`, if also present, is applied as an additional filter on top (`dates AND days`). This lets a config say "workdays, Monday through Thursday only" without needing new classes.
- Because `days` is a literal weekday filter, `days: [mon..fri]` combined with `dates: [workday]` can never match an adjusted weekend. That is the desired outcome for DeepSeek (weekends are off-peak regardless), but it would be wrong for a provider that charges peak on adjusted weekends. Prefer `dates` alone when the date classes alone express the policy.
- `dates: [all]` is equivalent to omitting `dates` for class purposes but still composes with `days`.

### Full-day windows

`start_time: "00:00"` with `end_time: "00:00"` means the whole day. The window helper returns True whenever `start_time == end_time`, so a full-day rule matches at every wall-clock time including both midnights; `start_time > end_time` keeps its existing wrap-past-midnight meaning. Configs that want the whole day should use this form rather than four overlapping windows.

## Architecture

Extend `litellm/litellm_core_utils/llm_cost_calc/time_based_pricing.py`. No new module is required, but the file gains two focused concerns:

1. **Calendar compilation.** Parse `calendar.holidays` and `calendar.workdays` into `frozenset[date]` plus a list of inclusive `(date, date)` ranges, once per distinct config.
2. **Date class matching.** Turn the compiled calendar plus `local_datetime.date()` into a `set[str]` of satisfied classes (`workday`, `weekend`, `holiday`, `all`, plus the literal `YYYY-MM-DD` when the date matches), then test it against each rule's `dates`.

Suggested additions:

```python
CalendarSpec = Dict[str, Any]  # {"holidays": [...], "workdays": [...]}

def compile_time_based_pricing_calendar(
    pricing_config: Mapping[str, Any],
) -> CompiledCalendar: ...

def get_matching_date_classes(
    compiled_calendar: CompiledCalendar,
    local_date: date,
) -> frozenset[str]: ...
```

`TimeBasedPricingResult` stays as-is; optionally extend it with `date_classes: Optional[List[str]]` for cost-breakdown debugging. Adding a key to a `total=False` `TypedDict` is backward compatible.

### Caching

`get_time_based_pricing_result` is called once per request on the cost path, so the calendar must not be re-parsed per call. Compile lazily into a module-level `lru_cache` keyed by the raw config (a stable string key over the `time_based_pricing` dict, or `id()`-free canonical JSON). Invalid entries must still be cacheable as "empty calendar" so a bad config cannot turn into a per-request parse.

Cost: one dict-hash lookup per request. The per-date work is `O(len(ranges))` for range membership plus `O(len(rules))` for rule matching, both bounded by config size and independent of request volume.

## Matching Flow

Inserted into the existing loop, before the time-window test:

1. Resolve `local_datetime` from `pricing_datetime` and `timezone` (unchanged).
2. If `dates` is present on the rule, compute the matching date classes once per call and skip the rule when `dates` and the class set are disjoint.
3. Apply the existing `days` filter, if present.
4. Apply the existing window test.
5. Return the first matching rule.

Rules should not overlap; first match in config order wins, as today.

## Cost Map Updates

The current eight `deepseek*` entries use `09:00-12:00` and `14:00-18:00` at `multiplier: 2.0` with no `days` and no calendar. They must be revisited together with this feature:

- Whether the static `input_cost_per_token` / `output_cost_per_token` values represent peak or off-peak prices determines whether the off-peak rule is `0.5` (static = peak) or the peak rule is `2.0` (static = off-peak). The multiplier must be expressed consistently against the static base price. Do not assume `2.0` is the only valid spelling.
- Weekend and statutory-holiday rules are required for correctness even outside holiday season, because ordinary weekends are already off-peak all day.
- The published peak window must be transcribed from DeepSeek's current notice, not carried over from an older price page.

## Request Time Semantics

Unchanged and worth restating because holidays make it visible: pricing uses the request start time. A request that begins on 2026-09-30 23:50 and finishes on 2026-10-01 00:10 is priced by 09-30. Do not re-evaluate the calendar at response time.

## Error Handling

All of the following fall back to the base price with debug logging, and must not raise:

- `calendar` not a dict, or `holidays` / `workdays` not lists.
- Unparseable date string or range (skipped individually, valid siblings kept).
- Unknown entry in a rule's `dates`.
- Timezone resolution failure (existing behavior).

A malformed `calendar` degrades that model to "no holiday awareness", i.e. exactly today's behavior. Partial validity is preferred over all-or-nothing so one bad date string does not silently disable the whole calendar; log at debug with the offending value.

## Tests

Unit tests in `apps/litellm/tests/test_litellm/test_cost_calculator.py`, alongside the existing time-based pricing tests:

Date classes:

- Ordinary workday resolves to `workday` only.
- Ordinary Saturday resolves to `weekend` only.
- A date in `calendar.holidays` resolves to `holiday`, not `workday`.
- A weekend date in `calendar.workdays` resolves to `workday`, not `weekend`.
- A date in both lists resolves to `workday`.
- Range strings are inclusive on both ends.

Rule matching:

- Existing config with no `dates` produces identical results to the current tests (regression guard).
- A `dates: [holiday]` full-day rule matches 2026-10-01 at 10:00 Asia/Shanghai and does not match 2026-10-12.
- A `dates: [workday]` rule does not match within the configured holiday block.
- With a DeepSeek-shaped config (no `workdays`), 2026-09-20 and 2026-10-10 resolve to `weekend`, so the peak rules do not match them.
- With `calendar.workdays` set, those same dates resolve to `workday` and the peak rules do match.
- `dates` combined with `days` requires both.
- Unknown date class skips the rule.

Resilience and integration:

- Missing `calendar` keeps weekend-blind behavior (backward compatible).
- Malformed date string keeps the remaining valid dates usable.
- An end-to-end cost test with the DeepSeek-shaped config asserts peak price inside the workday peak window and base price at 10:00 on 2026-10-01.
- A request starting before a holiday boundary and ending inside it uses the start date.

## Client Parity

The desktop client mirrors these semantics for displayed cost estimates:

- `apps/desktop/src/utils/timeBasedPricing.ts` (`resolveTimeBasedPricing`)
- `apps/desktop/src/stores/modelPriceStore.ts`
- types mirrored in `apps/website/app/services/types.ts`

If only the Python side gains calendar support, the client shows peak pricing while the gateway charges off-peak, which is a billing-visibility bug. The TypeScript resolver must gain the same `calendar` and `dates` semantics, and both sides need the same boundary cases asserted (2026-10-01 all-day off-peak, 2026-10-10 and 2026-09-20 as workdays). Keep the two implementations semantically identical and labeled as such in comments.

## Compatibility

- Absent `calendar` and absent `dates` → byte-for-byte identical behavior to today.
- `TimeBasedPricingResult` only gains optional keys.
- Cost breakdown metadata is additive.
- No new dependency, no file I/O, no network.

## Where The Calendar Is Configured

`time_based_pricing` is declared on `ModelInfoBase` (`litellm/types/utils.py`) and passed through unchanged by `get_model_info` (`litellm/utils.py`), so both the model cost map and a deployment's `model_info` can carry it. The two are not equivalent in practice.

**The repository's root `model_prices_and_context_window.json` is not read at runtime by default.** `litellm/__init__.py` defaults `model_cost_map_url` to the upstream BerriAI URL, and `get_model_cost_map` only falls back to the packaged `litellm/model_prices_and_context_window_backup.json` when that fetch fails. The backup file currently contains no `time_based_pricing` entries at all, so any calendar data added to the root JSON would have no effect unless one of these is set:

- `LITELLM_LOCAL_MODEL_COST_MAP=True` — uses the packaged backup, which freezes the whole cost map at image-build time and requires a new image for every provider price change.
- `LITELLM_MODEL_COST_MAP_URL=<our own JSON>` — self-hosted cost map; must respect the loader's integrity checks (minimum model count and maximum shrink ratio) or it silently falls back to the packaged backup.

**Therefore the calendar belongs in the deployment's `model_list[].model_info`**, not in the shared cost map. This path is deterministic, hot-updatable without rebuilding, and avoids the upstream-merge conflict described below. Two properties make it safe:

- `Router.get_model_info` applies the deployment's `model_info` with a top-level `model_info.update(user_model_info)` (`litellm/router.py`), so a deployment-level `time_based_pricing` **replaces the cost map's block wholesale** rather than merging rule lists.
- Proxy config loaded from the database merges with `_deep_merge_dicts` (`litellm/proxy/proxy_server.py`), which recurses into dicts but replaces lists, so `rules` and `calendar` are replaced as units.

Consequence for operators: because the block is replaced wholesale, a deployment that sets `time_based_pricing` must repeat the complete rule set it wants. Partial override of individual rules is not supported and must not be implied by the docs.

If the cost map is used anyway, keep the schema change (the `calendar` / `dates` fields) separate from the yearly holiday data. The yearly data is China-specific and expires annually; contributing it upstream is neither likely to be accepted nor maintainable in a fork that follows upstream. Only the schema belongs in the shared file.

## Rollout

1. Land the schema and helper changes behind the optional fields, with the regression test that proves existing configs are unchanged.
2. Configure the deepseek calendar and rules in the deployment `model_info`, with a multiplier direction consistent with the static base price. Do not rely on the repository's root cost map taking effect; confirm the effective cost map source first.
3. Mirror the semantics in the desktop resolver and add matching tests.
4. Update `docs/my-website/docs/provider_registration/add_model_pricing.md` and `docs/my-website/docs/proxy/custom_pricing.md` with the `calendar` and `dates` fields, the full-day window form, the whole-block-replacement caveat, and the DeepSeek example.

A reduced first step is acceptable if the full calendar is too much at once: implement `dates` with literal dates and ranges only, list the current holiday block explicitly in the model entry, and add the `holiday` / `weekday` / `weekend` classes in a follow-up. The schema above is forward compatible with that split.
