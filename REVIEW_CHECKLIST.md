# PrismaLab Review Checklist

This checklist is a project-level review standard for Codex/Claude.

If any rule here conflicts with `/Users/germanzukov/PrismaLab/CLAUDE.md`, `CLAUDE.md` is the source of truth.

## 1) Required Review Output (always)

Every review must include:

1. `Findings` (bugs/regressions/risks), sorted by severity.
2. `Architecture Risks` (mandatory separate block).
3. `Assumptions/Open Questions` (only if needed).

If there are no architecture risks, write explicitly:

`Architecture Risks: none`

## 2) Findings Format

For each finding include:

- Severity: `P1` / `P2` / `P3`
- Why it matters
- File + line reference
- `Merge-blocking`: `yes` or `no`
- Minimal fix direction

## 3) Severity Policy

- `P1` — wrong money/credits behavior, broken critical user flow, data corruption risk, security risk.
- `P2` — high-probability UX or logic regression, architecture debt that can quickly spread.
- `P3` — low-risk quality issue, style inconsistency, minor maintainability concern.

Merge policy:

- Any `P1` => do not merge.
- `P2` can merge only with explicit owner approval and follow-up task.

## 4) Project Architecture Invariants

Review must verify these invariants when relevant:

1. `bot.py` remains entrypoint/orchestration, not feature-logic dump.
2. `messages.py` stores user-facing texts; avoid hardcoded strings in handlers.
3. `keyboards.py` stores keyboard builders; avoid inline keyboard duplication.
4. Pack/style `credit_cost` must be consistent across catalog, detail, and actual charge/write-off paths.
5. Lock state authority is backend; frontend only renders backend lock DTO.
6. Photosets footer state rules remain deterministic (single source of truth function, no duplicate competing toggles).
7. Tariff selection state stays unified (`selectedTariff`-style approach), no parallel conflicting states.
8. Payment/user handoff flows stay consistent with current product behavior (bot continues flows where expected).
9. Dev/prod pack differences are env-config driven, not hardcoded in feature code.
10. Shared UI IDs used by balance/state updaters are not accidentally broken by markup refactors.

## 5) Anti-Spaghetti Checks

When reviewing changed code, check:

1. No duplicated business rules across route + service + frontend.
2. No copy-paste API/payment/tariff logic where helper extraction is straightforward.
3. No “god function” growth without decomposition (follow size guidance from `CLAUDE.md`).
4. No hidden cross-screen side effects in global state updates.
5. No accidental coupling of unrelated UI blocks via shared selectors.

## 6) Data Integrity and Security Checks

1. SQL uses parameterization (driver placeholders), no user-data interpolation in query strings.
2. Credits/money operations are atomic and race-safe.
3. External API/payment failures are logged with context and have explicit fallback behavior.
4. Frontend does not inject user-controlled content into `innerHTML`.

## 7) Validation Gates Before “Done”

Required:

1. `python3 -m pytest tests/ -v`
2. `python3 -c "from prismalab.bot import main"`

When behavior changed:

1. Add/adjust tests for new logic (happy path + at least one failure/edge path).
2. Manual smoke-check critical state matrix if UI/flow changed:
   - no persona + no credits
   - persona + no credits
   - persona + credits

## 8) Definition of Review Complete

A review is complete only when:

1. Findings are listed with severities and references.
2. Architecture Risks block is present (or explicit `none`).
3. Merge recommendation is explicit (`safe to merge` / `hold`).
