# MANIFEST — Grounding-Arbitration Redesign (HARDEN)

Author: Opus (declarative spec, zero code). Implementers: tier-0 to spec; Sonnet arch/property
tests; deepseek functional tests from the oracle below. Deterministic gate blocks.

## 0. Problem (grounded in live findings)
The engine optimises **coverage** (fill the cell) over **grounding** (assert only what is
evidenced). Three live Ozon-drill defects share ONE DNA — a weakly-grounded source asserts a
value, and nothing arbitrates it against stronger evidence in the same run:

| Defect | Confirmed mechanism (file:line) |
|---|---|
| DeWalt DWD024 «Режимы работы»=«Сверление» (dropped удар) | `llm_knowledge`, conf 95, evidence "this is a drill, only drilling mode". It was the SOLE proposer of a **collection** attr → `_merge` UNIONs (`pipeline.py:6630`) its single guess. web_search in the SAME run gave «бетон-Ø=16» (⇒ hammer) but on a DIFFERENT attribute. |
| Makita HP1630 «Страна»=«Япония» | force-routed to `web_search` (`classifier.py:150-163`); web_search∈`_TRUSTED_RETRIEVAL` exempts it from COUNTRY-drop (`pipeline.py:3659-3671`); only guard = Gate-A token-in-ANY-snippet (`:6115`) → brand-home-country leaks as product origin. |
| «Тип патрона»=«Быстрозажимной» vs «Комплектация»=«…ключевой патрон» | different attributes; `_merge` is per-attribute (`:6624`) → never compared. Комплектация is DESCRIPTION-verbatim (ground truth); enum-pick contradicts it. No cross-field validator exists. |

Merge today (`_merge_winner` `pipeline.py:2869`): **primary key = self-reported `confidence`
(`:2891`)**; `SOURCE_PRIORITY` (`base.py:66-81`) only breaks a conf tie (`:2893`). The priority
ladder EXISTS (cards=4 > vision/icecat=3 > web_search=2 > llm_knowledge/safe_enum=1) but is
subordinated to a self-reported float.

## 1. Spec (what changes)
Four coordinated changes. Confidence stops being an arbiter; source-class authority governs;
guess-sources abstain; cross-field contradictions are reconciled deterministically.

- **FIX-1 — merge by source-authority, not confidence.** `_merge_winner`: primary key =
  `SOURCE_PRIORITY[source]` (higher wins). `confidence` is REMOVED as an arbiter (kept on the
  object for logging only). Tie within the SAME priority class → deterministic, reproducible
  tiebreak (e.g. longer evidence, else stable source order) — NOT confidence.
- **FIX-2 — LLM_KNOWLEDGE = honest last-resort (NARROWED, round-1 refinement).** The ONLY abstain
  trigger: **LLM_KNOWLEDGE must NOT be the SOLE proposer of an `is_collection` (multi-select)
  attribute** — a type/brand guess cannot reliably enumerate a multi-select → **ABSTAIN (leave
  empty)**. This is the demonstrated DeWalt harm (режимы: collection, llm sole, restrictive single
  value).
  - **SAFE_ENUM_FILL is EXCLUDED from this abstain** — it carries its own grounding (Gate A verbatim
    + Gate B adversarial MUD-block) and is a legitimate type-default mechanism (Сезон/Стиль…). Do
    NOT blanket-drop it. (Round-1 lesson: a blanket evidence-length heuristic dropped SafeEnumFill
    defaults + broke 5 routing tests — and was misreported as pre-existing.)
  - **Scalar LLM_KNOWLEDGE sole-fills are KEPT** (a reasonable single-value inference = useful
    coverage; the harm is multi-select enumeration, not scalar inference). NO numbers-or-≥60-chars
    evidence heuristic — that proxy is wrong; use the structural rule (source==LLM_KNOWLEDGE AND
    is_collection AND sole-proposer).
  - EXTENSION POINT (pending research): before abstain, an optional **retrieval-fallback tier** (a
    search-native model that actually browses) may attempt a genuine grounded fill. If wired, it
    slots BETWEEN "no source found it" and "abstain". Until then: abstain.
- **FIX-3 — per-class grounding strength (country/origin).** For attribute classes where a generic
  snippet ≠ the product's value (country-of-origin first; extensible), Gate-A token-in-snippet is
  INSUFFICIENT: require cross-host corroboration OR product-model adjacency in the evidence, else
  ABSTAIN. Removes the blanket web_search exemption for the COUNTRY class (`:3659-3671`).
- **FIX-4 — deterministic cross-field validator (new home ~`pipeline.py:4404-4412`, pre-finalize).**
  Post-merge, reconcile known cross-field implications. Seed rules:
  - `концентр.-Ø в бетоне present` ⇒ «Режимы работы» MUST include an impact/удар mode (add it, or
    if that violates FIX-2 evidence, flag+drop the inconsistent partial).
  - «Комплектация» free-text names a chuck type ⇒ «Тип патрона» MUST equal it (description-grounded
    wins over enum-pick).
  On conflict: the **stronger-grounded** field (by SOURCE_PRIORITY) wins; the weaker is corrected or
  dropped, never left contradictory.

## 2. Invariants (must always hold)
- **INV-1 (authority-over-confidence).** For the same attribute, a higher-`SOURCE_PRIORITY` value
  ALWAYS wins over a lower one, regardless of `confidence`. `confidence` changes NO merge outcome.
- **INV-2 (last-resort abstain — narrowed).** If LLM_KNOWLEDGE is the SOLE proposer of an
  `is_collection` attribute, that attribute stays EMPTY (a multi-select guess is never emitted as a
  single restrictive value). SAFE_ENUM_FILL is NOT subject to this abstain (own gates). Scalar
  LLM_KNOWLEDGE sole-fills are preserved (coverage).
- **INV-3 (class-grounding).** A COUNTRY-class value survives only with corroboration or
  model-adjacency; a lone generic snippet ⇒ abstain.
- **INV-4 (no surviving contradiction).** After finalize, no two fields hold a known-incompatible
  pair (concrete-Ø vs no-impact-mode; komplektatsiya-chuck vs Тип-патрона).
- **INV-5 (regression / single-source correctness preserved).** An attribute filled by exactly one
  source that IS evidenced is unchanged. Existing correct single-select fills are byte-identical.
- **INV-6 (collections still union).** Multi-value collection merge remains a union across proposing
  sources (`:6630`) — FIX-2 only stops a GUESS-source from being the sole restrictive proposer.

## 3. Architecture rules
- `confidence` may remain on `AttributeValue` for logs/telemetry but MUST NOT gate acceptance or
  decide a merge. AUDIT every current confidence-outcome site and neutralise it or justify:
  `_merge_winner:2891/2894`, card-band `:2886`, consensus bump `:6612-6614`, `_merge_high_conf`
  `:6580`, `_remaining_targets` via `is_confident()` `:6565`→`base.py:134`. **CAUTION:**
  `_remaining_targets` uses confidence to decide "is this target still empty for the next stage" —
  changing it can cascade stage ordering; the implementer MUST preserve stage-emptiness semantics
  (a target already filled by a real source stays filled) while removing confidence as a MERGE
  arbiter. Flag any behaviour change here explicitly.
- The card-protection band (`_CARD_PROTECTION_BAND` `:103`) becomes REDUNDANT under INV-1
  (priority-primary already makes cards beat inference) — remove or fold in, don't leave a second
  competing arbiter.
- No source asserts a value below its class grounding threshold.
- **RISK to surface (user directive stands):** INV-1 makes a card ALWAYS beat web_search even if the
  card value is stale/wrong and web is right. Accepted per user directive (cards = the actual
  marketplace listing = most authoritative). The oracle regression set must include ≥1 card-vs-web
  case to confirm no unacceptable regression; if one appears, raise to Opus/USER, do not silently
  soften INV-1.

## 4. Oracle (input → expected; drives functional tests)
Ground truth from the 3 confirmed live defects + regression. Cheap models CODE these; they do NOT
invent expected values.

| # | Product / attr | Before (bug) | After (required) |
|---|---|---|---|
| O1 | DeWalt DWD024 «Режимы работы» (llm sole, collection) | «Сверление» only | NOT «Сверление» alone: either EMPTY (abstain, FIX-2) or {«Сверление»,«Сверление с ударом»} if FIX-4 бетон⇒удар fires. Never a lone restrictive guess. |
| O2 | Makita HP1630 «Страна» (web_search token) | «Япония» | EMPTY (abstain) — no country asserted without corroboration/adjacency. |
| O3 | «Тип патрона» vs «Комплектация»=«…ключевой патрон» | «Быстрозажимной» | «ключевой» (description-grounded) OR empty; MUST NOT contradict Комплектация. |
| R1 | Bosch GSB 13 RE «Количество скоростей» | 1 (correct) | 1 — UNCHANGED (regression). |
| R2 | Bosch GSB 13 RE «удары/мин»=44800 (web_search, evidenced) | 44800 | 44800 — UNCHANGED. |
| R3 | any attr where a CARD value competes with an llm/web value | — | card value wins (INV-1) — add a real fixture. |
| R4 | any correct single-source llm fill WITH evidence | — | UNCHANGED (INV-5). |

## 5. Gate (HARDEN — blocks)
- Gate-1 deterministic: ruff+mypy+radon+vulture+bandit + conformance(code↔this manifest) +
  arch/property tests (INV-1..6) + functional tests (oracle O1-O3, R1-R4) + mutation self-check.
- Gate-2 cross-family review (codemap+manifest), residual/design only.
- Gate-2D not applicable (code deliverable, not a document) — the oracle IS the content judge.
- **Differential/regression harness:** re-run the 3 defect fixtures + regression set against the
  patched pipeline; O1-O3 must flip, R1-R4 must NOT move. This is the authoritative semantic judge.

## 6. Out of scope (this manifest)
- The retrieval-fallback model tier (FIX-2 extension) — pending the model research; wired only on a
  separate go.
- Any change to web_search transport (Serper/scrape.do) — already working.

## 7. Round 3 — FIX-5 (chuck cross-field) + FIX-6 (annotation numeric grounding)
Live e2e (post FIX-1..4 deploy) exposed two adjacent hallucination classes the structural fixes do
not cover:
- DeWalt DWD024 «Тип патрона»=«SDS-Plus» — WRONG (DWD024 is a keyed 13mm chuck; «Комплектация» names
  «патронный ключ»; SDS-Plus belongs to rotary hammers, not a plain «Дрель»). RULE-B missed it: it
  only matched «ключевой патрон»/«быстрозажимной», not the «патронный ключ» (chuck-KEY tool) evidence.
- Makita HP1630 annotation invented «крутящий момент 48 Нм» — it took «48000 уд/мин» and misrendered
  it as torque. Free-text `_generate_annotation` confuses fields/units.

**FIX-5 — extend `_reconcile_cross_field_contradictions` (RULE-B):**
- «Комплектация» containing «патронный ключ» (chuck-key tool) ⇒ «Тип патрона» MUST be «Ключевой».
- TYPE consistency: for product type «Дрель» (plain drill), «Тип патрона»=«SDS-Plus» (or any
  rotary-hammer chuck) is INVALID → correct to the комплектация/description-implied type, or abstain.

**FIX-6 — annotation numeric grounding (`_generate_annotation`):**
- Prompt hardening: NEVER introduce a numeric value or unit not present VERBATIM in the provided
  characteristics; NEVER convert/compute/re-unit a number; if a spec isn't in the fields, describe
  qualitatively or omit.
- Deterministic post-check: extract every «number+unit» token from the generated annotation; each MUST
  match a filled field's (value, unit) AFTER normalising equivalent units/format (1,8 кг ≡ 1800 г;
  comma/dot; кг↔г, мм↔mm). A number+unit not backed by a field — especially a field number re-attached
  to a WRONG unit (48000 уд/мин → «48 Нм») → regenerate once with the violation named, else strip that
  clause. MUST NOT false-strip a legitimate unit reformat.

**INV-7:** every «number+unit» in a generated annotation corresponds to a filled field (value+unit,
unit-normalised); otherwise stripped/regenerated.

**Oracle (Round 3):**
- O4: «Тип патрона»=«SDS-Plus» + type=«Дрель» + «Комплектация» names «патронный ключ» → corrected to
  «Ключевой» (or dropped); never SDS-Plus for a plain drill.
- O5: annotation contains «48 Нм» but no field has 48 Нм torque (fields have «48000 уд/мин») → the
  «48 Нм» claim removed/regenerated; post-check flags it.
- O5b (regression, no false-strip): annotation says «вес 1,8 кг», field «Вес»=1800 г → KEPT unchanged
  (unit-equivalent, legitimate reformat).
