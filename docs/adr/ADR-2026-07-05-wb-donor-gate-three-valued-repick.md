# ADR: WB donor-gate — трёхзначный вердикт + budget re-pick вместо дропа всего донора

Дата: 2026-07-05. Статус: ACCEPTED (Fable 5). Затрагивает: `wb_card_source`, `donor_gate`.
Реализация: коммит `a333409`. Часть пака `ADR-2026-07-05-web-fetch-guard-regression-recovery.md` (гард #2).

## Проблема

`WbCardSource` берёт лучшую скачанную карту (`_pick_best_card`) и, если она гейт-требующая,
прогоняет через LLM donor-gate `is_same_product`. Два дефекта роняли recall и/или precision:
1. **fail-open кэшируется**: `is_same_product` bool, при сбое транспорта/парса возвращал True
   И КЭШИРОВАЛ — транзиентная сетевая ошибка замерзала как «тот же товар» навсегда (precision-риск).
2. **дроп всего донора на different**: LLM-вердикт «не тот товар» → `return []` → теряются
   ВСЕ скачанные карты, хотя следующая по релевантности могла быть верной (recall-потеря).

## Ключевой факт, определивший решение (проверено чтением исходников)

`is_same_product` вызывается из ДВУХ мест: title-path (основной) и article-path. Обёртка
должна остаться байт-идентичной для IceCat-вызова (fail-open bool-контракт). Enum наружу в
`is_same_product` ломает IceCat → трёхзначность прячем в НОВЫЙ метод `verdict()`, а
`is_same_product` переписываем как тонкую обёртку над ним.

## Решение: вариант A (трёхзначный verdict + budget re-pick). Контракт is_same_product сохранён.

- **`DonorVerdict(SAME, DIFFERENT, UNKNOWN)`** (зеркало FIX-16 identity-verdict). Маппинг:
  `parsed.same is True`→SAME, `False`→DIFFERENT, exception/None→UNKNOWN. **UNKNOWN НЕ
  кэшируется** (транзиент не замораживаем); SAME/DIFFERENT кэшируются.
- **`verdict(target, donor) -> tuple[DonorVerdict, bool]`** — один LLM-вызов, БЕЗ внутреннего
  ретрая (гейт не знает extract-бюджет). Второй элемент = was_llm_call (для учёта бюджета).
- **`is_same_product`** = тонкая обёртка: `v,_ = verdict(...); return v != DIFFERENT`
  (fail-open: SAME/UNKNOWN→True, DIFFERENT→False). Байт-идентична для IceCat.
- **`_rank_cards`** — рефактор `_pick_best_card`: возвращает ОТСОРТИРОВАННЫЙ список кандидатов
  (не одного лучшего), поведение-сохраняющий; `_pick_best_card` = тонкая обёртка `ranked[0]`.
- **`_GateBudget(cap=3)`** (`WB_DONOR_GATE_MAX_LLM`) — кап суммарных donor-gate LLM-вызовов
  на один extract-проход; общий на оба call-site.
- **`_select_gated_card`** — цикл по ranked: чистый (не-гейтимый) кандидат → принять без LLM;
  гейт-требующий и бюджет исчерпан → break; иначе `verdict()`: SAME→принять; DIFFERENT→
  следующий кандидат; UNKNOWN→ретрай×1 если бюджет есть, второй UNKNOWN→дроп кандидата
  (fail-safe, консервативно), считать в логах отдельно.

## Отвергнутые альтернативы

- **Enum вместо bool в `is_same_product`** — ломает IceCat-вызов (fail-open контракт).
- **Ретрай внутри `DonorMatchGate`** — гейт не знает extract-бюджет; ретрай должен жить в вызывающем.
- **Re-pick по нерелевантным кандидатам** (ниже brand_line-порога) или когда options пусты —
  воскрешает слабые совпадения.

## Безопасность / что НЕ трогаем

- **Sentinel-инвариант (БЛОКИРУЮЩИЙ)**: Mi Band 8 против пула [Mi Band 7, Mi Band 6] (обе
  DIFFERENT) ОБЯЗАН вернуть None — галлюн-донор не воскрешаем через re-pick. Закреплён тестом
  `test_sentinel_all_different_abstains`.
- Кап LLM=3 — не поднимать без пере-обоснования (стоимость).
- Направление fail-safe: после двойного UNKNOWN — ДРОП, не приём.

## Риски и митигации

- **Re-pick воскрешает галлюн**: митигация — re-pick только среди ≥brand_line кандидатов +
  каждый заново через гейт + sentinel-тест.
- **Бюджет-исчерпание молча роняет донора**: логируется `donor_gate_stats` с `cap_hit`.
- Оракул `test_wb_donor_gate_repick.py` (8 кейсов): first_same, different→repick,
  sentinel→abstain, unknown→retry, double_unknown→drop→next, budget_cap, clean_no_llm,
  skip_low_score. Регресс WB/donor 272 passed.

## План (диспетчеризация)

Код — tier-0 (deepseek) verbatim-спеком в 3 шага (donor_gate enum → `_rank_cards` рефактор →
`_GateBudget`+`_select_gated_card`); Opus приземлял, ловил mangling (round-1 deepseek сломал
`_rank_cards` — переспек verbatim). Архитектура — fable (детальный мандат).

## Допущения (проверить менеджеру)

- **ФЛАГНУТОЕ ОТКЛОНЕНИЕ**: article-path НЕ унифицирован под общий `_GateBudget` — остался на
  обёртке `is_same_product` (поведение цело + фикс fail-open-кэша), на DIFFERENT падает в
  re-pick'нутый main-path. Полная унификация обоих call-site под общий budget = отдельный
  follow-up (высокий риск рефактора / малая доп. ценность). Осознанно отложено.
