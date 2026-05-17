# Architecture docs

Здесь живут диаграммы и описания архитектуры cpAiFeatures.

## Чем рисуем

**Mermaid** — основной инструмент. GitHub рендерит автоматически. Открой любой `.md` файл здесь на GitHub — увидишь картинки.

Для редактирования:
- [mermaid.live](https://mermaid.live) — paste код, see preview, export PNG/SVG/draw.io
- VSCode extension `Markdown Preview Mermaid Support`
- IntelliJ / PyCharm: plugin `Mermaid`

**draw.io / diagrams.net** — для сложных схем. Файлы `.drawio` открываются в [app.diagrams.net](https://app.diagrams.net) или desktop приложении.

## Файлы

| Файл | Что показывает |
|---|---|
| [pipeline.md](pipeline.md) | Главный extraction pipeline — sequential stages, LLM-call counts, decision gates |
| [data-flow.md](data-flow.md) | Какие данные передаются между стадиями (JSON examples), типы Python |
| [cost-breakdown.md](cost-breakdown.md) | Сколько LLM-calls и денег стоит обработка одного товара в разных сценариях |

## Правила обновления

- При архитектурных изменениях — обновляй здесь до начала кодинга
- 1 диаграмма = 1 уровень абстракции (не мешать классы и flow на одной)
- Подписывай rectangles: «название» + «(1 LLM call)» если применимо
- Цветовая convention:
  - 🔵 синий: existing/stable
  - 🟢 зелёный: cheap / no LLM
  - 🟡 жёлтый: gate/decision
  - 🟣 фиолетовый: vision / image
  - 🔴 красный: expensive (web search, full pipeline)
