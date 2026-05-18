# Real-world product fixtures research

Собрано 2026-05-14. Используется для acceptance testing AI-pipeline заполнения характеристик.

Покрытие sources:
- **DescriptionSource** — товары 01, 02 (богатое описание от производителя)
- **LlmKnowledgeSource** — товары 03, 04 (известные бренды, LLM знает specs)
- **VisionSource** — товары 05, 06 (визуальные атрибуты, цвет, форма, материал)
- **WebSearchSource** — товары 07, 08 (точные технические цифры)
- **Mixed / edge cases** — товары 09, 10 (handmade без EAN; плохой листинг WB)

---

## 01_iphone_15_pro_titanium

**Тип:** Description-rich (официальная страница apple.com — исчерпывающее описание)

### Product
- **Name:** Apple iPhone 15 Pro 256GB Natural Titanium
- **Brand:** Apple
- **EAN/UPC:** не публикуется Apple открыто; retailers используют model# MQUD3LL/A (256GB Natural Titanium)
- **Category:** Электроника / Смартфоны / Apple iPhone
- **Product URL:** https://support.apple.com/en-us/111829
- **Image URLs:**
  - https://fdn2.gsmarena.com/vv/bigpic/apple-iphone-15-pro.jpg *(GSMArena CDN, стабильный)*
  - https://picsum.photos/seed/iphone15pro/400/400 *(placeholder для vision-тестов)*
- **Description (как на WB):**
  > Apple iPhone 15 Pro 256GB, цвет «натуральный титан». Чип A17 Pro, камера 48 Мп с трёхкратным оптическим зумом. Корпус из авиационного титана, экран 6.1" Super Retina XDR ProMotion 120 Гц.

### Ground truth attributes
| id | name | type | semantic | expected | tolerance | source URL | best source |
|---|---|---|---|---|---|---|---|
| 1 | Бренд | text | brand | Apple | — | https://support.apple.com/en-us/111829 | description |
| 2 | Модель | text | — | iPhone 15 Pro | — | https://support.apple.com/en-us/111829 | description |
| 3 | Объём памяти | text | storage | 256 ГБ | — | https://support.apple.com/en-us/111829 | description |
| 4 | Цвет | text | color | Натуральный титан | — | https://support.apple.com/en-us/111829 | description+vision |
| 5 | Вес (г) | numeric | weight | 187 | 3 | https://support.apple.com/en-us/111829 | web_search |
| 6 | Диагональ экрана (дюймы) | numeric | screen_size | 6.1 | 0.1 | https://support.apple.com/en-us/111829 | description |
| 7 | Чип | text | — | A17 Pro | — | https://support.apple.com/en-us/111829 | description |
| 8 | Материал корпуса | text | material | Титан | — | https://support.apple.com/en-us/111829 | knowledge |
| 9 | Основная камера (МП) | numeric | — | 48 | 0 | https://support.apple.com/en-us/111829 | description |
| 10 | Тип экрана | text | — | OLED | — | https://support.apple.com/en-us/111829 | knowledge |

---

## 02_lego_classic_10696

**Тип:** Description-rich (FMCG с EAN, стабильные данные)

### Product
- **Name:** LEGO Classic Medium Creative Brick Box 10696
- **Brand:** LEGO
- **EAN:** 5702015357180
- **UPC:** 673419233590
- **Category:** Игрушки / Конструкторы / LEGO Classic
- **Product URL:** https://www.lego.com/en-us/product/lego-medium-creative-brick-box-10696
- **Image URLs:**
  - https://m.media-amazon.com/images/I/81ZNaZ7WEFL._AC_SX425_.jpg *(Amazon CDN, стабильный)*
  - https://picsum.photos/seed/lego10696/400/400 *(placeholder)*
- **Description (как на WB):**
  > LEGO Classic 10696, 484 детали, 35 цветов кирпичиков. Набор для свободного творчества в пластиковом контейнере со съёмной крышкой. Для детей от 4 лет.

### Ground truth attributes
| id | name | type | semantic | expected | tolerance | source URL | best source |
|---|---|---|---|---|---|---|---|
| 1 | Бренд | text | brand | LEGO | — | https://www.lego.com/en-us/product/lego-medium-creative-brick-box-10696 | description |
| 2 | Номер набора | text | — | 10696 | — | https://www.lego.com/en-us/product/lego-medium-creative-brick-box-10696 | description |
| 3 | Количество деталей | numeric | — | 484 | 0 | https://www.lego.com/en-us/product/lego-medium-creative-brick-box-10696 | description |
| 4 | Серия | text | — | Classic | — | https://www.lego.com/en-us/product/lego-medium-creative-brick-box-10696 | description |
| 5 | Возраст (от лет) | numeric | — | 4 | 0 | https://www.lego.com/en-us/product/lego-medium-creative-brick-box-10696 | description |
| 6 | Вес упаковки (кг) | numeric | weight | 1.04 | 0.05 | https://brickset.com/sets/10696-1/Medium-Creative-Brick-Box | web_search |
| 7 | EAN | text | — | 5702015357180 | — | https://brickset.com/sets/10696-1/Medium-Creative-Brick-Box | web_search |
| 8 | Количество цветов деталей | numeric | — | 35 | 2 | https://www.lego.com/en-us/product/lego-medium-creative-brick-box-10696 | description |

---

## 03_nike_air_force_1_white

**Тип:** Knowledge-only (иконический силуэт — LLM знает материалы и историю без description)

### Product
- **Name:** Nike Air Force 1 '07 Men's Shoes White/White
- **Brand:** Nike
- **Style code:** CW2288-111
- **EAN/UPC:** 0883412740906 (размер 8.5 US; по размерам варьируется)
- **Category:** Обувь / Кроссовки / Nike
- **Product URL:** https://www.nike.com/t/air-force-1-07-mens-shoes-jBrhbr
- **Image URLs:**
  - https://static.nike.com/a/images/t_PDP_1728_v1/f_auto,q_auto:eco/af2e4099-87be-4f49-9f1f-36c7c1eb97ee/air-force-1-07-shoes-WrLlWX.png *(Nike static CDN)*
  - https://picsum.photos/seed/nikeaf1/400/400 *(placeholder)*
- **Description (как на WB):**
  > Nike Air Force 1 '07, белые/белые. Натуральная кожа, перфорированный носок. Подошва с воздушной подушкой Nike Air. Культовый баскетбольный силуэт 1982 года.

### Ground truth attributes
| id | name | type | semantic | expected | tolerance | source URL | best source |
|---|---|---|---|---|---|---|---|
| 1 | Бренд | text | brand | Nike | — | https://www.nike.com/t/air-force-1-07-mens-shoes-jBrhbr | knowledge |
| 2 | Модель | text | — | Air Force 1 '07 | — | https://www.nike.com/t/air-force-1-07-mens-shoes-jBrhbr | knowledge |
| 3 | Артикул | text | — | CW2288-111 | — | https://www.nike.com/gb/t/air-force-1-07-shoes-ojDkV4tL/CW2288-111 | description |
| 4 | Цвет | text | color | Белый | — | https://www.nike.com/t/air-force-1-07-mens-shoes-jBrhbr | vision |
| 5 | Материал верха | text | material | Кожа | — | https://www.nike.com/t/air-force-1-07-mens-shoes-jBrhbr | knowledge |
| 6 | Материал подошвы | text | — | Резина | — | https://www.nike.com/t/air-force-1-07-mens-shoes-jBrhbr | knowledge |
| 7 | Тип застёжки | text | — | Шнуровка | — | https://www.nike.com/t/air-force-1-07-mens-shoes-jBrhbr | vision |
| 8 | Высота | text | — | Низкие (Low) | — | https://www.nike.com/t/air-force-1-07-mens-shoes-jBrhbr | knowledge+vision |

---

## 04_sony_wh1000xm5

**Тип:** Knowledge + Mixed (известные наушники — LLM знает ключевые specs, но точные числа из web_search)

### Product
- **Name:** Sony WH-1000XM5 Wireless Noise Cancelling Headphones Black
- **Brand:** Sony
- **Model:** WH-1000XM5
- **Category:** Электроника / Наушники / Беспроводные
- **Product URL:** https://helpguide.sony.net/mdr/wh1000xm5/v1/en/contents/TP1000541014.html
- **Image URLs:**
  - https://m.media-amazon.com/images/I/71o8Q5XJS5L._AC_SX679_.jpg *(Amazon CDN)*
  - https://picsum.photos/seed/sonywh1000xm5/400/400 *(placeholder)*
- **Description (как на WB):**
  > Sony WH-1000XM5, беспроводные наушники с ANC. Bluetooth 5.2, кодеки SBC/AAC/LDAC. 30 часов автономной работы. Вес 250 г, складная конструкция.

### Ground truth attributes
| id | name | type | semantic | expected | tolerance | source URL | best source |
|---|---|---|---|---|---|---|---|
| 1 | Бренд | text | brand | Sony | — | https://helpguide.sony.net/mdr/wh1000xm5/v1/en/contents/TP1000541014.html | knowledge |
| 2 | Модель | text | — | WH-1000XM5 | — | https://helpguide.sony.net/mdr/wh1000xm5/v1/en/contents/TP1000541014.html | description |
| 3 | Вес (г) | numeric | weight | 250 | 5 | https://helpguide.sony.net/mdr/wh1000xm5/v1/en/contents/TP1000541014.html | web_search |
| 4 | Версия Bluetooth | text | — | 5.2 | — | https://helpguide.sony.net/mdr/wh1000xm5/v1/en/contents/TP1000541014.html | web_search |
| 5 | Поддерживаемые кодеки | text | — | SBC, AAC, LDAC | — | https://helpguide.sony.net/mdr/wh1000xm5/v1/en/contents/TP1000541014.html | web_search |
| 6 | Время зарядки (ч) | numeric | — | 3.5 | 0.2 | https://helpguide.sony.net/mdr/wh1000xm5/v1/en/contents/TP1000541014.html | web_search |
| 7 | Время работы (ч) | numeric | — | 30 | 1 | https://helpguide.sony.net/mdr/wh1000xm5/v1/en/contents/TP1000541014.html | knowledge |
| 8 | Шумоподавление | text | — | Активное (ANC) | — | https://helpguide.sony.net/mdr/wh1000xm5/v1/en/contents/TP1000541014.html | knowledge |
| 9 | Конструкция | text | — | Накладные (Over-ear) | — | https://helpguide.sony.net/mdr/wh1000xm5/v1/en/contents/TP1000541014.html | vision |

---

## 05_zara_basic_cotton_tshirt_white

**Тип:** Vision-critical (одежда — цвет, фасон, тип выреза определяются по фото)

### Product
- **Name:** BASIC COTTON T-SHIRT White
- **Brand:** Zara
- **Article/SKU:** p03253332
- **Category:** Одежда / Женские футболки / Базовые
- **Product URL:** https://www.zara.com/us/en/basic-cotton-t-shirt-p03253332.html
- **Image URLs:**
  - https://static.zara.net/assets/public/07bf/3571/56ff41c1b5f0/f617bafebbe2/03253332250-p/03253332250-p.jpg?ts=1700000000000&w=850 *(Zara CDN — NOTE: может ротироваться; используй placeholder если 404)*
  - https://picsum.photos/seed/zaratshirt/400/600 *(placeholder — вертикальное фото одежды)*
- **Description (как на WB):**
  > Zara базовая хлопковая футболка, белая. Круглый вырез, короткий рукав, свободный крой. Состав: 100% хлопок.

### Ground truth attributes
| id | name | type | semantic | expected | tolerance | source URL | best source |
|---|---|---|---|---|---|---|---|
| 1 | Бренд | text | brand | Zara | — | https://www.zara.com/us/en/basic-cotton-t-shirt-p03253332.html | vision |
| 2 | Цвет | text | color | Белый | — | https://www.zara.com/us/en/basic-cotton-t-shirt-p03253332.html | vision |
| 3 | Состав ткани | text | material | 100% хлопок | — | https://www.zara.com/us/en/basic-cotton-t-shirt-p03253332.html | description |
| 4 | Тип выреза | text | — | Круглый | — | https://www.zara.com/us/en/basic-cotton-t-shirt-p03253332.html | vision |
| 5 | Тип рукава | text | — | Короткий | — | https://www.zara.com/us/en/basic-cotton-t-shirt-p03253332.html | vision |
| 6 | Крой | text | — | Свободный (regular) | — | https://www.zara.com/us/en/basic-cotton-t-shirt-p03253332.html | description |
| 7 | Пол | text | — | Женский | — | https://www.zara.com/us/en/basic-cotton-t-shirt-p03253332.html | description |
| 8 | Тип изделия | text | — | Футболка | — | https://www.zara.com/us/en/basic-cotton-t-shirt-p03253332.html | vision |

---

## 06_apple_airpods_pro_2

**Тип:** Vision + Knowledge + Description (mixed — форм-фактор из фото, specs из знаний+офиц)

### Product
- **Name:** Apple AirPods Pro (2nd Generation) with MagSafe Case (USB-C) White
- **Brand:** Apple
- **Model:** MTJV3LL/A
- **Category:** Электроника / Наушники / True Wireless
- **Product URL:** https://support.apple.com/en-us/111851
- **Image URLs:**
  - https://m.media-amazon.com/images/I/61SUj2aKoEL._AC_SX679_.jpg *(Amazon CDN)*
  - https://picsum.photos/seed/airpodspro2/400/400 *(placeholder)*
- **Description (как на WB):**
  > Apple AirPods Pro 2 с MagSafe зарядным кейсом USB-C. Чип H2, активное шумоподавление в 2× лучше предыдущего поколения. До 6 часов прослушивания, всего до 30 часов с кейсом. IPX4, Bluetooth 5.3.

### Ground truth attributes
| id | name | type | semantic | expected | tolerance | source URL | best source |
|---|---|---|---|---|---|---|---|
| 1 | Бренд | text | brand | Apple | — | https://support.apple.com/en-us/111851 | knowledge |
| 2 | Модель | text | — | AirPods Pro (2nd Generation) | — | https://support.apple.com/en-us/111851 | description |
| 3 | Вес одного наушника (г) | numeric | weight | 5.3 | 0.2 | https://support.apple.com/en-us/111851 | web_search |
| 4 | Вес кейса (г) | numeric | weight | 50.8 | 1 | https://support.apple.com/en-us/111851 | web_search |
| 5 | Чип | text | — | H2 | — | https://support.apple.com/en-us/111851 | knowledge |
| 6 | Версия Bluetooth | text | — | 5.3 | — | https://support.apple.com/en-us/111851 | web_search |
| 7 | Время работы от зарядки (ч) | numeric | — | 6 | 0.5 | https://support.apple.com/en-us/111851 | description |
| 8 | Суммарное время с кейсом (ч) | numeric | — | 30 | 1 | https://support.apple.com/en-us/111851 | description |
| 9 | Влагозащита | text | — | IPX4 | — | https://support.apple.com/en-us/111851 | web_search |
| 10 | Цвет | text | color | Белый | — | https://support.apple.com/en-us/111851 | vision |

---

## 07_dyson_v15_detect

**Тип:** WebSearch-critical (точные технические характеристики — suction AW, вес, объём — нужен поиск)

### Product
- **Name:** Dyson V15 Detect Cordless Vacuum Cleaner Yellow/Iron
- **Brand:** Dyson
- **Model:** V15 Detect
- **Category:** Бытовая техника / Пылесосы / Беспроводные
- **Product URL:** https://www.dyson.com/vacuum-cleaners/cordless/v15/detect-yellow
- **Image URLs:**
  - https://m.media-amazon.com/images/I/71OtNc2IAWL._AC_SX679_.jpg *(Amazon CDN)*
  - https://picsum.photos/seed/dysonv15/400/400 *(placeholder)*
- **Description (как на WB):**
  > Dyson V15 Detect, беспроводной пылесос. Мощность всасывания 240 AW, двигатель Hyperdymium 125 000 об/мин. До 60 минут работы, объём контейнера 0.76 л, фильтрация HEPA, лазерная подсветка пыли.

### Ground truth attributes
| id | name | type | semantic | expected | tolerance | source URL | best source |
|---|---|---|---|---|---|---|---|
| 1 | Бренд | text | brand | Dyson | — | https://www.dyson.com/vacuum-cleaners/cordless/v15/detect-yellow | knowledge |
| 2 | Модель | text | — | V15 Detect | — | https://www.dyson.com/vacuum-cleaners/cordless/v15/detect-yellow | description |
| 3 | Мощность всасывания (AW) | numeric | — | 240 | 5 | https://www.dyson.com/vacuum-cleaners/cordless/v15/detect-yellow | web_search |
| 4 | Время работы (мин) | numeric | — | 60 | 5 | https://www.dyson.com/vacuum-cleaners/cordless/v15/detect-yellow | web_search |
| 5 | Объём контейнера (л) | numeric | — | 0.76 | 0.05 | https://www.manua.ls/dyson/v15-detect/specifications | web_search |
| 6 | Вес (кг) | numeric | weight | 2.96 | 0.1 | https://www.manua.ls/dyson/v15-detect/specifications | web_search |
| 7 | Скорость двигателя (об/мин) | numeric | — | 125000 | 1000 | https://www.dyson.com/vacuum-cleaners/cordless/v15/detect-yellow | web_search |
| 8 | Тип фильтрации | text | — | HEPA | — | https://www.dyson.com/vacuum-cleaners/cordless/v15/detect-yellow | description |
| 9 | Цвет | text | color | Жёлтый/Железо | — | https://www.dyson.com/vacuum-cleaners/cordless/v15/detect-yellow | vision |

---

## 08_xiaomi_14_black

**Тип:** WebSearch-critical + Multilingual brand (китайский вендор, точные specs требуют поиска)

### Product
- **Name:** Xiaomi 14 12GB+256GB Black
- **Brand:** Xiaomi
- **Category:** Электроника / Смартфоны / Xiaomi
- **Product URL:** https://www.mi.com/global/product/xiaomi-14/specs/
- **Image URLs:**
  - https://i01.appmifile.com/v1/MI_18455B3E/pms_1699267200.18873.jpg *(Xiaomi CDN — может меняться; используй placeholder если 404)*
  - https://picsum.photos/seed/xiaomi14/400/400 *(placeholder)*
- **Description (как на WB):**
  > Xiaomi 14, смартфон. Процессор Snapdragon 8 Gen 3, камера Leica 50 МП (тройная), дисплей LTPO OLED 6.36" 120 Гц. Аккумулятор 4610 мАч, IP68. Цвет: чёрный.

### Ground truth attributes
| id | name | type | semantic | expected | tolerance | source URL | best source |
|---|---|---|---|---|---|---|---|
| 1 | Бренд | text | brand | Xiaomi | — | https://www.mi.com/global/product/xiaomi-14/ | knowledge |
| 2 | Модель | text | — | Xiaomi 14 | — | https://www.mi.com/global/product/xiaomi-14/ | description |
| 3 | Процессор | text | — | Snapdragon 8 Gen 3 | — | https://www.gsmarena.com/xiaomi_14-12626.php | web_search |
| 4 | Диагональ экрана (дюймы) | numeric | screen_size | 6.36 | 0.05 | https://www.gsmarena.com/xiaomi_14-12626.php | web_search |
| 5 | Тип экрана | text | — | LTPO OLED | — | https://www.gsmarena.com/xiaomi_14-12626.php | web_search |
| 6 | Основная камера (МП) | numeric | — | 50 | 0 | https://www.mi.com/global/product/xiaomi-14/specs/ | description |
| 7 | Ёмкость аккумулятора (мАч) | numeric | — | 4610 | 10 | https://www.gsmarena.com/xiaomi_14-12626.php | web_search |
| 8 | Вес (г) | numeric | weight | 193 | 3 | https://www.gsmarena.com/xiaomi_14-12626.php | web_search |
| 9 | Влагозащита | text | — | IP68 | — | https://www.gsmarena.com/xiaomi_14-12626.php | web_search |
| 10 | Цвет | text | color | Чёрный | — | https://www.mi.com/global/product/xiaomi-14/ | vision |

---

## 09_etsy_handmade_candle

**Тип:** Handmade edge case — нет EAN, нет структурированных характеристик, WebSearch вернёт мало; описание от продавца неструктурировано

### Product
- **Name:** Handmade Lavender Vanilla Soy Candle — Relaxing Aromatherapy Gift
- **Brand:** LCCandleCottage *(Etsy seller)*
- **EAN:** отсутствует (handmade, не зарегистрирован в GS1)
- **Category:** Товары для дома / Свечи / Ароматические
- **Product URL:** https://www.etsy.com/listing/593118903/lavender-vanilla-soy-candle-relaxing
- **Image URLs:**
  - https://picsum.photos/seed/soycandlelavender/400/400 *(placeholder — Etsy блокирует hotlinking)*
  - https://picsum.photos/seed/soycandlejar/400/500 *(placeholder)*
- **Description (как на WB / как написал бы продавец):**
  > Ароматическая соевая свеча ручной работы, аромат «лаванда и ваниль». 100% натуральный соевый воск, хлопковый фитиль без свинца. Объём 16 oz (455 г), время горения 100 часов. Идеальный подарок.

### Ground truth attributes
| id | name | type | semantic | expected | tolerance | source URL | best source |
|---|---|---|---|---|---|---|---|
| 1 | Тип изделия | text | — | Свеча | — | https://www.etsy.com/listing/593118903/lavender-vanilla-soy-candle-relaxing | description |
| 2 | Аромат | text | — | Лаванда и ваниль | — | https://www.etsy.com/listing/593118903/lavender-vanilla-soy-candle-relaxing | description |
| 3 | Тип воска | text | material | Соевый | — | https://www.etsy.com/listing/593118903/lavender-vanilla-soy-candle-relaxing | description |
| 4 | Тип фитиля | text | — | Хлопковый | — | https://www.etsy.com/listing/593118903/lavender-vanilla-soy-candle-relaxing | description |
| 5 | Вес нетто (г) | numeric | weight | 455 | 20 | https://www.etsy.com/listing/593118903/lavender-vanilla-soy-candle-relaxing | description |
| 6 | Время горения (ч) | numeric | — | 100 | 10 | https://www.etsy.com/listing/593118903/lavender-vanilla-soy-candle-relaxing | description |

**Примечание для теста:** EAN = null — pipeline должен не падать при отсутствии штрихкода. WebSearch вернёт общую информацию о соевых свечах, но не точные характеристики именно этого изделия. Ожидаемое поведение: источники description + knowledge, web_search даёт generic данные (low confidence).

---

## 10_wb_poor_listing_phone_case

**Тип:** Edge case «плохой листинг» — типичный WB-продавец с нулевым описанием, без характеристик

### Product
- **Name:** Чехол для телефона (iPhone 13/14/15) прозрачный силиконовый
- **Brand:** NoName *(частная марка, не зарегистрированная)*
- **EAN:** не заполнен продавцом
- **Category:** Аксессуары / Чехлы для телефонов
- **Product URL:** https://www.wildberries.ru/catalog/0/search.aspx?search=чехол+силиконовый+прозрачный+iphone
  *(Wildberries поиск — конкретная карточка меняется; URL иллюстративный)*
- **Image URLs:**
  - https://picsum.photos/seed/phonecase/300/500 *(placeholder — фото нет стабильного)*
- **Description (НАМЕРЕННО ПЛОХОЕ — как в реальных плохих листингах WB):**
  > Чехол. Хорошая вещь, подходит для телефона. Прозрачный. Советую!

### Ground truth attributes
*(Реальные значения которые pipeline ДОЛЖЕН угадать из vision + knowledge, несмотря на нулевой description)*

| id | name | type | semantic | expected | tolerance | source URL | best source |
|---|---|---|---|---|---|---|---|
| 1 | Тип изделия | text | — | Чехол | — | *(из description)* | description |
| 2 | Цвет | text | color | Прозрачный | — | *(из description)* | vision |
| 3 | Материал | text | material | Силикон | — | *(из knowledge — типичный материал прозрачных чехлов)* | knowledge |
| 4 | Совместимые модели | text | — | iPhone 13, iPhone 14, iPhone 15 | — | *(из description)* | description |
| 5 | Бренд | text | brand | NoName / Без бренда | — | *(не определяемо)* | knowledge |

**Примечание для теста:** Description score должен быть низким (мало информации). Pipeline должен fallback на knowledge + vision. Ожидается что материал «силикон» придёт из LlmKnowledgeSource (типичный материал для прозрачных чехлов), а не из description. Атрибуты бренд и EAN — expected failures с low confidence.

---

## Summary

| # | Slug | Product | Attrs count | Primary expected source | Edge case |
|---|---|---|---|---|---|
| 01 | iphone_15_pro_titanium | Apple iPhone 15 Pro 256GB Natural Titanium | 10 | description | — |
| 02 | lego_classic_10696 | LEGO Classic Medium Creative Brick Box 10696 | 8 | description | EAN верифицирован |
| 03 | nike_air_force_1_white | Nike Air Force 1 '07 White/White CW2288-111 | 8 | knowledge | — |
| 04 | sony_wh1000xm5 | Sony WH-1000XM5 Black | 9 | mixed (knowledge+web_search) | — |
| 05 | zara_basic_tshirt_white | Zara Basic Cotton T-Shirt White | 8 | vision | Zara блокирует direct fetch |
| 06 | apple_airpods_pro_2 | Apple AirPods Pro 2 (MTJV3LL/A) White | 10 | mixed (all sources) | — |
| 07 | dyson_v15_detect | Dyson V15 Detect Yellow/Iron | 9 | web_search | Specs только на официальном PDF |
| 08 | xiaomi_14_black | Xiaomi 14 12+256GB Black | 10 | web_search | Multilingual brand |
| 09 | etsy_handmade_candle | LCCandleCottage Lavender Vanilla Soy Candle | 6 | description | Нет EAN; handmade |
| 10 | wb_poor_listing_phone_case | NoName силиконовый чехол iPhone (плохой листинг) | 5 | knowledge+vision | Намеренно пустой description |

**Итого:** 10 товаров, 83 ground truth атрибута.

---

## Notes on image URLs

- **Apple, Nike, Zara** — официальные CDN блокируют прямой hotlink или требуют auth. Для production vision-тестов использовать реальные HTTPS-URL с Amazon CDN (`m.media-amazon.com`) или GSMArena CDN (`fdn2.gsmarena.com`) — они стабильны.
- **Etsy, Wildberries** — изображения недоступны для внешнего fetch без авторизации. Используются `picsum.photos` плейсхолдеры с фиксированными seed; они возвращают стабильные реальные фото (ландшафт/объекты) пригодные для проверки что VisionSource вообще работает.
- **Zara CDN URL** помечен как ненадёжный — URL включает `ts` параметр который протухает. При наличии инфраструктуры рекомендуется сохранить фото локально в `fixtures/images/`.

## Notes on EAN coverage

- Верифицированный EAN: только LEGO 10696 (5702015357180 / 673419233590 — подтверждён Brickset + Amazon)
- Apple не публикует EAN на сайте; model# (MTJV3LL/A, MQUD3LL/A) является функциональным эквивалентом
- Nike EAN varies per size — 0883412740906 для размера 8.5 US
- Xiaomi, Dyson, Zara, Sony — EAN не верифицирован в открытых источниках без специального barcode DB

## Sources used

- https://support.apple.com/en-us/111829 (iPhone 15 Pro specs)
- https://support.apple.com/en-us/111851 (AirPods Pro 2 specs)
- https://helpguide.sony.net/mdr/wh1000xm5/v1/en/contents/TP1000541014.html (WH-1000XM5 specs)
- https://brickset.com/sets/10696-1/Medium-Creative-Brick-Box (LEGO 10696 EAN + dims)
- https://www.gsmarena.com/xiaomi_14-12626.php (Xiaomi 14 specs)
- https://www.dyson.com/vacuum-cleaners/cordless/v15/detect-yellow (Dyson V15 product page)
- https://www.manua.ls/dyson/v15-detect/specifications (Dyson V15 dims + weight)
- https://www.zara.com/us/en/basic-cotton-t-shirt-p03253332.html (Zara t-shirt)
- https://www.nike.com/t/air-force-1-07-mens-shoes-jBrhbr (Nike AF1)
- https://www.mi.com/global/product/xiaomi-14/specs/ (Xiaomi 14 official)
- https://www.etsy.com/listing/593118903/lavender-vanilla-soy-candle-relaxing (Etsy candle)
- https://eandata.com/lookup/0883412740906/ (Nike EAN by size)
