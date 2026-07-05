# Сценарий Talc Annotation UI для образца 2550382-1-10x

Назначение: режиссёрский сценарий для новой записи Talc Annotation UI. Сценарий показывает только один use case: как эксперт открывает образец `/sample/2550382-1-10x`, смотрит исходные данные, уточняет тальк и сравнивает ручную разметку с эвристикой и нейросетевой моделью.

Основано на предыдущей версии `talc_annotation_script_ru.md` и базовом тексте `script_ru.md`, но старый общий 36-сегментный тур заменён на конкретный сценарий по одному образцу. Английские названия UI оставлены в `screen`-разметке; дикторский текст написан по-русски, чтобы TTS звучал естественнее.

Источники для формулировок:

- предыдущий сценарий: `presentation/videos/demo_video_talc_ui_20260704/script_ru.md`;
- слайд Google Slides: `Размечаем для обучения ML модели SegFormer-B0`;
- Q&A по методологии талька: `docs/official/2026-07-04-qna-session-4-transcript.md`.

## Синхронизация со слайдом

Сценарий должен совпадать с логикой слайда `Размечаем для обучения ML модели SegFormer-B0`:

- ручные синие линии организаторов = приблизительная разметка зон талька;
- положительная область = не пиксельная истина, а слабая зона, где тальк вероятен;
- эксперт размечает тёмные хлопья в нерудной матрице, а не любое тёмное пятно;
- эвристика показывает объяснимую сегментацию областей талька;
- нейросетевая модель показывает ML-сегментацию талька;
- официальная Q&A #4: для изображений с синими линиями нет лабораторной или экспертной численной доли талька, поэтому команда должна явно описывать своё определение талька и сохранять порог 10% как правило класса.

## Разметка экрана

Каждый блок содержит:

- `route`: адрес Talc UI;
- `capture_filename`: имя скриншота, который нужно снять для монтажа;
- `pre_capture`: что сделать перед скриншотом;
- `ui_state`: состояние слоёв, классов и режима сравнения;
- `must_contain`: что должно быть видно в кадре;
- `do_not`: запреты для записи.

Все скриншоты сохранять в:

`presentation/videos/demo_video_talc_ui_20260704/screenshots/sample_2550382_1_10x/`

На видео оставить маленький таймкод в углу для синхронизации звука, субтитров и кадров.

## 00. Подготовка перед записью

```screen
route: /sample/2550382-1-10x
pre_capture:
  - открыть Talc UI напрямую на sample route
  - дождаться загрузки образца "2550382-1 10x"
  - включить Fit view, чтобы весь шлиф был читаем в viewer
  - держать на экране только Talc UI, без браузерных вкладок и других приложений
ui_state:
  display_layers:
    Background: on
    Blank White: off
    Original blue lines: off
    Talc cluster areas: off
    Sulfides: off
  segmentation_classes:
    Positive bag: off
    Talc: off
    Not Talc: off
  dark_pixel_preview_threshold: off
  comparison_mode: Current
  active_tool: Brush
do_not:
  - не нажимать Save
  - не нажимать Save & Next
```

## 01. Оригинальное изображение [00:00-00:13]

```screen
time: 00:00-00:13
route: /sample/2550382-1-10x
capture_filename: 01_original_background_only.png
pre_capture:
  - проверить, что включён только display layer Background
  - выключить все segmentation classes
ui_state:
  display_layers:
    Background: on
    Original blue lines: off
    Talc cluster areas: off
    Sulfides: off
  segmentation_classes:
    Positive bag: off
    Talc: off
    Not Talc: off
  comparison_mode: Current
must_contain:
  - чистая исходная фотография шлифа
  - top-right Display layers с включённым Background
  - left Segmentation classes без включённых классов
  - corner time mark
```

Открываем образец из датасета хакатона. Сейчас на экране только исходная фотография шлифа: без классов, без подсказок и без результатов модели. Это та картинка, с которой начинается экспертная проверка.

## 02. Исходные синие линии [00:13-00:30]

```screen
time: 00:13-00:30
route: /sample/2550382-1-10x
capture_filename: 02_original_blue_lines.png
pre_capture:
  - включить display layer Original blue lines
  - оставить segmentation classes выключенными
ui_state:
  display_layers:
    Background: on
    Original blue lines: on
    Talc cluster areas: off
    Sulfides: off
  segmentation_classes:
    Positive bag: off
    Talc: off
    Not Talc: off
  comparison_mode: Current
must_contain:
  - фотография шлифа
  - исходные синие линии поверх фотографии
  - видимый переключатель Original blue lines в Display layers
```

Теперь включаем исходные синие линии. Это грубая ручная разметка, сделанная поверх изображения в графическом редакторе. Она полезна как стартовая подсказка, но это не пиксельная маска талька и не точный экспертный контур.

## 03. Положительная область [00:30-00:55]

```screen
time: 00:30-00:55
route: /sample/2550382-1-10x
capture_filename: 03_positive_bag_from_blue_lines.png
pre_capture:
  - включить segmentation class Positive bag
  - Original blue lines оставить включённым первые 2-3 секунды, затем можно выключить для чистого кадра
ui_state:
  display_layers:
    Background: on
    Original blue lines: optional_on_for_context
    Talc cluster areas: off
    Sulfides: off
  segmentation_classes:
    Positive bag: on
    Talc: off
    Not Talc: off
  comparison_mode: Current
must_contain:
  - область Positive bag поверх исходного шлифа
  - понятная связь между синими линиями и заполненной областью
  - left Segmentation classes: Positive bag enabled, Talc disabled
```

Из этих линий инструмент строит положительную область. Мы называем её областью-кандидатом: внутри тальк вероятен, но мы не утверждаем, что каждый пиксель внутри уже тальк. Это просто зона поиска, где эксперту и алгоритму есть смысл более внимательно смотреть.

## 04. Почему это не доля талька [00:55-01:17]

```screen
time: 00:55-01:17
route: /sample/2550382-1-10x
capture_filename: 04_positive_bag_methodology_note.png
pre_capture:
  - удержать кадр на Positive bag
  - крупно показать процент Positive bag, если он виден в left widget
ui_state:
  display_layers:
    Background: on
    Original blue lines: off
  segmentation_classes:
    Positive bag: on
    Talc: off
    Not Talc: off
  comparison_mode: Current
must_contain:
  - Positive bag как широкая зона-кандидат
  - Talc class выключен
  - не должно выглядеть как финальная маска талька
source_reference:
  - docs/official/2026-07-04-qna-session-4-transcript.md, Q&A #4, timestamps 29:44-31:03 and 43:40-46:33
```

Важно: эта площадь не равна доле талька. В официальной сессии вопросов и ответов организаторы не дали даже выборочных численных долей талька для таких изображений. Поэтому мы явно фиксируем свою методологию: тальк ищем как тёмные хлопья в нерудной матрице, а порог десять процентов используем уже для итогового правила класса.

## 05. Предпросмотр тёмных пикселей [01:17-01:39]

```screen
time: 01:17-01:39
route: /sample/2550382-1-10x
capture_filename: 05_dark_pixel_preview_threshold_50.png
pre_capture:
  - выключить Original blue lines, если они ещё включены
  - установить Dark pixel preview threshold = 50
  - оставить Positive bag visible для контекста
ui_state:
  display_layers:
    Background: on
    Original blue lines: off
  segmentation_classes:
    Positive bag: on
    Talc: off
    Not Talc: off
  dark_pixel_preview_threshold: 50
  comparison_mode: Current
must_contain:
  - правый контрол Dark pixel preview threshold со значением 50
  - затемнённый или подсвеченный preview тёмных кандидатов
  - Positive bag как граница поиска
```

Дальше включаем предпросмотр тёмных пикселей и ставим порог пятьдесят. Это не меняет разметку. Слой только помогает глазу: яркие участки уходят на второй план, а тёмные кандидаты становятся заметнее. Это согласуется со слайдом: геолог ищет не любое пятно, а тёмные хлопья в нерудной матрице.

## 06. Инструменты ручной разметки [01:39-02:04]

```screen
time: 01:39-02:04
route: /sample/2550382-1-10x
capture_filename: 06_manual_annotation_tools.png
pre_capture:
  - навести курсор или последовательно подсветить toolbar tools
  - оставить Dark pixel preview threshold = 50
ui_state:
  active_tool_sequence:
    - Brush
    - Fill
    - Similar
    - Polygon
    - Rectangle
  segmentation_target: Talc
  dark_pixel_preview_threshold: 50
must_contain:
  - верхний toolbar с Brush, Fill, Similar, Rectangle, Polygon
  - активный класс Talc или видимый переключатель target class
  - viewer с исходным шлифом и тёмными кандидатами
do_not:
  - не сохранять образец
```

Эксперт не рисует всё с нуля. В интерфейсе есть кисть: левая кнопка добавляет выбранный класс, правая сразу стирает. Есть заливка для замкнутых областей, поиск похожих пикселей по цвету и яркости, полигон и прямоугольник для крупных фрагментов. Так мы быстро переводим грубую область-кандидат в пиксельную разметку талька.

## 07. Подтверждённые пиксели талька [02:04-02:23]

```screen
time: 02:04-02:23
route: /sample/2550382-1-10x
capture_filename: 07_talc_segmentation_enabled.png
pre_capture:
  - включить segmentation class Talc
  - для чистоты кадра выключить Positive bag или оставить его с низкой визуальной доминантой
  - Dark pixel preview threshold может оставаться 50 до конца сегмента
ui_state:
  display_layers:
    Background: on
    Original blue lines: off
  segmentation_classes:
    Positive bag: optional_off_or_low_context
    Talc: on
    Not Talc: off
  dark_pixel_preview_threshold: 50
  comparison_mode: Current
must_contain:
  - видимые пиксели класса Talc
  - left Segmentation classes: Talc enabled
  - исходный шлиф как фон
```

После ручной проверки мы получаем уже не область-кандидат, а подтверждённые пиксели талька. Именно этот слой нужен для обучения модели и для расчёта доли талька в анализируемой области.

## 08. Сохранение результата [02:23-02:38]

```screen
time: 02:23-02:38
route: /sample/2550382-1-10x
capture_filename: 08_save_controls_do_not_click.png
pre_capture:
  - удержать текущий результат с включённым Talc
  - показать top-right кнопки Save и Save & Next
ui_state:
  segmentation_classes:
    Talc: on
  dark_pixel_preview_threshold: 50
must_contain:
  - кнопка Save
  - кнопка Save & Next
  - текущая маска Talc на viewer
do_not:
  - не нажимать Save
  - не нажимать Save & Next
```

Когда эксперт закончил правку, кнопка сохранить записывает маски для текущего образца. Кнопка сохранить и перейти дальше делает то же самое и сразу открывает следующий образец. В этом ролике мы кнопки не нажимаем, чтобы не менять состояние демонстрационного набора.

## 09. Переход к режимам сравнения [02:38-02:55]

```screen
time: 02:38-02:55
route: /sample/2550382-1-10x
capture_filename: 09_comparison_mode_selector.png
pre_capture:
  - выключить Dark pixel preview threshold
  - открыть или показать right panel Comparison mode
ui_state:
  dark_pixel_preview_threshold: off
  comparison_mode: Current
  segmentation_classes:
    Talc: on
must_contain:
  - right panel Comparison mode selector
  - варианты Current, Heuristic, Neural Model
  - чистый фон без dark pixel preview
```

Теперь выключаем предпросмотр тёмных пикселей и переходим к режимам сравнения. Здесь можно смотреть текущую ручную маску отдельно или сравнивать её с двумя источниками: объяснимой эвристикой и нейросетевой моделью.

## 10. Эвристическая сегментация [02:55-03:25]

```screen
time: 02:55-03:25
route: /sample/2550382-1-10x
capture_filename: 10_heuristic_segmentation_after_run.png
pre_capture:
  - выбрать Comparison mode = Heuristic
  - нажать Run non-neural classifier / Run classifier
  - дождаться завершения, при долгом ожидании ускорить этот фрагмент монтажа
  - если результат уже есть в qa/non_neural_talcose, всё равно показать действие запуска или состояние после запуска
ui_state:
  dark_pixel_preview_threshold: off
  comparison_mode: Heuristic
  display_layers:
    Background: on
  segmentation_classes:
    Positive bag: off
    Talc: off_or_ignored_by_plain_heuristic_mode
must_contain:
  - слой Heuristic поверх изображения
  - fuchsia heuristic-only talc zone overlay, если используется текущая палитра
  - статус или статистика эвристики в right panel
  - отсутствие dark pixel preview
```

Сначала запускаем эвристический классификатор. Он повторяет понятную логику со слайда: исключаем рудную фазу, ищем тёмные хлопья в нерудной матрице, отбрасываем сплошные тёмные пятна и собираем близкие хлопья в зоны талька. Это не финальная истина, а прозрачная проверочная подсказка.

## 11. Нейросетевая сегментация [03:25-03:58]

```screen
time: 03:25-03:58
route: /sample/2550382-1-10x
capture_filename: 11_neural_model_segmentation_after_run.png
pre_capture:
  - выбрать Comparison mode = Neural Model
  - проверить ML talc probability threshold = 0.50, если поле видно
  - нажать Run model
  - дождаться появления neural/model talc overlay, при долгом ожидании ускорить wait в монтаже
  - если model_talc_mask.png уже есть, всё равно показать состояние после запуска модели
ui_state:
  dark_pixel_preview_threshold: off
  comparison_mode: Neural Model
  neural_model:
    model: SegFormer-B0 talc segmentation
    threshold: 0.50
  display_layers:
    Background: on
must_contain:
  - слой Neural Model поверх изображения
  - видимый результат ML-сегментации талька
  - кнопка Run model или статус завершённого запуска
```

Теперь запускаем нейросетевую модель талька. Это Сегформер Би-ноль: модель смотрит на пиксели в нерудной области и выдаёт свою маску талька. В отличие от синей линии, это уже плотная ML-сегментация, которую можно сравнивать с ручной правкой эксперта.

## 12. Финальный кадр для монтажа [03:58-04:12]

```screen
time: 03:58-04:12
route: /sample/2550382-1-10x
capture_filename: 12_slide_sync_three_sources.png
pre_capture:
  - в монтаже показать быструю последовательность из трёх кадров:
    1. 03_positive_bag_from_blue_lines.png
    2. 10_heuristic_segmentation_after_run.png
    3. 11_neural_model_segmentation_after_run.png
ui_state:
  montage_sources:
    approximate_annotation: Positive bag from original blue lines
    heuristic: Heuristic talc-zone segmentation
    ml: Neural Model talc segmentation
must_contain:
  - три источника сигнала в порядке слайда
  - одинаковый образец 2550382-1 10x
  - corner time mark
```

В итоге на одном и том же образце видны три уровня сигнала: грубая зона от синих линий, объяснимая эвристика и нейросетевая сегментация. Так интерфейс показывает, откуда берётся обучающая разметка талька и почему эксперт может быстро проверить результат перед использованием в модели.

## Контрольный список после записи

- В начале видно только исходное изображение: `Background` включён, все классы выключены.
- `Original blue lines` показан отдельно от классов.
- `Positive bag` объяснён как зона вероятного талька, а не как пиксельная маска.
- В тексте есть ссылка по смыслу на Q&A #4: численной доли талька для этих изображений нет, методологию задаёт команда.
- `Dark pixel preview threshold` установлен в `50`, затем выключен перед comparison modes.
- Показаны инструменты Brush, Fill, Similar, Polygon, Rectangle; для Brush озвучено, что правая кнопка стирает.
- `Talc` class показан как подтверждённые пиксели.
- `Save` и `Save & Next` объяснены, но не нажаты.
- Для эвристики перед скриншотом выполнен запуск классификатора или явно показан результат после запуска.
- Для нейросети перед скриншотом выполнен `Run model` или явно показан результат после запуска.
- Финальная последовательность совпадает со слайдом: приблизительная разметка организаторов, эвристическая сегментация, ML-сегментация.
