# Mindstorm: как усилить официальный OM-классификатор

Дата: 2026-07-03

Контекст: задача организаторов сформулирована как исследовательская; им интересны любые подходы, включая дополнительную разметку изображений. В этом документе идеи намеренно привязаны к узкому v2-scope: optical microscopy only, сульфиды, обычные/тонкие срастания, тальк, доли, официальный класс руды и проверяемый отчет.

## Короткий вывод

Самый сильный путь для заявки - не еще один большой UI, а демонстрация исследовательского цикла:

```text
официальное изображение
-> несколько слабых источников маски
-> карта согласия/сомнения
-> быстрая точечная доразметка сложных зон
-> улучшенная модель
-> интерпретируемые компоненты
-> класс руды + доказательства
```

То есть "killer feature" должен звучать так: система не только выдает ответ, но и показывает, где она уверена, где сомневается, какую минимальную разметку стоит добавить и как это меняет метрики.

## Killer features для финальной истории

### 1. Карта сомнений вместо просто confidence heatmap

Обычная confidence heatmap полезна, но слабая как история. Лучше сделать "карта спора источников":

- Petroscope/teacher считает пиксель сульфидом;
- ResUNet/SegFormer считает пиксель сульфидом;
- яркостно-морфологический baseline считает пиксель сульфидом;
- TTA/ensemble дает нестабильность;
- зона попадает в артефакт/блик/границу тайла.

На выходе:

- зеленое: все согласны;
- желтое: 2 из 3 источников согласны;
- красное: конфликт или нестабильность;
- серое: exclude/ignore.

Для жюри это сильнее, чем просто "модель с IoU": видно, что мы управляем слабой разметкой честно и не выдаем псевдомаску за ground truth.

### 2. Active learning queue: "покажи 20 самых полезных фрагментов"

Если дополнительно размечать изображения, не размечать случайные тайлы. Делать очередь:

1. выбрать тайлы/суперпиксели с максимальной площадью disagreement;
2. поднять приоритет зон, которые меняют итоговый класс: около 10% талька, около 50/50 ordinary/fine, крупные сульфидные компоненты;
3. показывать пользователю маленький crop с overlay;
4. сохранить JSON patch: `sulfide`, `not_sulfide`, `talc`, `exclude_artifact`, `uncertain`;
5. пересобрать training manifest и показать, как изменилась метрика или стабильность решения.

Минимальный demo-сценарий: "после 20 правок спорных зон доля ignore снизилась с X% до Y%, итоговый класс остался/изменился, зона решения стала стабильнее".

### 3. SAM2-assisted разметка как ускоритель, не как финальный классификатор

SAM/SAM2 не знает геологии, но хорошо подходит для интерактивного выделения границ. Его правильная роль:

- пользователь кликает точку/box на спорной зоне;
- SAM2 предлагает контур;
- наш код присваивает геологическую метку только после выбора человеком или после согласования с teacher/baseline;
- получившаяся маска идет в patch manifest.

Это можно показать как "быстрая попиксельная доразметка без CVAT", особенно для талка и границ сульфидных включений.

### 4. Паспорт каждого сульфидного включения

Для ordinary/fine срастаний сильная история не в end-to-end пиксельной магии, а в компонентной геологии. Для каждой компоненты сохранить:

- `component_id`;
- площадь;
- bbox;
- solidity / compactness;
- perimeter / area;
- skeleton width или proxy толщины;
- доля темной нерудной фазы внутри восстановленного footprint;
- локальная плотность фрагментации;
- predicted class: ordinary/fine;
- confidence / reason.

В отчете можно кликнуть/показать топ-5 компонентов, которые сильнее всего влияют на класс. Это дает объяснимость: "труднообогатимая, потому что 62% площади сульфидов относится к фрагментированным/замещенным компонентам".

### 5. Decision margin: не только класс, но и запас до границы

Официальное правило дискретное, но реальная полезность выше, если показывать запас:

- тальк: `talc_fraction - 10%`;
- ordinary/fine: `ordinary_fraction - fine_fraction`;
- зона риска: `abs(talc - 10%) < 2%` или `abs(ordinary - fine) < 5%`.

Если образец близок к порогу, система должна честно писать "на экспертную проверку". Для hackathon это выглядит зрелее, чем принудительная уверенность.

### 6. Robustness certificate для каждого запуска

К каждому результату добавить маленький блок:

- прогон при brightness/contrast jitter;
- прогон при JPEG quality/compression;
- прогон при другом tile offset;
- TTA flips/rotations, если геологически допустимы;
- drift доли сульфидов, талька и ordinary/fine;
- изменился ли итоговый класс.

Формулировка в отчете: "класс стабилен при стандартных возмущениях; максимальный дрейф доли талька 0.8 п.п." Это напрямую отвечает на требование устойчивости к вариативности микроскопии.

### 7. Ablation table как доказательство исследовательской работы

В презентацию или appendix добавить таблицу:

| Вариант | Сульфид proxy IoU | Image-level F1 | Talc error | Время | Комментарий |
| --- | ---: | ---: | ---: | ---: | --- |
| brightness baseline | | | | | объяснимый минимум |
| ResUNet | | | | | быстрый baseline |
| SegFormer-B0 | | | | | лучший текущий кандидат |
| ensemble/TTA | | | | | лучше confidence, дороже |
| + QA patches | | | | | эффект ручной доразметки |

Даже если часть ячеек пустая до экспериментов, структура показывает, что решение измеримое, а не "на глаз".

### 8. Research runner: один reproducible пакет для жюри

Сделать команду или script, который собирает evidence-папку:

```text
sample_id/
  source_preview.jpg
  overlay.png
  disagreement_map.png
  confidence_heatmap.png
  component_passport.csv
  metrics.csv
  ore_classification.json
  report.pdf
  run_summary.json
  model_card.md
```

Это не новая модель, но очень сильная упаковка. Жюри видит не только результат, но и воспроизводимость, источники, параметры и ограничения.

## Идеи по обучению и разметке

### A. Улучшить weak-label fusion

Текущий план "2 из 3 источников согласны = hard pseudo-label, конфликт = ignore" хороший для P0. Усиление:

- хранить не только hard mask, но и probability/confidence map;
- источникам дать веса по валидации: teacher, student, heuristic;
- для каждого источника считать sensitivity/specificity на небольшой проверенной подвыборке;
- попробовать STAPLE-подобную EM-фузию масок, если будет 3+ источника и хотя бы маленькая ручная проверка;
- disagreement не выбрасывать полностью: использовать его как active-learning приоритет.

Практический минимум: добавить `pseudo_confidence.png` и `source_votes.png` в outputs.

### B. Semi-supervised training поверх официальных изображений

После первого ResUNet/SegFormer:

1. обучить две разные модели или две инициализации;
2. на official unlabeled/weak-labeled тайлах получить pseudo labels;
3. использовать только пиксели высокой уверенности;
4. заставлять модели согласовываться на одинаковых изображениях под разными аугментациями.

Это CPS/teacher-student идея: полезно, потому что официальных изображений много, а попиксельной ground truth мало.

### C. SAM-enhanced WSSS для image-level папок

Image-level метки "рядовые", "тонкие/труднообогатимые", "оталькованные" можно использовать аккуратнее:

- обучить image classifier/patch classifier;
- получить CAM/attention heatmap по классам;
- SAM/SAM2 использовать для уточнения границ объектов, на которые указывает CAM;
- пересечь это с сульфидной маской и компонентами;
- не делать из этого прямую truth-mask; использовать как soft signal / candidate regions.

Особенно полезно для поиска участков, которые отличают ordinary vs fine, но финальная логика все равно должна идти через компоненты и морфологию.

### D. Multiple-instance learning для ordinary/fine

Вместо прямого `image -> class`:

```text
image = bag
sulfide components = instances
image-level label = row/fine/hard-to-process
model learns which components explain the bag label
```

Это естественно для задачи: официальная метка дана на изображение, а решение зависит от множества включений. MIL можно начать без сложной архитектуры:

- признаки компоненты из passport;
- LightGBM/logistic regression/MLP;
- attention pooling по компонентам;
- валидация group split по образцам.

Killer angle: "модель не просто классифицирует картинку, она показывает компоненты, которые повлияли на решение".

### E. Loss functions для малых и тонких классов

Для сульфидов и талка есть сильный class imbalance. Стоит проверить:

- BCE + Dice как baseline;
- Focal loss для hard examples и редких позитивных пикселей;
- Generalized Dice для дисбаланса;
- Boundary loss или Lovasz/IoU loss для лучшей границы;
- hard negative mining для темных не-тальковых областей.

Для талка важно: темная матрица, грязь, тени и артефакты должны попадать в hard negatives, иначе модель начнет считать "все темное = тальк".

### F. Domain adaptation и color robustness

Микроскопия сильно меняется от оборудования, освещения и подготовки образца. Поэтому:

- train augmentations должны имитировать brightness/color shifts, CLAHE variants, blur/sharpness, JPEG;
- validation лучше держать по sample/folder split, а не случайно по тайлам;
- `ч1` и `ч2` можно использовать как разные съемочные партии для holdout;
- внешние LumenStone/Cu/FeM использовать как pretraining/domain stress test, но не как официальный proof.

### G. Foundation-model features без полного переезда на foundation model

Рискованно заменять P0 на тяжелый foundation pipeline. Но можно точечно использовать:

- DINOv2/самопредобученные признаки для patch/component embeddings;
- SAM2 encoder/SAM2 prompts для boundary refinement;
- RoImAI/Petro-SAM как исследовательские ссылки: domain gap есть, но направление "boundary + semantics" правильное;
- Mask2Former только как P1/эксперимент после стабильного binary path.

## Deep search addendum: что добавилось после более глубокого поиска

### 1. OIA literature: продавать не "нейросеть", а цифровой ore-petrography protocol

В optical image analysis (OIA) для polished sections важна не только модель, но и воспроизводимый протокол измерений: стабильная настройка микроскопа/света, логируемые параметры, измерение grain size/liberation/association и контроль неоднозначных фаз. Для нашего проекта это означает:

- в `run_summary.json` нужно логировать не только модель, но и preprocessing, tile size/overlap, scale, threshold presets, ignored-area policy;
- в отчете стоит явно писать, что доли считаются по анализируемой площади после исключения артефактов;
- компонентный паспорт должен содержать не только ordinary/fine label, но и признаки, похожие на geometallurgical/OIA metrics: area, equivalent diameter, Feret proxy, perimeter, association with matrix/talc, liberation/replacement ratio;
- если источник сигнала плохо различает нерудную фазу и фон, сначала сегментировать opaque/bright sulfides, а уже затем искать non-opaque/talc в оставшейся матрице.

Практический вывод: это поддерживает текущую архитектуру `sulfides first -> components -> talc in matrix`, и дает язык для жюри: "мы автоматизировали классический quantitative ore petrography workflow, а не просто применили segmentation model".

### 2. Enhanced thresholding для талька и hard negatives

В работах по reflected-light OIA регулярно встречается проблема: non-opaque minerals могут иметь близкую отражательную способность к resin/epoxy/темной матрице. Поэтому простое правило "темное = тальк" опасно. Лучшее применение к нашему кейсу:

- для talc detector сначала ограничить область поиска: `analyzed_area - sulfide_mask - artifact_mask`;
- положительные talc-кандидаты брать только из converted blue-line annotations после QA;
- hard negatives брать из темной матрицы, теней, загрязнений, царапин, dark holes inside sulfide footprint, областей без LumenStone/Petroscope matrix/silicate support;
- добавить `talc_candidate_reason` / `talc_reject_reason` в debug JSON;
- в презентации сказать: "мы специально не считаем все темные пиксели тальком; у нас есть hard-negative контур".

### 3. Snorkel/STAPLE-like label model вместо голого majority vote

Сейчас majority/2-of-3 fusion прост и пригоден для P0. Более исследовательская версия:

```text
label functions:
  LF_heuristic_bright_sulfide
  LF_petroscope_sulfide
  LF_segformer_b0
  LF_resunet
  LF_sam_boundary_accept
  LF_artifact_detector
  LF_talc_blue_line_positive
  LF_dark_matrix_hard_negative

label model:
  estimates source reliability/correlation
  emits soft label probability
  emits abstain/ignore for contradictions
```

Минимальная реализация без Snorkel-зависимости:

- хранить `source_votes.tif/png` с битовыми каналами источников;
- вручную задать веса источников в YAML;
- считать `pseudo_confidence = weighted_agreement / active_sources`;
- `ignore_mask = confidence < threshold OR conflict_with_artifact`;
- активное ревью ранжировать по площади low-confidence и влиянию на итоговый класс.

Если будет хотя бы небольшая ручная проверка, можно оценить per-source precision/recall и заменить фиксированные веса на эмпирические.

### 4. Noisy-label training: co-teaching для pseudo-mask тайлов

Если обучать SegFormer/ResUNet на псевдомасках, модель может заучить ошибки teacher-а. Co-teaching-подход дает практичную идею:

- обучить две модели с разной архитектурой или инициализацией;
- на каждом batch выбирать low-loss пиксели/тайлы как более вероятно чистые;
- модель A выбирает "чистые" примеры для модели B и наоборот;
- спорные пиксели переводить в ignore, а не заставлять обе модели согласиться с ошибкой.

Для P0 это, вероятно, слишком долго. Но можно сделать lightweight version: после эпохи 1-3 исключить топ-N% тайлов с максимальным training loss и проверить, улучшается ли validation IoU/visual stability.

### 5. MIL/CLAM-like ordinary-vs-fine поверх компонент, а не поверх пикселей

Whole-slide pathology очень похожа по структуре данных: большое изображение, есть только slide/image-level label, локальной разметки мало или нет. В этой литературе сильный паттерн:

```text
large image = bag
patches/components = instances
bag label = class
attention pooling finds important instances
```

Для нас:

```text
official photo/panorama = bag
sulfide components or component-neighborhood crops = instances
image-level folder label = row/fine/hard-to-process
attention/weights = какие включения объясняют итог
```

Минимальный вариант без тяжелого DL:

- собрать `component_passport.csv` по всем official class-folder images;
- обучить logistic regression / LightGBM на агрегатах: median/percentile replacement ratio, area-weighted fine score, fraction of fragmented components;
- затем попробовать attention-MIL поверх компонентных embeddings;
- в отчете показывать top contributing components как объяснение ordinary/fine.

Это может быть одной из самых сильных research-фич, потому что напрямую использует official image-level labels и не притворяется, что есть pixel-level ordinary/fine truth.

### 6. Domain adaptation: полезны простые style/illumination tricks, не обязательно adversarial training

По microscopy/domain-adaptation papers видно, что domain shift от оборудования, цвета, яркости и подготовки образца может сильно ломать модель. Для нашего срока лучше:

- добавить FDA-like augmentation: перенос low-frequency color/style statistics между official images и proxy datasets;
- сделать stain/color jitter не случайным "как в natural images", а похожим на реальные сдвиги микроскопии: blue/pink/yellow shifts, CLAHE variants, gamma, vignetting;
- валидировать по folder/sample split, а не random tile split;
- добавить test-time normalization option, но не делать TENT/adversarial adaptation default без отдельного safety check;
- в model card явно разделить `trained_on_proxy`, `trained_on_official_pseudo`, `validated_on_official_holdout`.

ADVENT/DANN/TENT полезны как research references, но для P0 слишком рискованны: сложнее объяснить, сложнее воспроизвести, выше шанс нестабильности на одной панораме.

### 7. Large-image inference: фиксировать tile math, а не просто "режем на тайлы"

Для панорам 20k+ px важно отдельно контролировать ошибки на границах тайлов:

- overlap должен быть привязан к receptive field / effective receptive field модели;
- logits/probabilities лучше сшивать с window weighting, например Hann/Gaussian;
- хранить `tile_offset`, `overlap`, `blend_window`, `batch_size`, `device`, runtime/memory;
- robustness certificate должен включать прогон с другим tile offset: если итоговый класс меняется от сдвига сетки тайлов, это blocker;
- для connected components нужен global postprocess после сшивки, иначе зерна на границах тайлов будут раздроблены.

Минимальный artifact: `tiling_summary.json` + `tile_boundary_uncertainty.png`.

### 8. Losses and metrics: разделить "площадь" и "границу"

Для sulfide mask главная метрика - площадь/компоненты; для талка и thin structures важны границы и малые области. Проверить стоит не одну loss:

- sulfide binary: BCE + Dice / Lovasz hinge как baseline для area IoU;
- talc: Focal Tversky или BCE+Tversky, чтобы управлять FP/FN на редком классе;
- boundary/HD95: не обязательно обучать boundary loss, но точно считать boundary metric на talc blue-line-derived validation;
- report metric: не только pixel IoU, но и error in fractions: `abs(pred_talc_fraction - label_talc_fraction)` и drift of final ore class.

Для жюри доля талька и итоговый класс важнее красивого mean IoU.

### 9. Self-supervised/foundation direction: DINO/SAM только после domain adaptation

Более свежие microscopy papers предупреждают: natural-image foundation features могут плохо совпадать с microscopy domain. Правильная формулировка:

- SAM2 - ассистент разметки и boundary refinement;
- DINOv2/SSL - источник embeddings для component/patch retrieval и MIL, но лучше с continued self-supervised pretraining на official unlabeled crops;
- foundation path не должен заменять интерпретируемый P0, пока нет устойчивых метрик;
- если делать experiment, то "DINO embeddings + component features -> MIL ordinary/fine", а не "foundation model decides ore class".

### 10. Talc search result: drop-in публичного talc OM segmentation dataset не нашел

Глубокий поиск не дал готового открытого датасета polished-section OM segmentation именно для талька в нашем смысле. Есть microscopy/mineral identification and talc analysis literature, но не готовая pixel-level модель для "тальк как рассеянная темная фаза в матрице" на официальных изображениях.

Практический вывод остается прежним:

- official `Области оталькования` - главный supervision source;
- blue-line conversion + QA - обязательный путь;
- внешние sources годятся только для hard negatives, domain robustness и аргументации, но не для прямой claim "мы обучили talc по public ground truth".

## Reddit/practitioner addendum: идеи из практических тредов

Это не научные источники и не основание для заявлений о качестве. Но Reddit полезен как источник боли пользователей: где SAM ломается, как люди ускоряют разметку, какие workflow выглядят убедительно для практиков.

### 1. Не продавать "SAM wrapper"; продавать research loop

В нескольких тредах люди относятся скептически к продуктам, которые выглядят как "SAM с UI". Практический вывод для нас: killer feature должен быть не "мы подключили SAM2", а замкнутый цикл:

1. модель/эвристика/петроскоп дают черновую маску;
2. система показывает зоны конфликта и пограничные компоненты;
3. человек правит только top-N важных crops;
4. правки сохраняются как patches/provenance;
5. следующий запуск показывает, где качество выросло и где все еще спорно.

Формулировка для слайда: "SAM2 используется как интерактивный ассистент разметки, а ценность решения - в выборе правильных областей для проверки, трассировке правок и измерении устойчивости".

### 2. SAM в микроскопии может давать красивые, но неправильные границы

В materials microscopy thread пользователь пишет, что SAM automatic mask generator дает артефакты по границам зерен и oversegmentation, из-за чего результат почти непригоден для последующего измерения. Это близко к нашей задаче: нам важен не только визуально красивый overlay, но и проценты, площади, ordinary/fine граница и talc threshold.

Практический вывод:

- SAM2 нельзя использовать как единственный источник измеряемой маски;
- SAM2 надо запускать в режиме proposal/correction с явным `source=sam2_assist`;
- в отчет добавить `sam2_modified_fraction` и `manual_reviewed_fraction`;
- для демонстрации показывать кейс, где SAM помогает границу поправить, но финальное решение объясняется компонентами и метриками.

### 3. Flat-field / illumination correction как недорогой quality boost

В microscopy image-processing thread советуют сначала исправить градиент освещения: empty-field/flat-field correction или adaptive vignette correction, а уже затем thresholding/contours. Для наших официальных фото это может дать простой прирост стабильности без новой модели.

Минимальная задача:

1. добавить `illumination_profile.png` для каждого изображения;
2. сохранять `image_flatfield.png` / `image_illum_norm.png`;
3. запускать heuristic sulfide/talc candidates на нормализованном изображении;
4. в robustness certificate показать, что fraction меньше прыгает при изменении яркости/контраста.

Это особенно полезно для талька: не все темные зоны должны становиться talc-positive, но сначала надо убрать глобальную неравномерность света.

### 4. Edge/boundary-first подход для зерен и включений

В materials microscopy discussion прозвучала мысль: если вся картинка состоит из зерен, иногда правильнее искать границы, а не класс "grain vs not grain". Для нас это переносится так:

- для ordinary/fine важны границы и размеры сульфидных компонент;
- можно добавить auxiliary boundary channel: `sulfide_boundary_probability`;
- component passport должен хранить не только area/perimeter, но и boundary confidence;
- спорные touching components лучше отдавать в review queue, потому что они влияют на размерный класс.

Быстрый вариант без новой архитектуры: извлекать boundary band из текущей маски, считать border uncertainty и ранжировать компоненты, у которых boundary uncertainty сильно меняет площадь/эквивалентный диаметр.

### 5. Model-assisted labeling: маленький gold set + predict + review

В semi-automatic labeling тредах повторяется практический паттерн: сначала размечается маленький набор, затем обучается свой модельный черновик, потом он предсказывает большую коллекцию, а человек уже исправляет. Это лучше, чем слепо полагаться на общий VLM/SAM или размечать все вручную.

Для нашего проекта это можно сделать как артефакт:

- `gold_set_manifest.csv`: 20-50 вручную проверенных crops/images;
- `pseudo_label_manifest.csv`: что создано эвристикой/моделью;
- `review_patch_manifest.csv`: какие маски человек исправил;
- `retrain_delta.md`: как поменялись IoU/HD95/F1/AUC после добавления исправлений.

Даже если реального эксперта нет, можно честно назвать это `non_expert_gold_candidate` и не выдавать за геологическую истину.

### 6. Active learning: не только entropy, добавить diversity/core-set

В тредах про active learning есть важное предупреждение: простая entropy sampling часто не является сильной killer feature, особенно на малых данных. Для нас лучше сделать ranking как смесь:

`review_score = uncertainty * decision_impact * novelty`

Где:

- `uncertainty` - disagreement/entropy/TTA variance;
- `decision_impact` - насколько crop влияет на talc threshold 10% или ordinary/fine 50/50;
- `novelty` - расстояние в embedding/features от уже проверенных crops.

Практический результат: review queue должна показывать не просто "самые сомнительные пиксели", а "самые полезные 20 областей, которые изменят решение или расширят покрытие данных".

### 7. High-loss examples как дешёвый детектор плохих pseudo-labels

В weak-label segmentation thread советуют прогнать модель, посмотреть примеры с самым большим loss и либо исключить, либо переразметить worst offenders. Это хорошо ложится на наш weak-supervision sulfide dataset.

Минимальная реализация:

1. во время train/val сохранять `tile_loss.csv`;
2. экспортировать top-50 high-loss tiles;
3. помечать их как `needs_review`, `exclude_from_train`, `label_boundary_ambiguous` или `artifact`;
4. делать retrain без worst 1-3% и сравнивать метрики.

Это может дать быстрый прирост качества и выглядит исследовательски: мы не просто обучаемся на шумных pseudo-labels, а измеряем и чистим шум.

### 8. Edge-weighted loss для неоднозначных границ

В том же weak-label discussion отмечают, что неоднозначные границы портят обучение: разные аннотаторы дают более узкие/широкие маски. Идея для нас:

- не штрафовать одинаково центр уверенной сульфидной области и размытую границу;
- сделать `boundary_weight_map.png`;
- на good examples можно усилить loss около границ;
- на noisy pseudo-labels можно наоборот сделать ignore band вокруг спорной границы.

Это дает хороший эксперимент:

1. baseline BCE/Dice;
2. BCE/Dice + boundary weights;
3. BCE/Dice + ignore uncertain boundary band;
4. сравнение IoU/HD95/fraction error.

### 9. Annotation UX: локальный, быстрый, простой brush/magic-wand важнее тяжелого web-комбайна

В тредах про annotation tools часто всплывают CVAT, Label Studio, Roboflow, AnyLabeling, SAMAT, локальный brush/magic-wand workflow и нарезка больших проектов на small tasks. Для v2 это уже поддерживает направление `apps/talc_review_streamlit.py`, но можно усилить:

- режим "one crop, one decision": accept / edit / unsure / exclude;
- горячие клавиши для accept/next/uncertain;
- отдельная очередь `decision-critical`, а не просто список файлов;
- progress metrics: checked area %, changed area %, remaining decision risk;
- offline-first: все правки в JSON/PNG рядом с данными, без внешнего SaaS.

### 10. Мульти-масштабные изображения: stitching и tile provenance должны быть видны

В thread про dense instance segmentation обсуждают sliding windows, SAHI, remapping coordinates и stitching крупных объектов. У нас уже есть high-resolution/tiled inference; Reddit добавляет UX-вывод: надо показывать не только итоговую маску, но и как она собрана.

Идеи:

- `tile_grid_overlay.png` с зонами overlap;
- `tile_edge_artifact_score`;
- `component_crosses_tile_boundary`;
- повторить inference с offset half-tile и сравнить component/fraction delta;
- если компонент пересекает tile boundary, его ordinary/fine решение помечать как менее уверенное.

### 11. Mineral-property checklist как explainability card

В r/geology треде про сайт для идентификации минералов под микроскопом ценность была не в автоматической классификации, а в structured observations: цвет, рельеф/отражение, свойства, cross-reference с таблицами. Это полезная подсказка для нашего component passport.

Вместо "модель сказала, что это компонент N" можно сделать карточку:

- brightness/reflectance proxy;
- hue/saturation proxy;
- форма: компактность, elongation, convexity;
- соседства: sulfide near talc/artifact/matrix;
- размер: equivalent diameter, area percentile;
- вклад в решение ordinary/fine;
- warning: "визуальные признаки слабые/неполные, нужна проверка".

Это будет ближе к мышлению геолога, даже если признаки являются прокси.

### 12. Team annotation strategy: small tasks + multi-annotator disagreement

Для большого набора изображений практики рекомендуют дробить проект на небольшие задачи и распределять по людям. Если будет время на дополнительную разметку, лучше не просить "разметить все", а подготовить микрозадачи:

- 20 crops for talc threshold;
- 20 crops for sulfide boundary;
- 20 crops for touching/merged components;
- 20 crops for artifact/hard negative.

Если есть 2-3 разметчика, disagreement между ними сам становится сигналом uncertainty. Это можно сохранить как `annotator_votes.csv` и использовать аналогично source-disagreement map.

## More idea search addendum: что добавилось после поиска по tooling/workflows

Этот блок собран не только из papers, но и из официальных документаций/репозиториев инструментов вокруг microscopy annotation, dataset curation и mineral liberation analysis. Практический вывод: сильная заявка может выглядеть не как "еще одна модель сегментации", а как маленькая reproducible laboratory workflow system вокруг официальных OM-изображений.

### 1. Shallow interactive classifier как быстрый экспертный инструмент

Ilastik, Labkit, Trainable Weka Segmentation и QuPath сходятся в одном паттерне: пользователь рисует немного scribbles/annotations, система быстро обучает pixel classifier на цвете, текстуре, масштабе и локальных признаках, а затем показывает live prediction. Это не конкурирует с SegFormer, а дает полезный baseline и быстрый режим доразметки.

Идея для проекта:

- сделать `scripts/train_scribble_pixel_classifier.py` или режим в `sulfide_qa_streamlit.py`;
- вход: несколько foreground/background scribbles или corrected masks;
- признаки: RGB/HSV/Lab, grayscale, local mean/std, gradient, Hessian/LoG, multi-scale texture;
- модель: RandomForest/ExtraTrees/HistGradientBoosting;
- выход: `scribble_rf_probability.png`, `scribble_rf_mask.png`, `scribble_rf_features.json`;
- использовать как еще один independent source в `source_votes.png`.

Почему это killer feature: если нейросеть ошиблась на новом микроскопе/освещении, эксперт за 3-5 минут добавляет несколько штрихов и получает локально адаптированный классификатор без GPU.

### 2. Corrective annotation вместо полной разметки

RootPainter, MONAI Label и AIDE полезны как workflow-шаблон: человек не рисует маску с нуля, а исправляет ошибки текущей модели; модель дообучается на этих исправлениях; следующая порция становится быстрее. Для нас это можно реализовать легче, без полноценного server-client продукта.

Минимальная версия:

1. `review_queue/` содержит crops, ранжированные по `uncertainty * decision_impact * novelty`;
2. пользователь ставит `accept`, `fix`, `ignore`, `uncertain`;
3. исправления сохраняются как `patch_mask.png` + `patch_meta.json`;
4. `scripts/retrain_from_review_patches.py` добавляет patches в dataset;
5. `retrain_delta.md` показывает, что изменилось: IoU/HD95/F1/AUC, fraction error, число changed decisions.

Дополнительный отчет: `annotation_efficiency.json`:

- `reviewed_crops`;
- `changed_pixels`;
- `changed_components`;
- `decisions_changed`;
- `minutes_per_decision_risk_reduction`, если время размечено вручную.

### 3. Dataset curation dashboard: uniqueness, hardness, label issues

FiftyOne Brain и cleanlab подсказывают сильную data-centric линию. Для хакатона это может быть очень убедительно: показать, что мы не только обучили модель, но и нашли слабые места датасета и pseudo-labels.

Артефакт:

- `dataset_curation_report.md`;
- `near_duplicate_groups.csv`;
- `leaky_split_candidates.csv`;
- `hard_tiles.csv`;
- `label_issue_tiles.csv`;
- `embedding_clusters.png`;
- `representative_crops/`;
- `outlier_crops/`.

Практический scoring impact:

- не допускать leakage: тайлы одного образца не должны быть и в train, и в val;
- выбирать для annotation не похожие crops, а покрывающие разные кластеры;
- чистить pseudo-labels через `pred_probs + labels`, как в cleanlab segmentation;
- показывать "мы знаем, где модель не знает".

### 4. Microscopy-domain pretraining вместо ImageNet-only

MicroNet/NASA pretrained microscopy encoders и self-supervised EM literature подтверждают важную идею: для микроскопии in-domain pretraining может быть полезнее, чем обычный ImageNet, особенно при малом числе масок и доменном сдвиге.

Варианты:

- P0-lite: прогнать embeddings не из DINO/ImageNet, а из microscopy-pretrained encoder, если быстро ставится;
- P1: заменить encoder в U-Net/DeepLab-like модели на MicroNet-pretrained ResNet;
- P1: self-supervised continued pretraining на official unlabeled tiles: masked image modeling / DINO-style / SimCLR/MoCo;
- отчет: `pretraining_ablation.csv` с `imagenet`, `microscopy_pretrained`, `official_ssl`.

Важная оговорка: это не гарантированно улучшит SegFormer-B0 сейчас. Но как research direction это сильнее, чем "скачаем большой foundation model".

### 5. MicroSAM/MatSAM-style prompt engineering

μSAM показывает, что для microscopy default SAM лучше адаптировать или fine-tune; MatSAM показывает другую линию - не обязательно дообучать SAM, можно улучшать prompt generation под dense/dispersed microstructures.

Идея для нас:

- генерировать SAM prompts не равномерной сеткой по всему изображению, а по `analyzed_mask`;
- positive prompts ставить в high-confidence sulfide/talc candidate cores;
- negative prompts ставить на blue-line artifacts, glare, dark resin/matrix hard negatives;
- объединять prompt-grid + ROI prompts;
- после SAM делать soft-NMS/merge по компонентам;
- сохранять `sam_prompt_plan.json` и `sam_prompt_overlay.png`.

Это превращает SAM из "волшебной кнопки" в контролируемый engineering step.

### 6. Pixel classifier -> object classifier, как в QuPath

QuPath явно разделяет pixel classification и object/cell classification. Для руд это очень хороший architectural pattern:

1. pixel model находит sulfide/non-sulfide/talc/artifact;
2. connected components превращают пиксели в объекты;
3. object classifier решает `ordinary_component`, `fine_component`, `artifact`, `needs_review`;
4. ore-class rule агрегирует object-level факты.

Новая фича: `component_classifier_report.md`.

Поля:

- признаки компонента;
- classifier votes;
- threshold margin;
- nearest reviewed components;
- причины `needs_review`.

Это хорошо объясняет, почему мы не делаем один непрозрачный `image -> class`.

### 7. Mineral liberation / association report как промышленный язык

MLA/QEMSCAN/automated mineralogy материалы повторяют один набор outputs: abundances, associations, sizes, shapes, liberation. Даже если мы остаемся в official OM-only scope, можно говорить на этом языке.

Добавить в отчет:

- `sulfide_abundance_area_pct`;
- `talc_area_pct`;
- `component_size_distribution`;
- `fine_fraction_by_area`;
- `ordinary_fraction_by_area`;
- `locked_or_touching_components`;
- `sulfide_talc_contact_length`;
- `association_matrix.csv`: sulfide/talc/matrix/artifact adjacency;
- `liberation_proxy.csv`: component окружен matrix vs контактирует с talc/other components.

Это не должно называться полноценным MLA, но можно назвать "OM-derived liberation/association proxy".

### 8. Model/data cards как доверительный пакет

Model Cards, Datasheets for Datasets и AI FactSheets дают готовую структуру для "мы понимаем ограничения". Для хакатона это может выглядеть очень профессионально.

Артефакты:

- `model_card.md`;
- `dataset_card.md`;
- `weak_supervision_card.md`;
- `run_fact_sheet.md`.

Содержание:

- что является official label, pseudo-label, manual correction, non-expert QA;
- какие классы покрыты;
- какие метрики есть и чего нет;
- где нельзя использовать модель;
- какой checkpoint, commit, параметры tiling, augmentations;
- список known failure modes.

### 9. "Ask the expert" generator

Если нет времени на полноценную ручную разметку, можно сделать функцию, которая превращает uncertainty в конкретные вопросы эксперту:

- "Эта темная область - тальк или фон/смола?";
- "Эти две сульфидные зоны считать одной компонентой или двумя?";
- "Граница сульфидной фазы проходит по синей линии или внутри нее?";
- "Этот компонент влияет на ordinary/fine threshold, подтвердить размер?".

Артефакт:

- `expert_questions.md`;
- каждая строка: crop, overlay, current decision, why it matters, one binary/multiple-choice question.

Это лучше, чем общая просьба "разметьте еще данных".

### 10. Annotation budget simulator

Materials active-learning papers часто показывают эффект "достаточно 4-10% данных". Мы можем сделать маленький симулятор без новых экспериментов:

- взять существующий validation/proxy split;
- отсортировать tiles разными стратегиями: random, entropy, disagreement, diversity, decision-impact;
- для каждого бюджета `k=10/25/50/100` посчитать покрытие классов, coverage clusters, expected changed area;
- если есть retrain - добавить метрики после fine-tune.

Артефакт: `annotation_budget_simulation.md`.

Фраза для презентации: "Мы не требуем размечать весь архив. Система строит следующий annotation batch так, чтобы за минимальное число правок закрыть максимум неопределенности в решении".

### 11. Reference-tool interoperability

BioImage Model Zoo / ImageJ / QuPath / ilastik ecosystem подсказывает, что хороший science workflow должен уметь экспортировать и импортировать результаты в понятных форматах.

Быстрые экспорты:

- masks as PNG/TIFF;
- `components.csv`;
- `annotations.geojson`;
- `cvat_tasks/` or COCO-style masks;
- `qupath_like_measurements.csv`;
- `fiji_imagej_macro_example.ijm`, даже если минимальный.

Это не обязательно для score, но добавляет credibility: результат можно проверить вне нашего приложения.

### 12. "Small model on edge, strong model for annotation" distillation story

MONAI/AIDE/RootPainter и Reddit-сигналы дают хорошее объяснение, зачем использовать тяжёлые модели для разметки, если они уже умеют сегментировать: тяжелая модель/ассистент работает offline для annotation, а финальный pipeline должен быть воспроизводимым, быстрым и explainable.

Формулировка:

> Foundation/interactive models are used as annotation engines. The submitted ore classifier remains a deterministic, auditable pipeline: tiled segmentation, component metrics, talc fraction, decision margins, and reproducible reports.

## Deep search: приоритетный backlog после найденных papers

### P0 differentiation, если времени мало

1. `source_votes.png` / `disagreement_map.png` для сульфидов.
2. `component_passport.csv` + overlay with component ids.
3. `ore_classification.json` с decision margin и `needs_expert_review`.
4. `tiling_summary.json` и tile-offset robustness check.
5. Review queue: top-20 crops by `uncertainty * decision_impact`.
6. Model card: exact provenance of proxy labels, official pseudo labels, non-expert QA.
7. Illumination/flat-field normalization artifact and before/after fraction stability check.
8. High-loss tile export for pseudo-label cleaning.
9. Dataset curation report: near duplicates, hard tiles, label-issue candidates, representative/outlier crops.
10. OM-derived liberation/association proxy report from connected components.

### P1 research upgrade

1. STAPLE/Snorkel-like label model for weak source fusion.
2. Co-teaching or high-loss tile rejection during pseudo-label training.
3. MIL/CLAM-like ordinary/fine classifier over sulfide components.
4. FDA-style microscope color adaptation and domain-stress validation.
5. DINOv2/SSL embeddings for component retrieval and MIL, after official-crop adaptation.
6. Scribble-trained shallow pixel classifier as an extra source in `source_votes`.
7. MicroSAM/MatSAM-inspired prompt planning for controlled SAM-assisted masks.
8. Microscopy-pretrained encoder / official-crop SSL ablation.
9. Corrective-annotation retrain loop with `review_patch_manifest.csv` and `retrain_delta.md`.

### Не стоит брать в P0

1. Full adversarial domain adaptation as required path.
2. Heavy Mask2Former/foundation pipeline as only answer.
3. Synthetic talc images as evidence of measured accuracy.
4. End-to-end ore-class classifier without component metrics.
5. Any claim that public data gives talc ground truth for this task.
6. A generic annotation platform clone without ore-specific decision impact.
7. SAM/μSAM/MatSAM claims without checking actual official OM crops.

## Что можно быстро сделать перед сдачей

### Быстрые high-impact задачи

1. `disagreement_map.png` для binary sulfide dataset: source votes + ignore overlay.
2. `component_passport.csv` и debug overlay с id компонентов.
3. Простая decision margin в `ore_classification.json`: расстояние до 10% талька и до 50/50 ordinary/fine.
4. TTA на 4 преобразованиях для финального checkpoint: mean mask + uncertainty.
5. Очередь 20 спорных crops для ручной проверки; сохранить patches и показать before/after.
6. Ablation markdown/csv: heuristic vs ResUNet vs SegFormer, даже на proxy/held-out smoke.
7. Model card: что обучено на официальном, что на proxy, что является weak supervision.
8. Dataset curation report: hard tiles, near-duplicates/leaky split candidates, outlier crops.
9. Mineral liberation-style component report: размерные распределения, контакты, association matrix.
10. `expert_questions.md` для 20 самых важных вопросов к геологу.

### Средние задачи, если есть еще день

1. CPS/teacher-student fine-tune на official pseudo labels.
2. SAM2-assisted sulfide QA app по аналогии с talc review.
3. MIL-классификатор ordinary/fine по component features.
4. Robustness certificate для 3-5 панорам.
5. Boundary/Focal/Dice ablation для talc и sulfide.
6. Scribble RF/ExtraTrees pixel classifier как быстрый per-image adaptation baseline.
7. MicroSAM/MatSAM-like prompt planner for SAM2 candidate masks.
8. Microscopy-pretrained encoder or SSL ablation if dependencies are manageable.

## Что лучше не делать

- Не возвращать SEM/XRD в основной путь: это размоет официальную историю.
- Не делать главным решением `image -> ore_class` без маски и объяснений.
- Не называть pseudo-labels экспертной разметкой.
- Не валидировать на тайлах случайным split, если тайлы из одного образца попадают и в train, и в val.
- Не использовать synthetic/generative images как proof качества. Можно только как robustness augmentation.
- Не давать talc-positive label всем темным зонам; нужны hard negatives.
- Не обещать geologist-level ground truth, если разметка сделана non-expert QA.

## Проверенные papers и источники

- SAM2: promptable segmentation, data engine with user interaction, faster image segmentation than original SAM. Relevance: SAM2 assist for QA and boundary correction, not final geological classifier. https://arxiv.org/abs/2408.00714
- SAM-enhanced WSSS: CAM pseudo-labels can be refined with SAM masks to improve weakly supervised segmentation from image-level labels. Relevance: official class folders can guide candidate regions. https://arxiv.org/abs/2305.05803
- WSSS survey from image-level labels to foundation models. Relevance: validates the image-level-label strategy and its limits. https://arxiv.org/abs/2310.13026
- Cross Pseudo Supervision for semi-supervised semantic segmentation. Relevance: two-model consistency on official unlabeled/pseudo-labeled images. https://arxiv.org/abs/2106.01226
- Active Learning for Semantic Segmentation with Multi-class Label Query. Relevance: region/superpixel-level annotation can be cheaper than full masks. https://arxiv.org/abs/2309.09319
- Deep ensembles for uncertainty. Relevance: confidence/disagreement maps and OOD warning for microscopy shifts. https://arxiv.org/abs/1612.01474
- Temperature scaling/calibration. Relevance: probabilities in reports should be calibrated, not raw softmax confidence. https://arxiv.org/abs/1706.04599
- Boundary loss for highly unbalanced segmentation. Relevance: talc and thin boundaries are imbalanced; boundary-aware losses may help. https://arxiv.org/abs/1812.07032
- LumenStone dataset page. Relevance: polished-section proxy data with pixel masks; S2 is close to Norilsk sulfide associations, but not official labels. https://imaging.cs.msu.ru/en/research/geology/lumenstone
- Uncertainty to expand training sets for mineral segmentation, LumenStone S1v2. Relevance: domain/color distortions and uncertainty-based sample selection in geological polished sections. https://isprs-archives.copernicus.org/articles/XLVIII-2-W9-2025/123/2025/
- DeepLabv3+ reflected-light ore microscopy segmentation. Relevance: published evidence that reflected-light ore segmentation can exceed 90% F1, but domain shift matters. https://doi.org/10.1016/j.mineng.2021.107007
- Domain adaptation for reflected-light microscopy mineral segmentation. Relevance: models trained on one ore/sample/setup can generalize poorly; adaptation and official calibration are important. https://www.preprints.org/manuscript/202412.0572
- Segmenteverygrain. Relevance: practical pattern "U-Net prompts + SAM refinement + large-image tiling + interactive editing" for grain-like objects. https://doi.org/10.21105/joss.07953
- Petro-SAM. Relevance: recent petrographic direction combining boundary and semantic segmentation; useful inspiration, not a drop-in model for this reflected-light ore task. https://arxiv.org/abs/2604.14805
- RoImAI rock thin-section foundation model. Relevance: strong evidence that geological microscopy benefits from foundation-style models, but it targets thin sections and lithology, not official sulfide/talc OM classes. https://doi.org/10.1038/s44172-025-00565-5
- Ore Petrography Using Optical Image Analysis. Relevance: OIA on polished sections can quantify ore minerals, grain size, liberation, and associations; supports component passports and reproducible protocol framing. https://doi.org/10.3390/geosciences6020030
- Semi-automated iron ore characterisation based on reflected-light optical microscope analysis. Relevance: direct precedent for RLOM-based mineralogical characterisation and liberation analysis with image processing. https://doi.org/10.1016/j.mineng.2015.10.016
- Enhanced thresholding for non-opaque mineral segmentation in OIA. Relevance: warns against naive thresholding of non-opaque/dark phases and supports hard-negative talc strategy. https://doi.org/10.3390/min13030350
- Automated quantitative mapping by multispectral reflected-light microscopy. Relevance: supports the claim that reflected-light microscopy can be used for quantitative mineral mapping at lower operational burden than SEM-EDS systems. https://doi.org/10.1017/S1431927622003063
- Snorkel weak supervision. Relevance: formalizes weak sources as labeling functions with unknown accuracy/correlation; maps well to Petroscope, heuristic, SegFormer, ResUNet, artifact and talc rules. https://arxiv.org/abs/1711.10160
- STAPLE label fusion. Relevance: EM-style estimation of latent truth and annotator/source performance for multiple segmentation masks; useful as a conceptual upgrade from majority vote. https://doi.org/10.1109/TMI.2004.828354
- Co-teaching for noisy labels. Relevance: two-network training can reduce memorization of pseudo-label errors; possible upgrade for weak sulfide masks. https://arxiv.org/abs/1804.06872
- Attention-based Deep Multiple Instance Learning. Relevance: image-level official labels can supervise bags of sulfide components while attention explains which components drove the decision. https://arxiv.org/abs/1802.04712
- CLAM weakly supervised whole-slide learning. Relevance: high-resolution microscopy with slide-level labels, attention heatmaps, and interpretability; good analogue for official panorama/image-level labels. https://arxiv.org/abs/2004.09666
- ADVENT domain adaptation for segmentation. Relevance: shows entropy/adversarial ideas for domain-shifted semantic segmentation; useful as research reference, risky for P0. https://arxiv.org/abs/1811.12833
- FDA Fourier Domain Adaptation. Relevance: simple style/statistics transfer idea for microscope color/illumination shift. https://arxiv.org/abs/2004.05498
- Tent test-time adaptation. Relevance: source-free adaptation via entropy minimization and normalization statistics; possible experiment, not default without safety checks. https://arxiv.org/abs/2006.10726
- U-Net overlap-tile strategy. Relevance: canonical argument for seamless segmentation of arbitrarily large images under GPU memory limits. https://arxiv.org/abs/1505.04597
- Exact tile-based segmentation inference. Relevance: formal tile/halo/receptive-field math for microscopy images larger than GPU memory. https://doi.org/10.6028/jres.126.009
- Hann windows for patch-based segmentation. Relevance: windowed blending can reduce edge effects in tiled segmentation outputs. https://doi.org/10.1371/journal.pone.0229839
- Focal Tversky loss. Relevance: small rare-region segmentation; candidate for talc where false negative/false positive tradeoff matters. https://arxiv.org/abs/1810.07842
- Lovasz-Softmax loss. Relevance: direct IoU/Jaccard optimization; useful ablation for sulfide/talc segmentation. https://arxiv.org/abs/1705.08790
- DINOCell / microscopy-adapted self-supervised pretraining. Relevance: warns that natural-image features may be poorly aligned with microscopy unless adapted on domain images. https://arxiv.org/abs/2604.10609
- Reddit, materials microscopy segmentation discussion. Relevance: SAM auto masks can oversegment grain boundaries; boundary-first reasoning and preprocessing matter for measurement tasks. https://www.reddit.com/r/computervision/comments/1rwfdfz/segmentation_of_materials_microscopy_images/
- Reddit, SAMAT annotation tool thread. Relevance: local brush/magic-wand annotation UX with SAM masks is attractive for small focused projects. https://www.reddit.com/r/MachineLearning/comments/16r0pa5/p_made_a_simple_semantic_segmentation_annotation/
- Reddit, modern segmentation annotation practices. Relevance: practitioners want model-in-the-loop annotation and exportable workflows, not just manual drawing in a generic tool. https://www.reddit.com/r/computervision/comments/1cmwo4f/modern_best_practices_for_image_segmentation_tasks/
- Reddit, small microscopy segmentation dataset thread. Relevance: limited data, magnification shifts, class imbalance, augmentation limits, and model-assisted annotation are recurring practical blockers. https://www.reddit.com/r/computervision/comments/1hrxomf/help_with_image_segmentation/
- Reddit, microscopic object segmentation by image processing. Relevance: flat-field/vignette correction before thresholding is a simple practical improvement for microscopy masks. https://www.reddit.com/r/computervision/comments/1jjfmys/object_segmentation_in_microscopic_images_by/
- Reddit, semi-automatic labeling for industry images. Relevance: small labelled subset -> own model -> predictions -> human correction is the common robust loop; generic VLMs/SAM are insufficient for novel objects. https://www.reddit.com/r/computervision/comments/1ho0950/best_tools_or_models_for_semiautomatic_labeling/
- Reddit, moving away from manual labeling. Relevance: strong warning that "SAM wrapper" alone is weak; active learning, gold sets, and human review are where value appears. https://www.reddit.com/r/computervision/comments/1mhyr5d/anyone_else_moving_away_from_traditional_label/
- Reddit, selecting data to annotate. Relevance: active learning should decide between relabeling suspicious examples and labeling novel examples; useful for review queue design. https://www.reddit.com/r/computervision/comments/10d72lu/strategies_for_selecting_what_data_to_annotate/
- Reddit, mineral microscope property-table project. Relevance: mineral identification UX benefits from structured observations and reference properties; maps to component passport/explainability cards. https://www.reddit.com/r/geology/comments/m3p1dw/a_friend_and_i_made_a_website_to_help_identify/
- Reddit, dense instance segmentation with sliding windows. Relevance: multi-scale tiled inference needs coordinate remapping, stitching, and boundary-crossing component flags. https://www.reddit.com/r/computervision/comments/1meqpd2/instance_segmentation_nightmare_2700x2700_images/
- Segment Anything for Microscopy / μSAM. Relevance: domain-adapted SAM for microscopy supports interactive/automatic annotation and fine-tuning; reinforces SAM-as-assist rather than default natural-image SAM as truth. https://www.nature.com/articles/s41592-024-02580-4
- micro-sam GitHub. Relevance: concrete napari tool pattern for interactive 2D/3D microscopy segmentation, tracking, and fine-tuning. https://github.com/computational-cell-analytics/micro-sam
- ilastik. Relevance: proven interactive pixel/object classification workflow from sparse user labels and immediate feedback; inspiration for scribble-based ore adaptation. https://www.ilastik.org/
- QuPath pixel classification. Relevance: separates pixel classification, object creation, feature selection, probability overlays, and downstream measurements. https://qupath.readthedocs.io/en/stable/docs/tutorials/pixel_classification.html
- Labkit. Relevance: big-image microscopy labeling/segmentation with sparse scribbles, random forests, BigDataViewer, macro/HPC support; good model for large panoramas. https://doi.org/10.3389/fcomp.2022.777728
- Trainable Weka Segmentation. Relevance: lightweight trainable pixel classifier from limited annotations and image features; good fallback baseline and interpretable source. https://doi.org/10.1093/bioinformatics/btx180
- RootPainter3D / corrective annotation. Relevance: human corrects model errors, sparse corrected regions train the model, and correction time decreases; maps directly to review patches. https://arxiv.org/abs/2106.11942
- MONAI Label. Relevance: AI-assisted labeling service with interactive/automated segmentation, active learning, and continuous learning from user corrections. https://monai.readthedocs.io/projects/label/en/latest/
- AIDE active-learning annotation framework. Relevance: explicit loop of human annotations, model training, predictions, and selection of next images to annotate. https://github.com/microsoft/aerial_wildlife_detection
- FiftyOne Brain. Relevance: uniqueness, near duplicates, leaky split detection, mistakenness, hardness, representativeness, similarity and embeddings for dataset curation. https://docs.voxel51.com/brain.html
- cleanlab semantic segmentation label issues. Relevance: uses labels plus out-of-sample predicted probabilities to find likely mislabeled segmentation masks. https://docs.cleanlab.ai/v2.7.1/tutorials/segmentation.html
- MicroNet microscopy-pretrained encoders. Relevance: microscopy-domain pretraining can outperform ImageNet and improve few-shot/out-of-distribution segmentation. https://www.nature.com/articles/s41524-022-00878-5
- NASA pretrained microscopy models. Relevance: downloadable MicroNet encoders and code for transfer learning in microscopy segmentation. https://github.com/nasa/pretrained-microscopy-models
- Active learning for microstructure segmentation with tiny annotation budgets. Relevance: semi-supervised active learning can approach full-data performance with a small fraction of annotations; supports annotation-budget simulator. https://www.sciencedirect.com/science/article/pii/S2405829724006111
- Self-supervised learning in electron microscopy. Relevance: unlabeled microscopy pretraining can improve fine-tuning for segmentation and related tasks under limited annotation. https://arxiv.org/abs/2402.18286
- MatSAM. Relevance: material-microscopy-specific SAM prompt engineering and postprocessing for dense/dispersed structures; useful for controlled SAM2 prompt planning. https://arxiv.org/abs/2401.05638
- Mineral Liberation Analysis overview. Relevance: industrial reporting language around abundances, associations, sizes, shapes and grain boundaries; inspires OM-derived liberation/association proxy. https://www.mun.ca/creait/micro-analysis-facility/sem-mla/mineral-liberation-analysis-mla/
- ZEISS microscopy mining applications. Relevance: commercial microscopy workflows emphasize mineral classification, measurement, morphology, liberation and associations. https://www.zeiss.com/microscopy/en/applications/raw-materials-and-industrial-rd/mining.html
- Model Cards for Model Reporting. Relevance: structured model reporting, intended use, evaluation and limitations; maps to final model card. https://arxiv.org/abs/1810.03993
- Datasheets for Datasets. Relevance: dataset documentation for provenance, composition, collection and recommended use; maps to official/pseudo/QA dataset card. https://arxiv.org/abs/1803.09010

## Recommended next framing for presentation

Фраза для слайда:

> Мы используем не один "черный ящик", а исследовательский контур слабой разметки: несколько независимых источников маски, карта разногласий, точечная доразметка самых важных зон, переобучение и интерпретируемый отчет по каждому включению.

Фраза для вопроса про ручную разметку:

> Если появляется время эксперта, система не просит размечать все подряд. Она ранжирует 20-50 областей, где разметка даст максимальный прирост: зоны конфликта моделей, пороговые случаи талька и компоненты, влияющие на ordinary/fine решение.

Фраза для вопроса про научность:

> Мы сознательно разделяем ground truth, pseudo-label и non-expert QA. Это делает результат воспроизводимым: любой процент в отчете можно проследить до маски, источников сигнала, правок и параметров запуска.
