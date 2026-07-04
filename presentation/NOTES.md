- вылодить ALOLA Talc Annotation Demo (youtube, yandex disk)
- выложить AOLOA E2E UI Demo (yandex диск)

- добавить ссылку на ALOLA Talc Annotation Demo (yandex диск)
- добавить ссылку на AOLOA E2E UI Demo (yandex диск)

- Запустить на сервере 
- Сделать сервер не убиваемым

- Запустить второй интстанс
- Презентация

- Открыть репозитарий на github
- Открыть резервный репозитарий на сервере 
https://sourcecraft.dev/s1mb1o/2026-ore-thin-section-classifier

Уважаемые участники, ники на Github для доступа:
suharkom
MrTimmy89
Hammoudmsh
GibsonGrid
Plank-tony

Загрузить решение до 23:59



Решение должно быть представлено на платформу не позднее 4 июля 23:59 в следующем виде:
1. Ссылка на исходный код в VCS (системе контроля версий - GitHub, GitLab, Mercury или иные)
2. Ссылка на облачный диск (Яндекс, Google), где загружены:
1. Архив с исходным кодом проекта
2. Видео-демо работы проекта (видео, показывающее процесс работы вашего решения, с комментариями или без них, не длиннее 5 минут)
3. Ссылка на презентацию вашего проекта (облачный диск с файлом .pptx/.pdf или развернутая презентация на YouNote, Figma или иных сервисах)
4. Ссылка на ваше развернутое решение (при наличии) для его тестирования членами жюри.


---

Correct: heuristic sulfide mask
Correct: bright-phase proxy

Bright-phase proxy is our heuristic sulfide detector, not a neural model.
We use it as bootstrap.


Сделать режим панорамы

---

SegFormer-B2 sulfide/non-sulfide


Почти правильно, но я бы поправил формулировку.

Правильная версия:

1. Берём **LumenStone S1_v1 + S2_v2**, где **S2 ближе к Норильским сульфидным Cu-Ni рудам**.
2. Схлопываем минеральные классы LumenStone/Petroscope в бинарную маску:
   `sulfide / non-sulfide`.
   В sulfide входят chalcopyrite, galena, bornite, pyrrhotite, pyrite, pentlandite, sphalerite и др.; magnetite/hematite/native gold исключены.
3. Добавляем предоставленный Nornikel dataset, но не как clean GT, а как **weak labels** через bright-phase heuristic pseudo-mask.
4. Обучаем бинарные модели сегментации.
5. Сравниваем архитектуры.
6. Выбираем **SegFormer-B2** как текущий лучший sulfide/non-sulfide segmenter.

Важные поправки:

- Не только LumenStone S2: в текущем `binary_sulfide_dataset_v0` использованы **S1_v1 + S2_v2**.
- Не “дообучение используя модели Petroscope на Nornikel dataset” в строгом смысле. Фактически текущий B2 обучен на смеси:
  - LumenStone masks: `2976` tiles;
  - official Nornikel heuristic pseudo-labels: `5560` tiles.
- Petroscope/LumenStone используются как доменный proxy и источник mineral class ids/masks. Но текущий B2 checkpoint не является “Petroscope teacher -> Nornikel fine-tune” в чистом виде.
- В текущем documented benchmark я не вижу Mask2Former. Фактически сравнивались: **SegFormer-B0, SegFormer-B1, SegFormer-B2, ResUNet**.

Метрики выбора:

| Model | sulfide IoU | F1 | AUC | HD95 |
|---|---:|---:|---:|---:|
| SegFormer-B2 | **0.974381** | **0.987024** | **0.998811** | **23.57 px** |
| SegFormer-B1 | 0.971548 | 0.985569 | 0.998522 | 26.25 px |
| ResUNet | 0.956436 | 0.977733 | 0.996942 | 37.37 px |
| SegFormer-B0 | 0.953371 | 0.976129 | 0.996154 | 33.92 px |

Итого для слайда:

> Для сегментации сульфидов мы обучили бинарный `sulfide / non-sulfide` сегментер. Использовали LumenStone S1/S2 с минеральными масками, схлопнутыми в бинарную разметку, плюс weak pseudo-labels на официальных изображениях Nornikel. Сравнили ResUNet и семейство SegFormer B0/B1/B2. Лучший результат показал SegFormer-B2: IoU 0.974, F1 0.987, AUC 0.999, HD95 23.6 px на weak-label validation split, поэтому он выбран как основной sulfide/non-sulfide backbone. Метрики не выдаём за экспертную геологическую ground truth, потому что часть official-domain labels получена эвристически.




---

У нас есть режим что мы масщтабируем большие картинки чтобы работать с ними в UI,
при преближении мы используем больший форма (LOD Level Of Details)

Нарисувй 65Kx65K -> несколько вариантов пикселизации

Также если картинка большая - то ичпользуем tiling с нахлестом


--- 

указать утилиты в наличии

---

убрать streamlit

--- 

Сделать презентацию на основуную функциональности

---



