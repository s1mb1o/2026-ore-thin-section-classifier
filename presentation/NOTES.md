selectel я оплатил, должно хватить до ПН


Москва / ru-7a
16 vCPU 
– 64 ГБ RAM 
– 300 ГБ
– 1 × L4  24 ГБ, GPU
Alize: 111.88.124.80

RUTUBE:
https://studio.rutube.ru/videos/?show_moderation=1&ordering=calculated_date_asc#shorts

YOUTUBE:
https://studio.youtube.com/channel/UCh1sXbWa7snmpUknHG_QEig/videos/upload?filter=%5B%5D&sort=%7B%22columnType%22%3A%22date%22%2C%22sortOrder%22%3A%22DESCENDING%22%7D


Решение должно быть представлено на платформу не позднее 4 июля 23:59 в следующем виде:
1. Ссылка на исходный код в VCS (системе контроля версий - GitHub, GitLab, Mercury или иные)
2. Ссылка на облачный диск (Яндекс, Google), где загружены:
1. Архив с исходным кодом проекта
2. Видео-демо работы проекта (видео, показывающее процесс работы вашего решения, с комментариями или без них, не длиннее 5 минут)
3. Ссылка на презентацию вашего проекта (облачный диск с файлом .pptx/.pdf или развернутая презентация на YouNote, Figma или иных сервисах)
4. Ссылка на ваше развернутое решение (при наличии) для его тестирования членами жюри.


- вылодить ALOLA Talc Annotation Demo (youtube, yandex disk)
- выложить AOLOA E2E UI Demo (yandex диск)

- добавить ссылку на ALOLA Talc Annotation Demo (yandex диск)
- добавить ссылку на AOLOA E2E UI Demo (yandex диск)

- Запустить на сервере Alize
+ Сделать сервер не убиваемым 
- настроить DNS nornickel-ai-hackathon.alola.ru
- настроить ssh caddy
- lets encrypt
- https


- Запустить второй интстанс на gx10
+ настроить nornickel-ai-hackathon.3simbio.ru
- 
username: reviewer
password: zder96od

security find-generic-password -s nornickel-backup-basic-auth -a reviewer -w


https://nornickel-ai-hackathon.my.3simbio.ru/
https://nornickel-backup.my.3simbio.ru/workspace



- Презентация

- Открыть репозитарий на github
- Открыть резервный репозитарий на сервере 
https://sourcecraft.dev/s1mb1o/2026-ore-thin-section-classifier

Удалить упоминание что мы мониторили чужие проекты


example evidence pack на сайте


---

То есть LumenStone уже часть текущего обучения; Petroscope пока не надо заявлять как использованный teacher для B2, если мы не прогоняли его predictions и не включали их в manifest.




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


---

первая страница ограничения

неразмеченные данные и прочее



лабость эвристики (пре-лейбл)
Правило «тонкое» = OR(замещение внутри, низкая solidity, низкая compactness).
→ массивное зерно с однородным ядром, но сильно рваным контуром
   помечается «труднообогатимым» из-за границы, при нулевом замещении.

Решение: grain-app (human-in-the-loop) — приложение готово,
нужна разметка зёрен из dataset → обучение grain-классификатора,
который отделит «рваный край» от «тонкого срастания».
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


Коротко: да, но не как “берём Petroscope как финальную модель”.
Правильная история для обсуждения такая:
LumenStone/Petroscope как внешний доменный proxy
Petroscope/LumenStone дают нам минералогические классы polished-section микрофото.
Мы схлопываем минеральные классы в sulfide / non-sulfide.
Это даёт стартовую weak/silver разметку, но не финальную истину для Норникеля.

SegFormer-B2 как текущий production segmenter
Мы сравнили ResUNet, SegFormer-B0/B1/B2, теперь ещё гоняем Mask2Former.
Текущий лучший уже выбран: SegFormer-B2, fine-tuned от ImageNet-pretrained nvidia/mit-b2, не from scratch.
Его задача: стабильная бинарная маска сульфидов на наших снимках.

Дальше адаптация на dataset Норникеля
Прогоняем B2 по норникелевским изображениям.
Где есть экспертные/ручные исправления или QA disagreement, используем это для дообучения.
Petroscope можно использовать как teacher/vote source: если Petroscope, B2 и heuristic согласны, это high-confidence; если расходятся, пиксели лучше в ignore или в очередь QA.

Фраза для слайда/защиты:
Используем LumenStone/Petroscope как внешний доменный источник и teacher для weak-supervision. На этой основе обучаем и выбираем собственный binary sulfide/non-sulfide segmenter. Текущий лучший checkpoint — SegFormer-B2. Далее адаптируем его на данных Норникеля через pseudo-label agreement, expert QA и дообучение на исправленных масках.

То есть “после B2 идёт не Petroscope вместо B2, а domain adaptation B2 на норникелевских данных; Petroscope остаётся внешним teacher/baseline источником”.



B0/B1/B2 are model size variants of the SegFormer encoder family, specifically MiT: nvidia/mit-b0, nvidia/mit-b1, nvidia/mit-b2.
They mainly mean:
Variant	Meaning	Practical effect
SegFormer-B0	smallest MiT encoder	fastest, least memory, usually lower quality
SegFormer-B1	medium-small	better quality, still fairly fast
SegFormer-B2	larger encoder	slower/heavier, usually better masks

In our benchmark this pattern held:
B0: smallest fallback, best sulfide IoU 0.953371
B1: faster fallback, best sulfide IoU 0.971548
B2: current best, best sulfide IoU 0.974381
They are not “version 0/1/2 of our code”; they are standard backbone capacity levels from the SegFormer/MiT model family.



Не противоречие, но там легко перепутать три разные роли. Точнее надо говорить так:
LumenStone сейчас реально использован как внешний датасет с mineral masks/class ids. Мы схлопываем его классы в sulfide / non-sulfide.
Petroscope сейчас лучше называть не источником текущего B2 checkpoint, а доменным baseline/teacher-кандидатом. Его class ids/модель полезны для плана weak supervision, но текущий B2 не надо описывать как “Petroscope predictions → fine-tune on Nornikel”.
SegFormer-B2 сейчас обучен как наш student/checkpoint на binary_sulfide_dataset_v0: LumenStone masks + pseudo-labels/heuristics по официальным изображениям. Это не чистый Petroscope-teacher pipeline.


Текущий SegFormer-B2 обучен на binary weak/silver dataset: LumenStone mineral masks, схлопнутые в sulfide/non-sulfide, плюс pseudo-labels для официальных изображений. Petroscope рассматриваем как внешний доменный baseline/teacher для следующего этапа: disagreement maps, high-confidence pseudo-labels и domain adaptation на данных Норникеля.

То есть LumenStone уже часть текущего обучения; Petroscope пока не надо заявлять как использованный teacher для B2, если мы не прогоняли его predictions и не включали их в manifest.



Yes. Standard SegFormer/MiT variants go beyond B2:
Variant	Typical backbone	Relative size
B0	nvidia/mit-b0	smallest
B1	nvidia/mit-b1	small
B2	nvidia/mit-b2	medium
B3	nvidia/mit-b3	larger
B4	nvidia/mit-b4	large
B5	nvidia/mit-b5	largest common SegFormer variant

For our case, B3/B4/B5 are possible next experiments, but not automatically better for the hackathon pipeline:
They need more VRAM/time.
Our current validation is weak-label/pseudo-label based, so bigger can overfit label noise.
B2 already gives strong numbers: sulfide IoU 0.974381, F1 0.987024, AUC 0.998811.
B3 is the reasonable next SegFormer experiment; B4/B5 only make sense if B3 improves image-level ore classification or human QA quality, not just weak-label tile IoU.



IoU сейчас проверяется на validation split из outputs/binary_sulfide_dataset_v0/manifest.json: это 512px tiles, всего 8536 tiles, split 6948 train / 1588 val.
Почему так:
это единственный готовый pixel-level validation set для sulfide / non-sulfide, совместимый со всеми моделями;
он смешанный: LumenStone mineral masks, схлопнутые в binary sulfide, плюс weak/pseudo-label tiles из официальных изображений;
поэтому это model-selection IoU, а не финальная геологическая accuracy.
Почему “только так” сейчас:
в официальном датасете Норникеля в основном image-level классы, а не экспертные pixel masks сульфидов;
talc masks у нас есть для талька, но не для полноценной sulfide/non-sulfide валидации;
поэтому для honest validation нужно ещё сделать human QA/expert masks или хотя бы disagreement review pack.


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




\


---


We should improve in three places, but keep our current architecture. Do not pivot into a late full model rewrite.
1. Make the reviewer surface look finished
AlexUser97 is better than us at first impression: Streamlit dashboard, PDF/passport, CLI, Docker, batch, JSON/CSV, uncertainty/status. Our API is more auditable, but it can look less “demo-ready”.
Improve:
Add/finalize a compact result/passport page: original image, mask/overlay, talc %, final class, warning flags, artifact links.
Add a one-command demo path with 1-3 known examples.
Prefer “Демонстрационный отчет / паспорт шлифа” wording over accreditation/protocol claims.
2. Add clearly labeled hybrid talc QA
OpiumProger’s strongest useful idea is U-Net OR dark_gray_phase plus panorama handling. We should not copy their heuristic as ground truth, but we should expose our non-neural dark-zone talc QA cleanly.
Improve:
Label dark-zone output as non_neural_talcose_qa or dark_gray_phase_heuristic.
Show it as a reviewer/support signal, not the official talc mask unless manually accepted.
Keep the official decision path anchored on talc_fraction > 0.10.
3. Tighten trust and claims
Both competitor repos make stronger metric/model claims than we should copy without proof. AlexUser97 also has a dangerous threshold mismatch: README says >10%, code uses 0.15.
Improve:
Add a threshold consistency check/test for every final path: >10% / 0.10.
Do not claim F1/AUC unless we can show grouped split, leakage audit, and exact validation source.
Make our honesty a feature: “auditable deterministic first slice + explicit review flags” is safer than unsupported neural claims.
I would prioritize in this order: final passport/result surface, example evidence pack, threshold/claim audit, then optional hybrid talc QA polish. That beats both repos on judge trust while closing the packaging gap.