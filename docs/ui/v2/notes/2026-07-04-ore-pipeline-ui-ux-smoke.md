# Ore Pipeline v2 UI UX Smoke Test

Date: 2026-07-04

Scope: live v2 ore-pipeline UI at `http://127.0.0.1:63589` in Russian UI mode. Pages checked: Workspace, Status, Settings, History, and Series. Loaded the latest completed run `run_20260704_142126_273705000_3a5a6640` to inspect result visualization, side-by-side comparison, legends, and Edit & Recalculate.

## Summary

No blocking console/runtime errors were found in the checked UI pages. The recent viewer changes mostly work: final-layer legends show class percentages, talc clusters are off by default and cyan/light blue, the side-by-side splitter reaches the full practical range, the preprocessing popup no longer creates page-level overflow, and Edit & Recalculate uses the same bottom-left zoom widget pattern and a slider-based brush size control.

Remaining UX issues are layout/responsiveness issues, not pipeline blockers.

## Checks Performed

- Live browser DOM/geometry checks on `/workspace`, `/status`, `/settings`, `/history`, and `/batch/`.
- Console error scan on all checked pages: no browser console errors observed.
- Loaded a completed run from the Workspace history sidebar.
- Inspected final-view class legend, viewer option row, Russian mouse hints, Status model spacing, preprocessing settings popup, and Edit & Recalculate popup.
- Switched side-by-side comparison to `сульфиды` and dragged the splitter to both extremes.
- Temporary narrow viewport smoke at `390 x 844`, then reset viewport.
- Python regression suite: `python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v` passed, 48 tests.

## Findings

### P1: Status page creates page-level horizontal overflow

Evidence:

- Desktop/current browser viewport: document width `1032 px`, scroll width `1159 px`.
- Mobile smoke viewport: document width `375 px`, scroll width `1153 px`.
- The `status-grid` / panel layout remains about `1143 px` wide, so cards and the refresh button extend past the visible page instead of wrapping to the content column.

Impact:

- The Status page needs horizontal scrolling even though its content is dashboard-like and should be readable without page-level sideways scroll.
- On narrow screens, all status cards become effectively a fixed desktop-width strip.

Suggested fix:

- Make `status-page`, `status-grid`, `status-log-grid`, and status panels use `minmax(0, 1fr)` tracks and `width: 100%`.
- Replace fixed multi-column card sizing with responsive grid rules such as `grid-template-columns: repeat(auto-fit, minmax(220px, 1fr))`.
- Ensure long checkpoint/path text wraps inside cards with `overflow-wrap: anywhere`.

### P2: Workspace viewer option row wraps at the current desktop viewport

Evidence:

- At the live browser width, `показать тайлы`, `только контуры`, and `толщина контура` stay on the first line, while `прозрачность` wraps to a second line.
- This conflicts with the requested one-row viewer control strip under the image.

Impact:

- The control row reads as two unrelated rows, and the opacity control is visually detached from the overlay controls it belongs to.

Suggested fix:

- Keep the controls in a single flex row at normal desktop widths by tightening gaps/labels or allowing the range controls to shrink.
- For small/mobile widths, intentionally switch to a compact stacked layout rather than accidental wrapping.

### P2: Edit & Recalculate statistics labels wrap into syllables

Evidence:

- In the right panel, labels such as `сульфиды`, `не-сульфиды`, and `обычные срастания` wrap into narrow syllable/letter columns.
- The values remain readable, but the metric names are hard to scan.

Impact:

- Expert correction workflows need quick class-stat comparison; broken labels slow review and look unstable.

Suggested fix:

- Increase the right-panel minimum width or use a two-line stat row layout: class name on the first line, value/share on the second.
- Disable aggressive word breaking for stat labels and let rows grow vertically.
- Consider using a compact table with columns `Класс`, `Площадь`, `%` for the editor statistics.

### P2: Workspace mobile smoke still reports page-level overflow

Evidence:

- At `390 x 844`, `/workspace` reports document width `375 px`, scroll width `581 px`.
- No single visible element extended past the viewport in the bounded scan, so the overflow is likely from an internal viewer/segmented-control minimum width or an offscreen retained layer.

Impact:

- The Workspace is not fully mobile-safe yet. This is lower risk for the expected desktop lab workflow, but it can affect demos on narrow browser panes.

Suggested fix:

- Inspect mobile CSS for hidden/offscreen viewer controls and segmented selectors.
- Add a browser regression that asserts `document.documentElement.scrollWidth <= clientWidth + 2` for the empty Workspace at a 390 px viewport.

## Confirmed Working

- Workspace final legend separates ordinary/fine/talc, talc clusters/artefacts, and background; class percentages are shown.
- `кластеры талька` is unchecked by default and uses cyan/light blue.
- Russian hints are present: `Колесо мыши - масштаб` and `Нажатие колеса - панорама`.
- Preprocessing settings popup fits the viewport without horizontal page overflow.
- Status `Модели` card separates sulfide and talc model details with a blank line and `white-space: pre-line`.
- History table `Статус` column shows `Готово`; the wide table is contained in its own scroll area rather than creating page-level overflow.
- Series and Settings pages did not show page-level overflow in the checked desktop/mobile smoke.
- Side-by-side splitter was visible for `сульфиды` comparison and could be dragged from about `0.2%` to `99.7%` of the viewer width.
- Edit & Recalculate popup opens for the loaded run; the brush size is a `2..240 px` slider with live `px` output, and the bottom-left zoom widget plus mouse hint row are present.

## Test Notes

- The project unittest suite for `tests/test_ore_pipeline_web.py` passed: 48 tests.
- The browser pytest suite was not runnable in this local Python environment because `pytest` is not installed for `/opt/homebrew/opt/python@3.14/bin/python3.14`.
- This was a UX smoke pass; it did not start new runs, delete history, save settings, or apply destructive edits.
