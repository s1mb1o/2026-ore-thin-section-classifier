# Scripts

Utility CLIs for dataset manifests, pseudo-label generation, training launchers, and evaluation should live here.

Keep heavy GPU training jobs outside Streamlit. Streamlit may emit the exact command, but training should run as a separate script on the selected GPU host.

## Talc Blue-Line Conversion

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --summary-json outputs/talc_blue_line_conversion_summary.json
```

Use `--sulfide-mask-dir path/to/binary_sulfide_masks` when the binary sulfide
detector produces masks named by image stem. Without that directory, the
converter uses the conservative bright-phase sulfide heuristic.
