#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-data/external/lumenstone}"
DOWNLOAD_DIR="$ROOT_DIR/downloads/whole"
EXTRACT_DIR="$ROOT_DIR/full"
METADATA_DIR="$ROOT_DIR/metadata/whole"
PART_SIZE="${PART_SIZE:-67108864}"
JOBS="${JOBS:-16}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
EXTRACT="${EXTRACT:-1}"
CURL_MAX_TIME="${CURL_MAX_TIME:-240}"
CURL_SPEED_LIMIT="${CURL_SPEED_LIMIT:-32768}"
CURL_SPEED_TIME="${CURL_SPEED_TIME:-45}"

mkdir -p "$DOWNLOAD_DIR" "$EXTRACT_DIR" "$METADATA_DIR"

manifest() {
  # label|public_url|expected_filename|expected_size|expected_md5
  printf '%s\n' \
    'S1_v1|https://disk.yandex.ru/d/aiWh3rBEwdJ2_g|S1_v1.zip|534897733|4da00e8dc59ea5e840967e4a9fd736f8' \
    'S1_v2|https://disk.yandex.ru/d/ITijo9m-QIxCgg|S1_v2.zip|645664381|1750f9d3827569880002e4a6c0780227' \
    'S2_v1|https://disk.yandex.ru/d/XRKdK6gP1Y_KVA|S2_v1.zip|242253740|e9bcfa73e16aced44f3c7cc6824d08f5' \
    'S2_v2|https://disk.yandex.ru/d/wYK_5JyQy0pIcg|S2_v2.zip|418742024|c739c4fb57f684a7e01b9608c80d1635' \
    'S3_v1|https://disk.yandex.ru/d/hgy7gV-VWSKh3g|S3_v1.zip|273624127|543718d528421a73cfc4b5fbec441e08' \
    'S3_v2|https://disk.360.yandex.ru/d/1ItlsInqs3iiow|S3_v2.zip|5227181560|94edbdf8872fb702334ec99b5696f683' \
    'V1_v1|https://disk.yandex.ru/d/QteBl5VmKR7W9Q|V1_v1.zip|104705467|198b6ce76ac8882194c763e26f1fbb86' \
    'P1_v1_source|https://disk.yandex.ru/d/Oy1zraYV0d0KfQ|source.zip|3249457910|c6f79b981b6538eacdd1081d9d683cbc' \
    'P1_v1_preprocessed|https://disk.yandex.ru/d/s8nJGGj4eE584A|corrected.zip|1772063192|c791cf1cb6df031498ba634219fcbc48' \
    'P1_v1_panoramas|https://disk.yandex.ru/d/OPmF2GTaJFbRFw|results.zip|730418553|b145ff9418f4384e41dd986b3a2f2239' \
    'P1_v2_source|https://disk.360.yandex.ru/d/F2wcAjnIFzGUvQ|src.zip|6270076395|6d6380738d2c1848f6cddba79fad1d19' \
    'P2_v1_source|https://disk.360.yandex.ru/d/oYaKp1xkvA_slQ|src.zip|2775031908|ad732980efceb83c9793fc4245011406'
}

file_size() {
  if [[ -f "$1" ]]; then
    stat -f '%z' "$1"
  else
    printf '0'
  fi
}

file_md5() {
  md5 -q "$1"
}

is_verified() {
  local file="$1" expected_size="$2" expected_md5="$3"
  [[ -f "$file" ]] || return 1
  [[ "$(file_size "$file")" == "$expected_size" ]] || return 1
  [[ "$(file_md5 "$file")" == "$expected_md5" ]] || return 1
}

public_resource_json() {
  local public_url="$1"
  curl -L -sG 'https://cloud-api.yandex.net/v1/disk/public/resources' \
    --data-urlencode "public_key=$public_url"
}

json_field() {
  local field="$1"
  ruby -rjson -e "j=JSON.parse(STDIN.read); v=j[$field]; puts v if v"
}

final_download_url() {
  local direct_url="$1"
  local location
  location="$(curl -sI "$direct_url" | awk '/^location:/ {sub(/^location: /,""); sub(/\r$/,""); print; exit}')"
  if [[ -n "$location" ]]; then
    printf '%s\n' "$location"
  else
    printf '%s\n' "$direct_url"
  fi
}

download_url_for() {
  local label="$1" public_url="$2"
  local json direct_url attempt

  for attempt in 1 2 3 4 5; do
    json="$(public_resource_json "$public_url" || true)"
    printf '%s\n' "$json" > "$METADATA_DIR/$label.resource.json"
    direct_url="$(printf '%s\n' "$json" | json_field '"file"' 2>/dev/null || true)"
    if [[ -n "$direct_url" ]]; then
      final_download_url "$direct_url"
      return 0
    fi
    sleep $((attempt * 2))
  done

  printf 'ERROR: failed to get download URL for %s\n' "$label" >&2
  return 1
}

http_status() {
  awk 'tolower($1) ~ /^http\// {status=$2} END {print status}' "$1"
}

download_part() {
  local final_url="$1" part="$2" start="$3" end="$4" expected_part_size="$5"
  local current_size range_start headers tmp status new_size

  current_size="$(file_size "$part")"
  if [[ "$current_size" == "$expected_part_size" ]]; then
    return 0
  fi
  if [[ "$current_size" -gt "$expected_part_size" ]]; then
    rm -f "$part"
    current_size=0
  fi

  range_start=$((start + current_size))
  headers="$part.headers.$$"
  tmp="$part.chunk.$$"
  rm -f "$headers" "$tmp"

  curl -L --fail --silent --show-error \
    --connect-timeout 20 \
    --max-time "$CURL_MAX_TIME" \
    --speed-limit "$CURL_SPEED_LIMIT" \
    --speed-time "$CURL_SPEED_TIME" \
    --range "${range_start}-${end}" \
    --dump-header "$headers" \
    --output "$tmp" \
    "$final_url" || true

  status="$(http_status "$headers")"
  if [[ "$status" != "206" ]]; then
    rm -f "$headers" "$tmp"
    return 1
  fi

  if [[ -f "$tmp" ]]; then
    cat "$tmp" >> "$part"
  fi
  rm -f "$headers" "$tmp"

  new_size="$(file_size "$part")"
  if [[ "$new_size" -gt "$expected_part_size" ]]; then
    rm -f "$part"
    return 1
  fi
  [[ "$new_size" == "$expected_part_size" ]]
}

download_ranges() {
  local label="$1" url="$2" out="$3" expected_size="$4"
  local part_dir="$DOWNLOAD_DIR/.parts/$label"
  local part_count start end part expected_part_size current_size

  mkdir -p "$part_dir"
  part_count=$(( (expected_size + PART_SIZE - 1) / PART_SIZE ))

  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    local direct_url final_url pids="" pids_count=0 missing=0
    final_url="$(download_url_for "$label" "$url")"

    for i in $(seq 0 $((part_count - 1))); do
      start=$((i * PART_SIZE))
      end=$((start + PART_SIZE - 1))
      if [[ "$end" -ge $((expected_size - 1)) ]]; then
        end=$((expected_size - 1))
      fi
      part="$part_dir/part_$(printf '%02d' "$i")"
      expected_part_size=$((end - start + 1))
      current_size="$(file_size "$part")"
      if [[ "$current_size" != "$expected_part_size" ]]; then
        missing=1
        download_part "$final_url" "$part" "$start" "$end" "$expected_part_size" &
        pids="${pids} $!"
        pids_count=$((pids_count + 1))

        if [[ "$pids_count" -ge "$JOBS" ]]; then
          for pid in $pids; do
            wait "$pid" || true
          done
          pids=""
          pids_count=0
        fi
      fi
    done

    if [[ "$missing" == "0" ]]; then
      break
    fi

    for pid in $pids; do
      wait "$pid" || true
    done
  done

  for i in $(seq 0 $((part_count - 1))); do
    start=$((i * PART_SIZE))
    end=$((start + PART_SIZE - 1))
    if [[ "$end" -ge $((expected_size - 1)) ]]; then
      end=$((expected_size - 1))
    fi
    part="$part_dir/part_$(printf '%02d' "$i")"
    expected_part_size=$((end - start + 1))
    current_size="$(file_size "$part")"
    if [[ "$current_size" != "$expected_part_size" ]]; then
      printf 'ERROR: incomplete part for %s: %s has %s bytes, expected %s\n' "$label" "$part" "$current_size" "$expected_part_size" >&2
      return 1
    fi
  done

  rm -f "$out"
  for i in $(seq 0 $((part_count - 1))); do
    cat "$part_dir/part_$(printf '%02d' "$i")" >> "$out"
  done
}

extract_zip() {
  local label="$1" file="$2"
  local target="$EXTRACT_DIR/$label"
  if [[ "$EXTRACT" != "1" ]]; then
    return 0
  fi

  mkdir -p "$target"
  unzip -q -n "$file" -d "$target"
  rm -rf "$target/__MACOSX"
}

download_one() {
  local label="$1" url="$2" expected_filename="$3" expected_size="$4" expected_md5="$5"
  local out="$DOWNLOAD_DIR/${label}__${expected_filename}"

  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$label"

  if is_verified "$out" "$expected_size" "$expected_md5"; then
    printf '  already verified: %s\n' "$out"
  elif [[ "$label" == "S2_v2" ]] && is_verified "$ROOT_DIR/downloads/S2_v2.zip" "$expected_size" "$expected_md5"; then
    ln -f "$ROOT_DIR/downloads/S2_v2.zip" "$out"
    printf '  reused existing verified S2_v2 archive\n'
  else
    download_ranges "$label" "$url" "$out" "$expected_size"
  fi

  if ! is_verified "$out" "$expected_size" "$expected_md5"; then
    printf 'ERROR: verification failed for %s\n' "$label" >&2
    printf '  file: %s\n' "$out" >&2
    printf '  size: %s expected %s\n' "$(file_size "$out")" "$expected_size" >&2
    printf '  md5: %s expected %s\n' "$(file_md5 "$out")" "$expected_md5" >&2
    return 1
  fi

  unzip -tq "$out" >/dev/null
  extract_zip "$label" "$out"
  printf '  ok: %s\n' "$out"
}

manifest > "$METADATA_DIR/manifest.tsv"

while IFS='|' read -r label url expected_filename expected_size expected_md5; do
  download_one "$label" "$url" "$expected_filename" "$expected_size" "$expected_md5"
done < "$METADATA_DIR/manifest.tsv"

printf '[%s] complete\n' "$(date '+%Y-%m-%d %H:%M:%S')"
