#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/astria_person_smoke.sh <READ_PACK_ID> [CANARY_PACK_ID CANARY_TUNE_ID]
#
# Example (read-only):
#   scripts/astria_person_smoke.sh 4343
#
# Example (read-only + canary generation):
#   scripts/astria_person_smoke.sh 4343 4343 1504944

READ_PACK_ID="${1:-}"
CANARY_PACK_ID="${2:-}"
CANARY_TUNE_ID="${3:-}"

if [[ -z "${READ_PACK_ID}" ]]; then
  echo "Usage: $0 <READ_PACK_ID> [CANARY_PACK_ID CANARY_TUNE_ID]" >&2
  exit 1
fi

API_KEY="${ASTRIA_API_KEY:-${PRISMALAB_ASTRIA_API_KEY:-}}"
if [[ -z "${API_KEY}" ]]; then
  echo "Set ASTRIA_API_KEY (or PRISMALAB_ASTRIA_API_KEY) first." >&2
  exit 1
fi

BASE_URL="${ASTRIA_BASE_URL:-https://api.astria.ai}"

echo "== READ-ONLY CHECK (no charges) =="
PACK_JSON="$(curl -fsS "${BASE_URL}/p/${READ_PACK_ID}" \
  -H "Authorization: Bearer ${API_KEY}")"

echo "${PACK_JSON}" | jq '{
  id,
  title,
  slug,
  model_type,
  costs,
  cost_by_class,
  num_images_by_class,
  default_num_images
}'

HAS_PERSON_COST="$(echo "${PACK_JSON}" | jq -r 'if (.costs // {} | has("person")) then "yes" else "no" end')"
echo "supports_costs.person=${HAS_PERSON_COST}"

if [[ -z "${CANARY_PACK_ID}" || -z "${CANARY_TUNE_ID}" ]]; then
  echo
  echo "Canary skipped (pass CANARY_PACK_ID + CANARY_TUNE_ID to run generation test)."
  exit 0
fi

echo
echo "== CANARY CHECK (charges possible) =="
TITLE="person-smoke-$(date +%s)"
PAYLOAD="$(jq -nc \
  --arg title "${TITLE}" \
  --argjson tune_id "${CANARY_TUNE_ID}" \
  '{tune:{tune_ids:[$tune_id], name:"person", title:$title}}')"

CREATE_JSON="$(curl -fsS -X POST "${BASE_URL}/p/${CANARY_PACK_ID}/tunes" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}")"

echo "create_response:"
echo "${CREATE_JSON}" | jq '.'

POLL_TUNE_ID="$(echo "${CREATE_JSON}" | jq -r '(.id // .tune_id // .tune.id // empty) | tostring')"
if [[ -z "${POLL_TUNE_ID}" || "${POLL_TUNE_ID}" == "null" ]]; then
  POLL_TUNE_ID="${CANARY_TUNE_ID}"
fi
echo "poll_tune_id=${POLL_TUNE_ID}"

echo
echo "Polling prompts up to 60s..."
for i in {1..12}; do
  PROMPTS_JSON="$(curl -fsS "${BASE_URL}/tunes/${POLL_TUNE_ID}/prompts" \
    -H "Authorization: Bearer ${API_KEY}" || true)"
  if [[ -n "${PROMPTS_JSON}" ]]; then
    PROMPT_COUNT="$(echo "${PROMPTS_JSON}" | jq 'if type=="array" then length else 0 end')"
    IMAGE_COUNT="$(echo "${PROMPTS_JSON}" | jq '[.[]? | (.images // [])[]?] | length')"
    echo "attempt=${i} prompts=${PROMPT_COUNT} images=${IMAGE_COUNT}"
    if [[ "${PROMPT_COUNT}" -gt 0 ]]; then
      echo "Canary PASS: prompts created with name=person."
      exit 0
    fi
  else
    echo "attempt=${i} prompts=0 images=0"
  fi
  sleep 5
done

echo "Canary FAIL/TIMEOUT: no prompts observed in 60s."
exit 2
