#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v rg >/dev/null 2>&1; then
    echo "ERROR: ripgrep (rg) is required for the anonymity check." >&2
    exit 2
fi

COMMON_ARGS=(
    --line-number
    --hidden
    --glob '!.git/**'
    --glob '!**/scripts/check_anonymity.sh'
    --glob '!*.pyc'
)

PATH_OR_NETWORK_PATTERN='(/mnt/shared-storage-user/|/home/[^/]+/|ai4good|10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3})'
CREDENTIAL_PATTERN='(api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|password)[[:space:]]*[:=][[:space:]]*["'\''][^"'\'']+["'\'']'

failed=0

echo "Checking private paths and network addresses..."
if rg "${COMMON_ARGS[@]}" --regexp "${PATH_OR_NETWORK_PATTERN}" "${ROOT_DIR}"; then
    failed=1
fi

echo "Checking likely embedded credentials..."
if rg --ignore-case "${COMMON_ARGS[@]}" --regexp "${CREDENTIAL_PATTERN}" "${ROOT_DIR}"; then
    failed=1
fi

echo "Checking accidentally included artifacts..."
if find "${ROOT_DIR}" \
    \( -name '*.parquet' -o -name '*.safetensors' -o -name '*.ckpt' \
       -o -name '*.pt' -o -name '*.pth' -o -name '*.log' -o -name '.env' \) \
    -print -quit | grep -q .; then
    find "${ROOT_DIR}" \
        \( -name '*.parquet' -o -name '*.safetensors' -o -name '*.ckpt' \
           -o -name '*.pt' -o -name '*.pth' -o -name '*.log' -o -name '.env' \) \
        -print
    failed=1
fi

if [[ "${failed}" -ne 0 ]]; then
    echo "Anonymity check failed. Review the matches above." >&2
    exit 1
fi

echo "Anonymity check passed."
