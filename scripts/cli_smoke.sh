#!/usr/bin/env bash
# End-to-end CLI smoke test.
#
# Runs the ``lite-llama`` CLI against every checkpoint present under my_weight/
# and asserts each one emits some text and exits cleanly. This is what should
# have caught the CUDA-driver-vs-torch mismatch and the earlier CUDA Graph bug.
#
# Usage:
#   scripts/cli_smoke.sh                       # exercise every converted model
#   PYTHON=... scripts/cli_smoke.sh <name>...  # only the named checkpoints
#
# The script auto-discovers a compatible Python. If nothing works, it prints a
# diagnostic and exits non-zero.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MY_WEIGHT="${REPO_ROOT}/my_weight"

if [[ ! -d "${MY_WEIGHT}" ]]; then
    echo "no checkpoints under ${MY_WEIGHT}; run lite-llama-convert first" >&2
    exit 1
fi

# ---------------------------------------------------------------------------- #
# Resolve a Python whose torch.cuda actually works. Users occasionally end up
# with a torch build that targets a newer CUDA than their driver supports; when
# that happens, torch.cuda.is_available() returns False and every subsequent
# ``torch.load(..., map_location='cuda')`` call fails deep inside pickle.
# ---------------------------------------------------------------------------- #
_pick_python() {
    local candidates=()
    if [[ -n "${PYTHON:-}" ]]; then candidates+=("${PYTHON}"); fi
    candidates+=("${REPO_ROOT}/.venv/bin/python")
    candidates+=("${REPO_ROOT}/../vllm/.venv/bin/python")
    candidates+=("python3")
    for candidate in "${candidates[@]}"; do
        if [[ ! -x "$(command -v "${candidate}" 2>/dev/null)" ]] && [[ ! -x "${candidate}" ]]; then
            continue
        fi
        if "${candidate}" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
                2>/dev/null; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

if ! PY="$(_pick_python)"; then
    echo "ERROR: no Python interpreter with working torch.cuda found." >&2
    echo "       Install a torch build matching your NVIDIA driver, e.g." >&2
    echo "       uv pip install torch --index-url https://download.pytorch.org/whl/cu124" >&2
    exit 1
fi

TORCH_VERSION=$("${PY}" -c 'import torch; print(torch.__version__)')
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo unknown)
echo "using python=${PY}"
echo "  torch=${TORCH_VERSION}  driver=${DRIVER_VERSION}"

# ---------------------------------------------------------------------------- #
# Helpers: dispatch text (``chat``) vs multimodal (``vl-chat``) per checkpoint.
# ---------------------------------------------------------------------------- #
_is_multimodal() {
    local model_dir="$1"
    "${PY}" - "${model_dir}" <<'PY' 2>/dev/null
import json, sys
cfg = json.load(open(sys.argv[1] + "/config.json"))
sys.exit(0 if cfg.get("model_type") in ("llava", "qwen3_vl") else 1)
PY
}

_is_qwen3_vl() {
    local model_dir="$1"
    "${PY}" - "${model_dir}" <<'PY' 2>/dev/null
import json, sys
cfg = json.load(open(sys.argv[1] + "/config.json"))
sys.exit(0 if cfg.get("model_type") == "qwen3_vl" else 1)
PY
}

_find_test_image() {
    for candidate in \
        "${REPO_ROOT}/images/llava_test/dog.jpeg" \
        "${REPO_ROOT}/images/llava_test/panda.jpg" \
        "${REPO_ROOT}/images/llava_test/dog2.png"; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

_smoke_one() {
    local name="$1"
    local model_dir="${MY_WEIGHT}/${name}"
    if [[ ! -f "${model_dir}/config.json" ]] || ! ls "${model_dir}"/*.pth >/dev/null 2>&1; then
        echo "  [skip] ${name}: no config.json or *.pth"
        return
    fi

    local log err
    log="$(mktemp)"
    err="$(mktemp)"
    local ok=1
    if _is_multimodal "${model_dir}"; then
        local img
        if ! img="$(_find_test_image)"; then
            echo "  [skip] ${name}: no test image under images/llava_test/"
            rm -f "${log}" "${err}"
            return
        fi
        echo "  [vl  ] ${name}"
        # stdout carries generated text; the logger and warnings go to stderr.
        # LLaVA wants the vicuna-style string with a <image> marker; Qwen3-VL
        # takes a plain user message (the chat template is applied downstream).
        local vl_prompt='USER: <image>\nDescribe the animal in one sentence. ASSISTANT:'
        if _is_qwen3_vl "${model_dir}"; then
            vl_prompt='Describe the animal in one sentence.'
        fi
        if ! "${PY}" -m lite_llama.cli vl-chat \
                --model-dir "${model_dir}" --image "${img}" \
                --prompt "${vl_prompt}" \
                --temperature 0.0 --top-p 1.0 --max-gen-len 16 --max-seq-len 2048 \
                >"${log}" 2>"${err}"; then
            ok=0
        fi
    else
        echo "  [chat] ${name}"
        if ! LITE_LLAMA_MODEL_DIR="${model_dir}" \
            "${PY}" -m lite_llama.cli chat \
                --temperature 0.0 --top-p 1.0 --max-gen-len 16 --max-seq-len 512 \
                <<< $'The capital of France is\nexit\n' \
                >"${log}" 2>"${err}"; then
            ok=0
        fi
    fi

    if [[ ${ok} -ne 1 ]]; then
        echo "    FAIL: CLI crashed for ${name}"
        sed 's/^/      /' "${err}" >&2
        sed 's/^/      /' "${log}" >&2
        rm -f "${log}" "${err}"
        return 1
    fi

    # stdout only holds the CLI banner, the ">>> " prompts and the streamed
    # tokens, so drop the two known banner lines and keep the rest.
    local generated
    generated="$(
        grep -Ev '^Loaded .* Type .exit. to quit\.$' "${log}" \
            | tr -d '\r' \
            | tr -s '[:space:]' ' ' \
            | sed 's/^ *//; s/ *$//' \
            | sed 's/^>>> *//'
    )"
    if [[ -z "${generated}" ]]; then
        echo "    FAIL: no generated text captured for ${name}"
        echo "      --- stdout ---" >&2
        cat "${log}" >&2
        echo "      --- stderr ---" >&2
        tail -20 "${err}" >&2
        rm -f "${log}" "${err}"
        return 1
    fi
    # Guard against the CUDA Graph stale-pointer class of bug, which produced
    # replacement chars rather than readable text.
    if [[ "${generated}" == *$'\ufffd'* ]]; then
        echo "    FAIL: output contains replacement characters (garbled) for ${name}"
        echo "      ${generated}" >&2
        rm -f "${log}" "${err}"
        return 1
    fi
    echo "    ok: ${generated:0:90}"
    rm -f "${log}" "${err}"
}

names=("$@")
if [[ ${#names[@]} -eq 0 ]]; then
    mapfile -t names < <(find "${MY_WEIGHT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi

failed=0
for name in "${names[@]}"; do
    if ! _smoke_one "${name}"; then
        failed=$((failed + 1))
    fi
done

if [[ ${failed} -gt 0 ]]; then
    echo "FAIL: ${failed} checkpoint(s) failed the CLI smoke test" >&2
    exit 1
fi
echo "OK: every checkpoint passed the CLI smoke test"
