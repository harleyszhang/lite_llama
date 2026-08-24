"""Record the backend-registry GIF: probing, selecting, and switching backends.

Drives the real :class:`~lite_llama.kernels.backends.BackendRegistry` and renders
one frame per registry decision, so the thing to look at is the probe column: a
backend the machine cannot run (fp8_native on sm86) is reported N/A and the next
candidate wins, and an env-var override moves the arrow without touching code.

Every line rendered comes from ``explain_selection()`` on this machine.

Usage:
    python scripts/gen_backend_registry_gif.py
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

W, H = 1180, 430
TITLE_H, PAD, LINE_H = 36, 18, 25
BG, TITLE_BG, TITLE_FG = (14, 16, 20), (32, 36, 44), (222, 226, 232)
PROMPT_FG, DIM, TEXT_FG = (118, 214, 118), (128, 136, 148), (222, 226, 232)
OK_FG, NA_FG, PICK_FG = (118, 214, 118), (245, 99, 72), (253, 188, 64)


@dataclass
class Scene:
    """One registry query: the command that produced it and its explain output."""

    command: str
    explain: str
    caption: str


def _explain_fresh(op: str, env: dict[str, str] | None = None) -> str:
    """Return explain_selection(op) from a clean registry, honouring *env*.

    The registry caches its choice per op, and the env override is read during
    selection, so a switch has to be evaluated against a fresh registry rather
    than the one an earlier scene already resolved.
    """
    import lite_llama.kernels.backends.registry as reg

    saved = {k: os.environ.get(k) for k in (env or {})}
    try:
        for key, value in (env or {}).items():
            os.environ[key] = value
        reg._REGISTRY = None  # force a rebuild so the probe order runs again
        return reg.explain_selection(op)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reg._REGISTRY = None


def record() -> list[Scene]:
    """Query the real registry for every scene the GIF shows."""
    import torch

    cc = (
        ".".join(str(x) for x in torch.cuda.get_device_capability())
        if torch.cuda.is_available()
        else "no CUDA"
    )
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    return [
        Scene(
            command="lite-llama explain --op linear",
            explain=_explain_fresh("linear"),
            caption=f"{device} (sm{cc}): fp8_native probes N/A, triton_quant wins on priority",
        ),
        Scene(
            command="lite-llama explain --op attention",
            explain=_explain_fresh("attention"),
            caption="attention picks the Triton FlashAttention-2 path over torch SDPA",
        ),
        Scene(
            command="LITE_LLAMA_LINEAR_BACKEND=torch_linear lite-llama explain --op linear",
            explain=_explain_fresh("linear", {"LITE_LLAMA_LINEAR_BACKEND": "torch_linear"}),
            caption="one env var pins the fallback backend - no code change",
        ),
        Scene(
            command="LITE_LLAMA_LINEAR_BACKEND=cutlass lite-llama explain --op linear",
            explain=_explain_fresh("linear", {"LITE_LLAMA_LINEAR_BACKEND": "cutlass"}),
            caption="an unknown backend does not crash: the registry warns and falls back",
        ),
    ]


def _line_colour(line: str) -> tuple[int, int, int]:
    if line.lstrip().startswith("->"):
        return PICK_FG
    if "N/A" in line:
        return NA_FG
    if "WARN" in line:
        return PICK_FG
    if " OK " in line:
        return OK_FG
    return TEXT_FG


def render(scene: Scene, reveal: int, fonts) -> Image.Image:
    """Render *scene* with only its first *reveal* explain lines shown."""
    body, bold, small = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    draw.text((12, 9), "lite-llama  —  kernel backend registry", fill=TITLE_FG, font=small)
    for index, colour in enumerate([(245, 99, 72), (253, 188, 64), (94, 193, 117)]):
        draw.ellipse([W - 78 + index * 18, 11, W - 68 + index * 18, 21], fill=colour)

    y = TITLE_H + PAD
    draw.text((PAD, y), f"$ {scene.command}", fill=PROMPT_FG, font=body)
    y += LINE_H + 10
    draw.line([PAD, y, W - PAD, y], fill=(52, 58, 68))
    y += 12

    lines = scene.explain.splitlines()
    for line in lines[:reveal]:
        font = bold if line.lstrip().startswith("->") else body
        draw.text((PAD, y), line, fill=_line_colour(line), font=font)
        y += LINE_H

    if reveal >= len(lines):
        draw.text((PAD, H - PAD - LINE_H), scene.caption, fill=DIM, font=small)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/images/backend_registry.gif")
    ap.add_argument("--duration", type=int, default=420, help="ms per frame")
    args = ap.parse_args()

    scenes = record()
    for scene in scenes:
        picked = scene.explain.splitlines()[-1].strip()
        print(f"{scene.command}\n    {picked}")

    fonts = (
        ImageFont.truetype(FONT_PATH, 17),
        ImageFont.truetype(BOLD_PATH, 17),
        ImageFont.truetype(FONT_PATH, 15),
    )

    images: list[Image.Image] = []
    for scene in scenes:
        total = len(scene.explain.splitlines())
        # Reveal the probe lines one at a time, then hold the resolved choice.
        for reveal in range(1, total + 1):
            images.append(render(scene, reveal, fonts))
        images += [render(scene, total, fonts)] * 4

    frames = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in images]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
        optimize=True,
    )
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
