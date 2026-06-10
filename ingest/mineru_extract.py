"""Optional MinerU PDF → Markdown (CLI). Falls back to MarkItDown when unavailable."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ingest.chunker import normalize_md_text

logger = logging.getLogger(__name__)

MINERU_CMD = os.getenv("MINERU_CMD", "mineru")
MINERU_TIMEOUT = int(os.getenv("MINERU_TIMEOUT", "900"))
# pipeline = CPU 可跑；勿用 vlm-*（需大 GPU / sglang）
MINERU_BACKEND = os.getenv("MINERU_BACKEND", "pipeline")
MINERU_LANG = os.getenv("MINERU_LANG", "ch")
# Mac 預設 mps 易 segfault；Linux 未設則交給 MinerU 自動選裝置
MINERU_DEVICE = os.getenv("MINERU_DEVICE", "cpu" if sys.platform == "darwin" else "")

MINERU_SOURCE_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".webp"})
MODEL_DOWNLOAD_HINT = "uv run mineru-models-download -m pipeline -s huggingface"


def mineru_available() -> bool:
    """CLI 在 PATH 且能執行 --version（需 mineru[pipeline] 依賴完整）。"""
    cmd = shutil.which(MINERU_CMD)
    if not cmd:
        return False
    try:
        proc = subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _pick_markdown(tmp_root: Path, stem: str) -> Path:
    candidates = list(tmp_root.rglob("*.md"))
    if not candidates:
        raise FileNotFoundError("MinerU 輸出目錄內沒有 .md")
    stem_lower = stem.lower()
    ranked = sorted(
        candidates,
        key=lambda p: (
            stem_lower in p.stem.lower(),
            p.stat().st_size,
        ),
        reverse=True,
    )
    return ranked[0]


def convert_with_mineru(source_path: str | Path, output_path: str | Path) -> str:
    """
    Run MinerU CLI and write normalized MD to output_path.

    預設 ``--backend pipeline``；Mac 預設 ``-d cpu``。首次使用需下載模型：
    ``uv run mineru-models-download -m pipeline -s huggingface``
    """
    src = Path(source_path).resolve()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not mineru_available():
        raise FileNotFoundError(
            f"MinerU 無法執行（`{MINERU_CMD} --version` 失敗）。"
            f"請執行：uv sync --group mineru && {MODEL_DOWNLOAD_HINT}"
        )

    device_note = f", device={MINERU_DEVICE}" if MINERU_DEVICE else ""
    print(f"🔄 MinerU 解析: {src}（backend={MINERU_BACKEND}, lang={MINERU_LANG}{device_note}）")
    with tempfile.TemporaryDirectory(prefix="mineru_") as tmp:
        tmp_path = Path(tmp)
        cmd = [
            MINERU_CMD,
            "-p",
            str(src),
            "-o",
            str(tmp_path),
            "-b",
            MINERU_BACKEND,
            "-l",
            MINERU_LANG,
        ]
        if MINERU_DEVICE:
            cmd.extend(["-d", MINERU_DEVICE])
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                timeout=MINERU_TIMEOUT,
                capture_output=True,
                text=True,
            )
            if proc.stderr:
                logger.debug("mineru stderr: %s", proc.stderr[:500])
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"MinerU 逾時（>{MINERU_TIMEOUT}s）") from exc

        if not list(tmp_path.rglob("*.md")):
            # MinerU CLI 內部吞 exception 仍可能 exit 0
            combined = f"{proc.stderr or ''}\n{proc.stdout or ''}".strip()
            detail = combined[-2000:] if len(combined) > 2000 else combined
            hint = ""
            if not Path.home().joinpath("mineru.json").is_file():
                hint = f" 請先執行：{MODEL_DOWNLOAD_HINT}"
            elif "cache_position" in combined:
                hint = " 可能為 transformers 與 MinerU 版本不相容，請 uv sync --group mineru 升級 MinerU>=2.1.9"
            raise RuntimeError(
                f"MinerU 未產生 .md（exit={proc.returncode}）: {detail}{hint}"
            )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:400]
            logger.warning("MinerU exit=%s 但有 .md 輸出: %s", proc.returncode, detail)

        md_src = _pick_markdown(tmp_path, src.stem)
        text = normalize_md_text(md_src.read_text(encoding="utf-8"))
        out.write_text(text, encoding="utf-8")

    print(f"✅ MinerU → {out}（{len(text)} 字）")
    return str(out)
