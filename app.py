"""Streamlit web interface for kaggle_competition_collector.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "competition_archive"
TOTAL_PHASES = 9
MAX_LOG_LINES = 500


def extract_competition_slug(value: str) -> str:
    """Return a Kaggle competition slug from a slug or competition URL."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Kaggle yarışma bağlantısını girin.")

    if "kaggle.com/" in cleaned.lower() and not cleaned.lower().startswith(("http://", "https://")):
        cleaned = f"https://{cleaned}"

    if cleaned.lower().startswith(("http://", "https://")):
        parsed = urlparse(cleaned)
        host = parsed.netloc.lower().split(":", 1)[0]
        if host not in {"kaggle.com", "www.kaggle.com"}:
            raise ValueError("Bağlantı kaggle.com alan adına ait olmalı.")

        parts = [part for part in parsed.path.split("/") if part]
        try:
            competition_index = parts.index("competitions")
            slug = parts[competition_index + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                "Bağlantı şu biçimde olmalı: https://www.kaggle.com/competitions/yarisma-adi"
            ) from exc
    else:
        slug = cleaned

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", slug):
        raise ValueError("Yarışma adı yalnızca harf, rakam, tire ve alt çizgi içerebilir.")

    return slug


def resolve_output_root(raw_value: str) -> Path:
    """Resolve an optional user-supplied output directory."""
    if not raw_value.strip():
        return DEFAULT_OUTPUT_ROOT.resolve()

    expanded = os.path.expandvars(raw_value.strip())
    output_root = Path(expanded).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return output_root.resolve()


def build_command(
    *,
    slug: str,
    output_root: Path,
    download_data: bool,
    collect_browser: bool,
    collect_screenshots: bool,
    collect_notebooks: bool,
    collect_discussions: bool,
    profile_data: bool,
    headed_browser: bool,
    pause_before_tabs: bool,
    max_notebooks: int,
    max_discussions: int,
) -> list[str]:
    """Build the existing CLI command from UI options."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--competition",
        slug,
        "--output",
        str(output_root),
        "--max-notebooks",
        str(max_notebooks),
        "--max-discussions",
        str(max_discussions),
    ]

    if not download_data:
        command.append("--no-download")
    if not collect_browser:
        command.append("--no-browser")
    if not collect_screenshots:
        command.append("--no-screenshots")
    if not collect_notebooks:
        command.append("--no-notebooks")
    if not collect_discussions:
        command.append("--no-discussions")
    if not profile_data:
        command.append("--no-profile")

    if collect_browser:
        command.append("--headed" if headed_browser else "--headless")
        if headed_browser and pause_before_tabs:
            command.append("--pause-before-tabs")

    return command


def command_as_text(command: list[str]) -> str:
    if platform.system() == "Windows":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def run_collector(command: list[str], log_box: Any, progress_bar: Any, phase_box: Any) -> tuple[int, str]:
    """Run main.py and stream combined stdout/stderr into the Streamlit page."""
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )

    collected_lines: list[str] = []
    assert process.stdout is not None

    for raw_line in iter(process.stdout.readline, ""):
        line = raw_line.rstrip("\r\n")
        collected_lines.append(line)

        phase_match = re.search(r"Phase\s+(\d+)/(\d+):\s*(.+)", line)
        if phase_match:
            current_phase = min(int(phase_match.group(1)), TOTAL_PHASES)
            phase_box.info(f"Aşama {current_phase}/{TOTAL_PHASES}: {phase_match.group(3)}")
            progress_bar.progress(current_phase / TOTAL_PHASES)

        visible_lines = collected_lines[-MAX_LOG_LINES:]
        log_box.code("\n".join(visible_lines), language="text")

    return_code = process.wait()
    if return_code == 0:
        progress_bar.progress(1.0)

    return return_code, "\n".join(collected_lines)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def find_latest_archive(output_root: Path, slug: str) -> Path | None:
    archive_dir = output_root / slug / "archives"
    candidates = list(archive_dir.glob(f"{slug}_*.zip")) if archive_dir.exists() else []
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def parse_quality_counts(report: str) -> tuple[int | None, int | None, int | None]:
    match = re.search(
        r"Summary:\*\*\s*✅\s*(\d+)\s*OK\s*\|\s*⚠️\s*(\d+)\s*WARNING\s*\|\s*❌\s*(\d+)\s*FAILED",
        report,
    )
    if not match:
        return None, None, None
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def open_local_folder(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    system = platform.system()
    if system == "Windows":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif system == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def render_results(result: dict[str, Any]) -> None:
    output_root = Path(result["output_root"])
    slug = result["slug"]
    latest_dir = output_root / slug / "latest"
    quality_path = latest_dir / "collection_quality_report.md"
    summary_path = latest_dir / "SUMMARY_REPORT.md"
    handoff_path = latest_dir / "AI_HANDOFF_INSTRUCTIONS.md"
    quality_report = read_text(quality_path)
    ok_count, warning_count, failed_count = parse_quality_counts(quality_report)

    st.divider()
    st.subheader("Son çalıştırma")
    st.caption(f"Tamamlanma zamanı: {result['completed_at']}")

    if result["return_code"] != 0:
        st.error(f"Toplama işlemi hata kodu {result['return_code']} ile sona erdi.")
    elif failed_count and failed_count > 0:
        st.warning(f"İşlem tamamlandı ancak kalite raporunda {failed_count} başarısız kontrol var.")
    else:
        st.success("Yarışma bilgileri başarıyla toplandı.")

    metric_columns = st.columns(3)
    metric_columns[0].metric("Başarılı kontrol", "—" if ok_count is None else ok_count)
    metric_columns[1].metric("Uyarı", "—" if warning_count is None else warning_count)
    metric_columns[2].metric("Başarısız", "—" if failed_count is None else failed_count)

    st.markdown("**Çıktı klasörü**")
    st.code(str(latest_dir), language="text")

    button_columns = st.columns(3)
    if button_columns[0].button("Çıktı klasörünü aç", key=f"open-{slug}-{result['completed_at']}"):
        try:
            open_local_folder(latest_dir)
        except Exception as exc:  # UI boundary: show actionable error
            st.error(f"Klasör açılamadı: {exc}")

    if summary_path.exists():
        button_columns[1].download_button(
            "Özet raporu indir",
            data=summary_path.read_bytes(),
            file_name=summary_path.name,
            mime="text/markdown",
            key=f"summary-{slug}-{result['completed_at']}",
        )

    archive_path = find_latest_archive(output_root, slug)
    if archive_path is not None:
        button_columns[2].download_button(
            "Arşiv ZIP indir",
            data=archive_path.read_bytes(),
            file_name=archive_path.name,
            mime="application/zip",
            key=f"archive-{slug}-{result['completed_at']}",
        )

    with st.expander("Kalite raporunu göster", expanded=True):
        st.markdown(quality_report or "Kalite raporu bulunamadı.")

    with st.expander("SUMMARY_REPORT.md önizlemesi"):
        summary_text = read_text(summary_path)
        st.markdown(summary_text if summary_text else "Özet rapor bulunamadı.")

    with st.expander("AI handoff talimatları"):
        handoff_text = read_text(handoff_path)
        st.markdown(handoff_text if handoff_text else "AI handoff dosyası bulunamadı.")

    with st.expander("Çalıştırılan komut ve loglar"):
        st.code(result["command"], language="powershell")
        st.code(result["logs"], language="text")


st.set_page_config(
    page_title="Kaggle Competition Collector",
    page_icon="🏁",
    layout="wide",
)

st.title("🏁 Kaggle Competition Collector")
st.write(
    "Kaggle yarışma bağlantısını yapıştırın. Uygulama yarışma sayfalarını, verileri, "
    "notebook'ları ve tartışmaları toplayarak analiz için hazır bir klasör oluşturur."
)

with st.sidebar:
    st.header("Toplama seçenekleri")
    download_data = st.checkbox("Veri dosyalarını indir", value=True)
    collect_browser = st.checkbox("Yarışma sayfalarını tara", value=True)
    collect_screenshots = st.checkbox("Ekran görüntülerini kaydet", value=True)
    collect_notebooks = st.checkbox("Notebook'ları topla", value=True)
    collect_discussions = st.checkbox("Tartışmaları topla", value=True)
    profile_data = st.checkbox("Veri profillemesi yap", value=True)

    st.divider()
    max_notebooks = int(
        st.number_input("Maksimum notebook", min_value=0, max_value=100, value=15, step=1)
    )
    max_discussions = int(
        st.number_input("Maksimum tartışma", min_value=0, max_value=100, value=25, step=1)
    )

    st.divider()
    headed_browser = st.checkbox(
        "Tarayıcı penceresini göster",
        value=False,
        help="Giriş yapılması veya yarışma kurallarının kabul edilmesi gerekiyorsa açın.",
    )
    pause_before_tabs = st.checkbox(
        "Sekmeleri toplamadan önce bekle",
        value=False,
        help="Yalnızca tarayıcı penceresi gösterilirken kullanılır.",
    )

competition_url = st.text_input(
    "Kaggle yarışma bağlantısı",
    placeholder="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction",
)
output_directory = st.text_input(
    "Çıktı ana klasörü (isteğe bağlı)",
    placeholder=str(DEFAULT_OUTPUT_ROOT),
    help="Boş bırakırsanız proje içindeki competition_archive klasörü kullanılır.",
)

start_clicked = st.button("Yarışma bilgilerini çek", type="primary", use_container_width=True)

if start_clicked:
    try:
        competition_slug = extract_competition_slug(competition_url)
        output_root = resolve_output_root(output_directory)
        output_root.mkdir(parents=True, exist_ok=True)

        command = build_command(
            slug=competition_slug,
            output_root=output_root,
            download_data=download_data,
            collect_browser=collect_browser,
            collect_screenshots=collect_screenshots,
            collect_notebooks=collect_notebooks,
            collect_discussions=collect_discussions,
            profile_data=profile_data,
            headed_browser=headed_browser,
            pause_before_tabs=pause_before_tabs,
            max_notebooks=max_notebooks,
            max_discussions=max_discussions,
        )

        st.subheader("Toplama işlemi")
        phase_box = st.empty()
        progress_bar = st.progress(0.0)
        log_box = st.empty()
        phase_box.info("Program başlatılıyor…")

        return_code, logs = run_collector(command, log_box, progress_bar, phase_box)
        completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.session_state["last_run"] = {
            "slug": competition_slug,
            "output_root": str(output_root),
            "return_code": return_code,
            "logs": logs,
            "command": command_as_text(command),
            "completed_at": completed_at,
        }
        phase_box.success("Program tamamlandı.") if return_code == 0 else phase_box.error(
            "Program hata ile tamamlandı."
        )
    except ValueError as exc:
        st.error(str(exc))
    except FileNotFoundError as exc:
        st.error(f"Program dosyası bulunamadı: {exc}")
    except Exception as exc:  # UI boundary: keep the page alive and show the error
        st.exception(exc)

if "last_run" in st.session_state:
    render_results(st.session_state["last_run"])
