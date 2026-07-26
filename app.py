"""Streamlit web interface for kaggle_competition_collector."""

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

from collector.data_size_inspector import DatasetInspectionError, format_bytes, inspect_competition_data


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "competition_archive"
TOTAL_PHASES = 9
MAX_LOG_LINES = 500
DEFAULT_LARGE_DATA_GB = 5.0


def extract_competition_slug(value: str) -> str:
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
            index = parts.index("competitions")
            slug = parts[index + 1]
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
    max_notebooks: int,
    max_discussions: int,
) -> list[str]:
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
    return command


def command_as_text(command: list[str]) -> str:
    if platform.system() == "Windows":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def run_collector(command: list[str], log_box: Any, progress_bar: Any, phase_box: Any) -> tuple[int, str]:
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
    lines: list[str] = []
    assert process.stdout is not None
    for raw_line in iter(process.stdout.readline, ""):
        line = raw_line.rstrip("\r\n")
        lines.append(line)
        match = re.search(r"Phase\s+(\d+)/(\d+):\s*(.+)", line)
        if match:
            current = min(int(match.group(1)), TOTAL_PHASES)
            phase_box.info(f"Aşama {current}/{TOTAL_PHASES}: {match.group(3)}")
            progress_bar.progress(current / TOTAL_PHASES)
        log_box.code("\n".join(lines[-MAX_LOG_LINES:]), language="text")
    return_code = process.wait()
    if return_code == 0:
        progress_bar.progress(1.0)
    return return_code, "\n".join(lines)


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
    if platform.system() == "Windows":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def render_inspection(inspection: dict[str, Any], threshold_bytes: int) -> bool:
    total_bytes = int(inspection["total_bytes"])
    is_large = total_bytes >= threshold_bytes
    columns = st.columns(3)
    columns[0].metric("Toplam veri boyutu", inspection["total_display"])
    columns[1].metric("Dosya sayısı", inspection["file_count"])
    largest = inspection.get("largest_file") or {}
    columns[2].metric("En büyük dosya", largest.get("size_display", "—"))

    if largest:
        st.caption(f"En büyük dosya: `{largest.get('name', '')}`")

    with st.expander("Veri dosyalarını göster"):
        st.dataframe(
            [
                {"Dosya": item["name"], "Boyut": item["size_display"]}
                for item in inspection["files"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    if is_large:
        st.error(
            f"Bu yarışmanın verisi yaklaşık **{inspection['total_display']}**. "
            f"Belirlenen büyük veri sınırı **{format_bytes(threshold_bytes)}**."
        )
    else:
        st.success(
            f"Veri boyutu **{inspection['total_display']}**. "
            f"Büyük veri sınırı olan {format_bytes(threshold_bytes)} değerinin altında."
        )
    return is_large


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

    if "--no-download" in result["command"]:
        st.info("Bu çalıştırmada veri indirme kapalıydı.")
    if "cloudflare page" in result["logs"].lower():
        st.error("Kaggle/Cloudflare engeli görüldü. Görünür tarayıcı moduyla yeniden çalıştırın.")

    columns = st.columns(3)
    columns[0].metric("Başarılı kontrol", "—" if ok_count is None else ok_count)
    columns[1].metric("Uyarı", "—" if warning_count is None else warning_count)
    columns[2].metric("Başarısız", "—" if failed_count is None else failed_count)

    st.code(str(latest_dir), language="text")
    buttons = st.columns(3)
    if buttons[0].button("Çıktı klasörünü aç", key=f"open-{result['completed_at']}"):
        try:
            open_local_folder(latest_dir)
        except Exception as exc:
            st.error(f"Klasör açılamadı: {exc}")
    if summary_path.exists():
        buttons[1].download_button(
            "Özet raporu indir",
            data=summary_path.read_bytes(),
            file_name=summary_path.name,
            mime="text/markdown",
            key=f"summary-{result['completed_at']}",
        )
    archive_path = find_latest_archive(output_root, slug)
    if archive_path:
        buttons[2].download_button(
            "Arşiv ZIP indir",
            data=archive_path.read_bytes(),
            file_name=archive_path.name,
            mime="application/zip",
            key=f"archive-{result['completed_at']}",
        )

    with st.expander("Kalite raporunu göster", expanded=True):
        st.markdown(quality_report or "Kalite raporu bulunamadı.")
    with st.expander("SUMMARY_REPORT.md önizlemesi"):
        st.markdown(read_text(summary_path) or "Özet rapor bulunamadı.")
    with st.expander("AI handoff talimatları"):
        st.markdown(read_text(handoff_path) or "AI handoff dosyası bulunamadı.")
    with st.expander("Çalıştırılan komut ve loglar"):
        st.code(result["command"], language="powershell")
        st.code(result["logs"], language="text")


st.set_page_config(page_title="Kaggle Competition Collector", page_icon="🏁", layout="wide")
st.title("🏁 Kaggle Competition Collector")
st.write(
    "Kaggle yarışma bağlantısını yapıştırın. Önce veri boyutunu kontrol edin, ardından toplama işlemini başlatın."
)

with st.sidebar:
    st.header("Toplama seçenekleri")
    download_data = st.checkbox("Veri dosyalarını indir", value=True, key="download_data_v4")
    collect_browser = st.checkbox("Yarışma sayfalarını tara", value=True, key="browser_v4")
    collect_screenshots = st.checkbox("Ekran görüntülerini kaydet", value=True, key="screens_v4")
    collect_notebooks = st.checkbox("Notebook'ları topla", value=True, key="notebooks_v4")
    collect_discussions = st.checkbox("Tartışmaları topla", value=True, key="discussions_v4")
    profile_data = st.checkbox("Veri profillemesi yap", value=True, key="profile_v4")
    max_notebooks = int(st.number_input("Maksimum notebook", 0, 100, 15, 1))
    max_discussions = int(st.number_input("Maksimum tartışma", 0, 100, 25, 1))
    headed_browser = st.checkbox("Tarayıcı penceresini göster", value=True, key="headed_v4")
    large_data_gb = float(
        st.number_input(
            "Büyük veri uyarı sınırı (GB)",
            min_value=0.1,
            max_value=1000.0,
            value=DEFAULT_LARGE_DATA_GB,
            step=0.5,
        )
    )

competition_url = st.text_input(
    "Kaggle yarışma bağlantısı",
    placeholder="https://www.kaggle.com/competitions/playground-series-s6e7",
)
output_directory = st.text_input(
    "Çıktı ana klasörü (isteğe bağlı)",
    placeholder=str(DEFAULT_OUTPUT_ROOT),
)

try:
    current_slug = extract_competition_slug(competition_url) if competition_url.strip() else ""
except ValueError as exc:
    current_slug = ""
    st.error(str(exc))

if st.session_state.get("inspection_slug") != current_slug:
    st.session_state.pop("dataset_inspection", None)
    st.session_state.pop("large_download_confirmed", None)
    st.session_state["inspection_slug"] = current_slug

check_clicked = st.button(
    "Veri boyutunu kontrol et",
    use_container_width=True,
    disabled=not bool(current_slug),
)

if check_clicked:
    try:
        with st.spinner("Kaggle dosya listesi kontrol ediliyor…"):
            st.session_state["dataset_inspection"] = inspect_competition_data(current_slug)
            st.session_state["large_download_confirmed"] = False
    except DatasetInspectionError as exc:
        st.error(str(exc))

inspection = st.session_state.get("dataset_inspection")
threshold_bytes = int(large_data_gb * 1024**3)
is_large = False
if inspection and inspection.get("slug") == current_slug:
    is_large = render_inspection(inspection, threshold_bytes)

large_download_confirmed = False
if download_data and is_large:
    large_download_confirmed = st.checkbox(
        f"{inspection['total_display']} veriyi indirmeyi onaylıyorum",
        value=False,
        key="large_download_confirmed",
    )

if not download_data:
    st.info("Veri indirme kapalı; boyut kontrolü yapılabilir fakat data klasörü indirilmeyecek.")

inspection_ready = bool(inspection and inspection.get("slug") == current_slug)
can_start = bool(current_slug) and inspection_ready
if download_data and is_large:
    can_start = can_start and large_download_confirmed

start_clicked = st.button(
    "Yarışma bilgilerini çek",
    type="primary",
    use_container_width=True,
    disabled=not can_start,
)

if current_slug and not inspection_ready:
    st.warning("Toplama işleminden önce **Veri boyutunu kontrol et** düğmesine basın.")
elif download_data and is_large and not large_download_confirmed:
    st.warning("Büyük veri indirmesini başlatmak için onay kutusunu işaretleyin.")

if start_clicked:
    try:
        output_root = resolve_output_root(output_directory)
        output_root.mkdir(parents=True, exist_ok=True)
        command = build_command(
            slug=current_slug,
            output_root=output_root,
            download_data=download_data,
            collect_browser=collect_browser,
            collect_screenshots=collect_screenshots,
            collect_notebooks=collect_notebooks,
            collect_discussions=collect_discussions,
            profile_data=profile_data,
            headed_browser=headed_browser,
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
            "slug": current_slug,
            "output_root": str(output_root),
            "return_code": return_code,
            "logs": logs,
            "command": command_as_text(command),
            "completed_at": completed_at,
        }
        if return_code == 0:
            phase_box.success("Program tamamlandı.")
        else:
            phase_box.error("Program hata ile tamamlandı.")
    except Exception as exc:
        st.exception(exc)

if "last_run" in st.session_state:
    render_results(st.session_state["last_run"])
