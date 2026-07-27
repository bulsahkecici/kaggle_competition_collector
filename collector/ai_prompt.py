"""Generate a copy-paste prompt for analyzing a collected Kaggle package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROMPT_FILENAME = "COPY_PASTE_AI_PROMPT_TR.md"


def write_ai_analysis_prompt(
    output_dir: Path,
    slug: str,
    api_data: dict[str, Any] | None = None,
) -> Path:
    """Write a Turkish prompt that tells an AI exactly how to use the ZIP package."""
    api_data = api_data or {}
    metadata = api_data.get("metadata", {}) if isinstance(api_data, dict) else {}
    title = metadata.get("title") or slug

    quality_summary = _quality_summary(output_dir / "collection_quality_report.md")
    notebook_count = _directory_count(output_dir / "code_notebooks", "nb_*")
    discussion_count = _directory_count(output_dir / "discussions" / "threads", "thread_*.md")
    data_files = _data_file_names(output_dir / "data")

    prompt = f"""# Yapay zekâya gönderilecek hazır görev

Aşağıdaki metni, `{slug}` yarışması için oluşturulan ZIP dosyasını yüklediğin yapay zekâya **ayrı bir mesaj olarak** gönder.

---

Bu mesajı yalnızca onaylama ve benden ek talimat bekleme. Yüklediğim arşivi şimdi aç, içeriğini sistematik biçimde incele ve aşağıdaki görevi tamamla.

## Yarışma

- Başlık: **{title}**
- Kaggle slug: `{slug}`
- Arşiv kalite özeti: {quality_summary}
- Toplanan notebook klasörü sayısı: {notebook_count}
- Toplanan tartışma sayısı: {discussion_count}
- Veri dosyaları: {', '.join(data_files) if data_files else 'Arşivde veri dosyası bulunmayabilir'}

## Ana görev

Sen deneyimli bir Kaggle Grandmaster ve kıdemli veri bilimcisin. Amacın, arşivdeki yarışma belgelerini, veri profillerini, topluluk tartışmalarını, leaderboard bilgilerini ve public notebook kodlarını birlikte değerlendirerek bu yarışmada rekabetçi bir çözüm geliştirmek.

Arşivi yüzeysel biçimde özetleme. Dosyalardaki gerçek kanıtları karşılaştır, çelişkileri belirt ve önerilerini hangi dosya/notebook/tartışmadan çıkardığını açıkça yaz. Arşivde bulunmayan bilgi için tahmin yürütme; belirsizliği belirt.

## İnceleme sırası

Önce aşağıdaki dosyaları oku:

1. `SUMMARY_REPORT.md`
2. `collection_quality_report.md`
3. `METRIC_EXPLAINED.md`
4. `RULES_RISK_REPORT.md`
5. `SUBMISSION_FORMAT.md`
6. `data_profile/` altındaki tüm raporlar
7. `CODE_NOTEBOOKS_SUMMARY.md`
8. `discussions/discussion_insights.md`
9. `leaderboard/my_submission_trend.md` ve mevcutsa submission geçmişi
10. En güçlü, en farklı ve en çok kanıt sağlayan public notebook klasörlerindeki `notebook_code.py`, `notebook.md`, `metadata.json` ve `outputs_summary.md`
11. Gerekli gördüğün önemli `discussions/threads/` dosyaları

Bütün notebookları aynı ağırlıkta ele alma. Önce özetlerden yaklaşım kümeleri oluştur; ardından her kümeden temsilci ve güçlü notebookları ayrıntılı incele. Aynı kodun kopyalarını veya küçük varyasyonlarını tekrar tekrar analiz etme.

## Üretilecek çıktı

### 1. Yarışmanın teknik özeti

Hedef değişkeni, problem tipi, veri boyutu, sınıf dağılımı, eksik değerler, train-test farkları, submission formatı ve değerlendirme metriğini açıkla. Metrik için hangi doğrulama hatalarının yanıltıcı olacağını belirt.

### 2. Veri ve leakage denetimi

Kimlik sütunları, sentetik veri izleri, duplicate kayıtlar, train-test dağılım kayması, hedef sızıntısı, post-processing riski ve public leaderboard overfitting ihtimalini değerlendir. Her bulguyu kanıt düzeyine göre `güçlü`, `orta` veya `zayıf` olarak sınıflandır.

### 3. Notebook karşılaştırması

En az 10 anlamlı notebook veya yaklaşım ailesini karşılaştıran bir tablo oluştur. Şu sütunları kullan:

`Notebook/yaklaşım | Model | Feature engineering | CV yöntemi | OOF/LB skoru | Güçlü yanı | Zayıf yanı | Tekrar kullanılacak fikir`

Sadece başlıkları listeleme; gerçek kodu ve açıklamaları inceleyerek karşılaştır.

### 4. Tartışmalardan çıkarılan bilgiler

Toplulukta tekrar eden en önemli fikirleri, metric tuzaklarını, veri üretim hipotezlerini, leakage iddialarını, private leaderboard risklerini ve işe yaramadığı bildirilen yöntemleri özetle. Tartışma iddialarını doğrulanmış gerçek gibi sunma.

### 5. En güvenilir doğrulama stratejisi

Bu yarışma için uygulanabilir bir cross-validation planı oluştur. Fold yöntemi, seed sayısı, stratification/grouping gereksinimi, OOF üretimi, threshold veya decision rule optimizasyonu ve CV-LB korelasyon takibini ayrıntılandır.

### 6. Rekabetçi pipeline

Aşamalı bir çözüm tasarla:

- Baseline
- Güçlü tek model
- Farklı model aileleri
- Feature engineering
- Probability calibration veya sınıf önceliği düzeltmesi
- OOF tabanlı ensemble
- Post-processing
- Ablation testleri
- Seed stability
- Final submission seçimi

Her aşama için beklenen faydayı, hesaplama maliyetini ve başarısızlık riskini yaz.

### 7. Uygulanabilir deney planı

Öncelik sıralı deney tablosu oluştur:

`Deney ID | Değişiklik | Hipotez | CV ölçümü | Başarı kriteri | Maliyet | Sonraki karar`

İlk etapta en fazla 12 deney öner. Birbirini aynı anda değiştiren kontrolsüz deneylerden kaçın.

### 8. Kod üretimi

Arşivdeki en iyi fikirleri kopyala-yapıştır biçiminde birleştirme. Temiz ve yeniden üretilebilir bir proje tasarla. Aşağıdaki dosya yapısını öner ve ardından çalıştırılabilir başlangıç kodunu üret:

- `src/config.py`
- `src/data.py`
- `src/features.py`
- `src/models.py`
- `src/cv.py`
- `src/ensemble.py`
- `train.py`
- `predict.py`
- `requirements.txt`

Kodda sabit seed, loglama, OOF kayıtları, model artefactları, deney konfigürasyonu ve submission doğrulaması bulunmalı. Kod üretmeden önce arşivdeki gerçek kolon adlarını ve hedefi doğrula.

### 9. Son karar

En sonunda şunları açıkça yaz:

- Şu an uygulanacak en iyi ilk model
- En değerli üç farklılaştırıcı fikir
- Kaçınılması gereken üç yaklaşım
- Private leaderboard için en güvenli ensemble
- İlk çalıştırılacak komutlar

## Çalışma kuralları

- Arşivdeki kurallara aykırı yöntem önermeden önce `RULES_RISK_REPORT.md` içeriğini kontrol et.
- Public LB skorunu tek başarı ölçütü olarak kullanma.
- Leak veya leaderboard probing iddialarını otomatik olarak çözüm kabul etme.
- Ben istemeden Kaggle’a submission gönderme.
- Dosyaya erişemediğin veya okuyamadığın bir bölüm olursa bunu açıkça belirt; varmış gibi davranma.
- Yanıtı Türkçe yaz, teknik terimleri gerektiğinde İngilizce bırak.
- Görevi şimdi başlat; benden yeniden “incele” dememi bekleme.
"""

    path = output_dir / PROMPT_FILENAME
    path.write_text(prompt, encoding="utf-8")
    return path


def _quality_summary(path: Path) -> str:
    if not path.exists():
        return "Kalite raporu bulunamadı"
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "**Summary:**" in line:
            return line.replace("**Summary:**", "").strip()
    return "Kalite özeti ayrıştırılamadı"


def _directory_count(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.glob(pattern) if path.is_dir() or path.is_file())


def _data_file_names(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and not path.name.endswith(".sha256")
    )
