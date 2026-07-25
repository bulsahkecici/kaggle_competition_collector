# Web arayüzü

Streamlit arayüzü, mevcut `main.py` toplama motorunu değiştirmeden tarayıcı üzerinden kullanmanızı sağlar.

## Kurulum

Proje klasöründe sanal ortam aktifken bağımlılıkları güncelleyin:

```powershell
pip install -r requirements.txt
python -m playwright install chromium
```

Kaggle API oturumu daha önce açılmadıysa:

```powershell
kaggle auth login
```

## Arayüzü başlatma

Windows'ta proje klasöründeki aşağıdaki dosyaya çift tıklayın:

```text
start_ui.bat
```

Alternatif olarak PowerShell'den:

```powershell
streamlit run app.py
```

Uygulama varsayılan olarak şu adreste açılır:

```text
http://localhost:8501
```

## Kullanım

1. Kaggle yarışmasının bağlantısını giriş alanına yapıştırın.
2. Sol menüden veri indirme, notebook, tartışma ve ekran görüntüsü seçeneklerini belirleyin.
3. Gerekirse **Tarayıcı penceresini göster** seçeneğini açın. Bu seçenek, Kaggle girişi veya yarışma kurallarının kabulü gerektiğinde kullanışlıdır.
4. **Yarışma bilgilerini çek** düğmesine basın.
5. Çalışma loglarını ve aşama ilerlemesini sayfadan takip edin.
6. İşlem bittiğinde özet raporu veya arşiv ZIP dosyasını arayüzden indirin.

## Çıktılar

Varsayılan çıktı klasörü:

```text
competition_archive/<competition-slug>/latest/
```

Arayüz aşağıdaki sonuçları gösterir:

- Kalite kontrolü sayıları
- `collection_quality_report.md`
- `SUMMARY_REPORT.md` önizlemesi
- `AI_HANDOFF_INSTRUCTIONS.md`
- Çıktı klasörünü açma düğmesi
- Özet raporu indirme düğmesi
- En son arşiv ZIP dosyasını indirme düğmesi

## Önemli not

Streamlit arayüzünde **Tarayıcı penceresini göster** ve **Sekmeleri toplamadan önce bekle** birlikte seçilirse, `main.py` terminalden Enter bekleyebilir. Normal otomatik kullanım için tarayıcıyı headless bırakmanız önerilir. Kuralları kabul etmeniz gereken bir yarışmada önce tarayıcı penceresini göstererek giriş yapın, ardından otomatik çalıştırmayı tekrar başlatın.
