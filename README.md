# kaggle_competition_collector

A local Python tool that collects and organises **everything visible to your Kaggle account** on a competition page, then exports it into a structured folder ready to hand off to an AI assistant for deep competition analysis.

---

## What it collects

| Source | What |
|--------|------|
| **Kaggle API** | Competition metadata (title, deadline, evaluation metric, reward, team count, tags, …) |
| **Kaggle API** | List of data files (names, sizes) |
| **Kaggle API / CLI** | Competition data files (downloaded & unzipped) |
| **Browser (Playwright)** | Full page text of: Overview, Data description, Evaluation, Rules, Leaderboard, Discussion |
| **Browser** | Full-page PNG screenshots of every tab |
| **Browser** | Visible discussion thread titles |

---

## Output structure

```
competition_archive/
└── <competition-slug>/
    ├── latest/                       ← current run for this competition only
    │   ├── SUMMARY_REPORT.md         ← hand this to your AI assistant
    │   ├── AI_HANDOFF_INSTRUCTIONS.md
    │   ├── collection_quality_report.md
    │   ├── run_metadata.json
    │   ├── competition_metadata.json
    │   ├── files_manifest.json
    │   ├── collection_log.txt
    │   ├── pages/
    │   ├── screenshots/
    │   ├── data/
    │   ├── code_notebooks/
    │   ├── discussions/
    │   ├── leaderboard/
    │   └── data_profile/
    └── archives/
        ├── <competition-slug>_YYYYMMDD_HHMMSS/
        └── <competition-slug>_YYYYMMDD_HHMMSS.zip
```

Each competition gets its own folder, so runs for different competitions do not
share a single `latest/` directory.

---

## Requirements

- Python 3.10+
- A Kaggle account
- `~/.kaggle/kaggle.json` configured (see [Kaggle API docs](https://www.kaggle.com/docs/api))

---

## Installation

```bash
# 1. Clone / copy this project
cd kaggle_competition_collector

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install the Playwright browser engine (one-time)
playwright install chromium

# 5. Configure credentials
#    Option A – Kaggle API key (required for data download)
#      Place your kaggle.json at  %USERPROFILE%\.kaggle\kaggle.json
#      Content: {"username": "you", "key": "your-api-key"}
#
#    Option B – Browser login credentials (optional, for browser collection)
cp .env.example .env
#      Edit .env and fill in KAGGLE_USERNAME and KAGGLE_PASSWORD
```

---

## Usage

### Basic — collect everything

```bash
python main.py --competition playground-series-s6e5
```

### With a full URL

```bash
python main.py --competition https://www.kaggle.com/competitions/titanic
```

### Custom output folder

```bash
python main.py --competition titanic --output C:\kaggle_data
```

### Show the browser window (manual login / debugging)

```bash
python main.py --competition titanic --headed
```

### Skip data download (faster, for re-scraping page content)

```bash
python main.py --competition titanic --no-download
```

### API / data only — no browser at all

```bash
python main.py --competition titanic --no-browser
```

### Skip specific tabs

```bash
python main.py --competition titanic --skip-tabs leaderboard discussion
```

---

## Credentials & login strategy

### Kaggle API (data download)

The tool uses the [official kaggle Python package](https://github.com/Kaggle/kaggle-api).
It automatically reads `~/.kaggle/kaggle.json`.  You can also set the
`KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables (or put them in `.env`).

### Browser login

Browser session is handled in this priority order:

1. **Saved session** — `~/.kaggle_collector/browser_state.json`  
   Created automatically after the first successful login.  Reused on every subsequent run so you never need to log in again.
2. **`.env` credentials** — `KAGGLE_USERNAME` + `KAGGLE_PASSWORD`  
   Filled into the login form automatically.
3. **Manual login** (`--headed` mode)  
   The browser opens visually; you log in by hand and press Enter in the terminal.  The session is then saved for future headless runs.

---

## Competition rules & data access walls

The tool **never** tries to bypass acceptance walls or access restrictions.  
If a competition requires you to accept its rules before you can view the data
or certain pages, the tool will:

- Print a clear warning with the URL to visit
- Save a screenshot of the wall for reference
- Continue collecting whatever else it can access

---

## Compliance notes

- Only publicly visible information (to your own account) is collected.
- No credentials, cookies, or API keys are ever hardcoded in source code.
- Polite delays (≥ 2 s) are inserted between every browser page request.
- All content is saved locally; nothing is sent to any third-party service.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `kaggle: command not found` | Run `pip install kaggle` |
| `playwright: No module named playwright` | Run `pip install playwright && playwright install chromium` |
| 403 on data download | Accept the competition rules on Kaggle first |
| Browser login fails (auto) | Check `KAGGLE_USERNAME`/`KAGGLE_PASSWORD` in `.env`, or use `--headed` |
| Pages show very little text | Use `--headed` to watch the page render; some competitions need JS-heavy wait times |
| Session expired | Delete `~/.kaggle_collector/browser_state.json` and re-run with `--headed` |
