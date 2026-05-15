# Framing Divergence Explorer

A user-facing demonstration tool for the FDI (Framing Divergence Indicators)
metric framework. Built with Streamlit.

The app takes two news articles about the same event and reports how their
framing differs along three interpretable dimensions:

- **Words (L)** — How loaded is the language?
- **People (E)** — How are the same people described?
- **Facts (C)** — What's covered vs left out?

---

## Setup

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download the spaCy English model:**

   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **(Optional but recommended) Set up Google Cloud Natural Language API:**

   - Create a service account in the Google Cloud Console.
   - Download the service account JSON key.
   - Set the environment variable:

     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
     ```

   Without this, the **People** indicator will be disabled — the other two
   indicators still work.

4. **(Optional) Enable percentile scoring against the audited corpus:**

   Place `results_fdi_pairs.csv` (from `fdi_pipeline.ipynb`) in the project
   root. The app will then show each indicator's percentile against the
   1,308-pair audit distribution.

---

## Run

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501.

## Architecture

| Component | Backend | Speed |
|---|---|---|
| `compute_L` (Words) | Local lexicon matching, pure Python | Instant |
| `compute_C` (Facts) | spaCy `en_core_web_sm` entity extraction, Jaccard distance | ~1 second per article |
| `compute_E` (People) | Google Cloud NLP `analyze_entity_sentiment` | ~2 seconds per article |

All three are wrapped in `compute_all_indicators`, cached by content hash
via `@st.cache_data`. Re-running with the same input is instant.

Highlighting is implemented in pure Python (no JavaScript) by rendering
HTML spans inside `st.markdown(..., unsafe_allow_html=True)`.

---

## License & citation

Masters thesis prototype. Cite as: Ubaidah, D. I. A. (2026), *Auditing
Commercial NLP Services for Linguistic Framing Bias in News Media*,
Masters thesis, Monash University Faculty of Information Technology.
