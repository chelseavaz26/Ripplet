# Ripplet

**AI-driven social media analytics — sentiment, demographics, trends, and influence, from one shared dataset.**

Built for SIH 2026, Problem Statement: *Social Media Analytics*.

---

## What it does

Ripplet takes a dataset of tweets and turns it into four kinds of audience insight, all connected to the same underlying data — not four separate scripts glued together:

- **Sentiment & Emotion** — polarity (positive/negative/neutral) and discrete emotion (joy, anger, fear, sadness, surprise, disgust) for every tweet, using transformer models fine-tuned on Twitter text. Also flags possible irony/sarcasm as a lower-confidence signal.
- **Demographics** — aggregate, anonymized audience breakdown by language, region, and interest category, inferred from bio and location text. Deliberately does **not** guess age or gender at an individual level — the data doesn't support that reliably.
- **Trends** — hashtag and keyword frequency tracked over time, flagging terms that spike above their normal rate.
- **Network & Influence** — a directed graph of who mentions or retweets whom, ranked by PageRank to surface key opinion leaders (KOLs) and how influence flows through the audience.

A time slider drives all four views together, and an **Influencer Spotlight** panel ties them into one example: pick a top-ranked user, see their actual tweet, its sentiment, and which trending topic it matches — proof the four modules share one dataset, not four disconnected reports.

## Live demo

`https://ripplet.streamlit.app/`

## Scope, stated up front

This is a prototype, and its scope was chosen deliberately rather than left implicit:

- **One platform (X/Twitter), one dataset.** The architecture (raw data → cleaning → shared timestamped store → analytics modules) is designed to generalize to other platforms, but each platform's trend/network/demographic logic needs its own adapter — a subreddit isn't a hashtag, a comment tree isn't a retweet chain.
- **Timeline replay, not a live feed.** Continuous live ingestion via the official X API was evaluated and ruled out for this prototype stage — pay-per-use pricing with no free tier makes it cost-prohibitive to run continuously during development. Instead, the historical dataset is replayed in timestamp order to simulate a live stream, and the pipeline is built so a real API feed could be swapped in without changing the architecture.
- **No individual-level demographic guessing.** Age and gender are left out entirely rather than inferred unreliably from bio text — a deliberate accuracy and privacy choice, not an oversight.
- **Irony/sarcasm detection is flagged, not asserted.** Even strong published models for this task top out around 70–80% accuracy. The dashboard shows it as a "possible irony" signal with a confidence caveat, never as a fact.

## Tech stack

- **Data pipeline:** pandas, Parquet
- **Sentiment & emotion:** `cardiffnlp/twitter-roberta-base-sentiment-latest`, `j-hartmann/emotion-english-distilroberta-base` (via Hugging Face `transformers`)
- **Irony detection:** `cardiffnlp/twitter-roberta-base-irony`
- **Demographics:** `langdetect`, `pycountry`, keyword-based interest classification
- **Network analysis:** NetworkX (PageRank, centrality), pyvis for visualization
- **Dashboard:** Streamlit

## Project structure

```
Ripplet/
├── data/                   # raw source dataset (read-only)
├── pipeline/
│   ├── cleaner.py          # one-time raw → clean Parquet conversion
│   ├── loader.py           # get_window() / stream_batches() — the only way modules read data
│   └── enrich_sentiment.py # offline batch model inference (sentiment, emotion, irony)
├── modules/
│   ├── sentiment/
│   ├── demographics/
│   ├── trends/
│   └── network/
└── dashboard/
    └── app.py              # Streamlit app tying all four modules together
```

Every module reads data through `pipeline/loader.py` and nothing else — this keeps sentiment, demographics, trends, and network analysis all describing the same underlying tweets.

## Running it locally

```bash
# 1. Clone and enter the project
git clone https://github.com/chelseavaz26/Ripplet.git
cd Ripplet

# 2. Set up the environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# 3. (First run only) Enrich the dataset with model predictions
#    This runs sentiment, emotion, and irony models once and saves the
#    results — the dashboard itself never runs models live.
python pipeline/enrich_sentiment.py

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

## Known limitations

- **Emotion classifier shows a strong "fear" skew** on this dataset — consistent with a general-purpose emotion model responding to pandemic-related vocabulary rather than necessarily capturing genuine emotional stance.
- **Author identity is a display name, not a handle** (the source dataset has no handle field), while mentioned users are identified by handle — so an author who is also a highly-mentioned figure may not be recognized as the same node in the network graph.
- **Irony detection will produce false positives** — it's shown as a flagged signal for exactly this reason, not a confirmed label.

## Team

Built for Smart India Hackathon 2026.
