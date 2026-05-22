"""P1.5 — Sentiment scoring + importance-weighted aggregation.

FinBERT (``ProsusAI/finbert``) scores each piece of text into
``{positive, neutral, negative}`` probabilities, condensed into a single
polarity in [-1, 1]. The three channels (news / Reddit / X) are aggregated
*separately* because they behave very differently — news is lagging, Reddit
is contrarian, X is noisy and fast — and a combined-mean would smear those
signals together.

Each item's contribution to the per-(ticker, date) mean is weighted by a
**composite importance score** built from three orthogonal axes:

    importance = engagement × source_quality × confidence

* ``engagement``     — ``log1p(score) + 1``. Diminishing returns on raw
                       upvote / like counts so a single viral post can't
                       drown out the rest of the day's chatter. News rows
                       have no engagement signal, so default to 1.0.
* ``source_quality`` — curated per-channel multiplier. For news we trust
                       Reuters / Bloomberg / Moneycontrol / Economic Times
                       more than unknown blogs. For Reddit we trust
                       on-topic subs (r/IndianStockMarket, r/NIFTY50) more
                       than off-topic ones. X is flat (no per-account
                       credibility model yet).
* ``confidence``     — ``1 - neutral_probability`` clipped to a floor.
                       FinBERT outputs how neutral the text is; items it
                       was confident about (highly pos or highly neg) carry
                       more weight than mushy neutral text that adds noise.

Recency decay (``_exp_weights``) is then applied multiplicatively on top of
the composite importance, so the final weight used in ``np.average`` is:

    final_weight = recency_decay × engagement × source_quality × confidence

Output schema (one row per ticker, date, source):

    ticker, date, source,
    sent_mean_3d, sent_mean_7d,
    sent_pos_share, sent_neg_share,
    sent_volume_z, n_3d, n_7d
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ingestion._firestore import batch_write, init_firestore_client, wipe_collection
from ingestion.aliases import load_aliases

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "Data" / "archive" / "sentiment_features.parquet"
NEWS_PATH = Path(__file__).resolve().parents[1] / "Data" / "archive" / "news.parquet"
REDDIT_PATH = Path(__file__).resolve().parents[1] / "Data" / "archive" / "reddit_posts.parquet"
X_PATH = Path(__file__).resolve().parents[1] / "Data" / "archive" / "x_posts.parquet"

DEFAULT_MODEL = "ProsusAI/finbert"
HALF_LIFE_FACTOR = 0.5
BASELINE_DAYS = 90

# Firestore: one document per (ticker, source) holding the latest sentiment
# snapshot. Each refresh **wipes the collection** before writing, so no stale
# rows accumulate over time.
FIRESTORE_COLLECTION = "sentiment_latest"

# Curated credibility multipliers used by the importance-weighted aggregator.
# Comparisons are case-insensitive; lookups normalise to lowercase first.
NEWS_SOURCE_WEIGHTS = {
    # Tier-1 wires & top-tier Indian business press
    "reuters": 1.5,
    "bloomberg": 1.5,
    "financial times": 1.5,
    "wall street journal": 1.5,
    "wsj": 1.5,
    "moneycontrol": 1.5,
    "economic times": 1.5,
    "the economic times": 1.5,
    "et markets": 1.5,
    "mint": 1.5,
    "livemint": 1.5,
    "business standard": 1.5,
    "bq prime": 1.5,
    "bqprime": 1.5,
    "cnbc-tv18": 1.5,
    "cnbctv18": 1.5,
    "ndtv profit": 1.5,
    "ndtv": 1.5,
    "the hindu businessline": 1.5,
    "hindu businessline": 1.5,
    # Tier-2 aggregators / mid-credibility outlets
    "yahoo finance": 1.0,
    "investing.com": 1.0,
    "bloombergquint": 1.0,
    "forbes": 1.0,
    "reuters india": 1.0,
    "marketwatch": 1.0,
    "cnbc": 1.0,
}
REDDIT_SOURCE_WEIGHTS = {
    # On-topic for NIFTY 50 swing trading
    "indianstockmarket": 1.5,
    "indiainvestments": 1.5,
    "nifty50": 1.5,
    # Adjacent but noisier
    "stockmarketindia": 1.0,
    "dalalstreettalks": 1.0,
}
DEFAULT_SOURCE_WEIGHT = 0.5  # unknown source / subreddit
X_SOURCE_WEIGHT = 1.0  # flat for X — no per-account credibility model yet
MIN_CONFIDENCE = 0.1  # floor so genuinely-neutral text isn't zeroed out


def _engagement_weight(score: float) -> float:
    """``log1p(max(score, 0)) + 1`` — diminishing returns on raw counts.

    score=0  → 1.0
    score=10 → log(11)+1 ≈ 3.4
    score=1000 → log(1001)+1 ≈ 7.9
    So a 1000-upvote post is worth ~2.3× a 10-upvote one, not 100×.
    """
    safe = max(float(score or 0.0), 0.0)
    return float(math.log1p(safe) + 1.0)


def _source_quality_news(source: Optional[str]) -> float:
    if not source:
        return DEFAULT_SOURCE_WEIGHT
    return NEWS_SOURCE_WEIGHTS.get(str(source).strip().lower(), DEFAULT_SOURCE_WEIGHT)


def _source_quality_reddit(subreddit: Optional[str]) -> float:
    if not subreddit:
        return DEFAULT_SOURCE_WEIGHT
    return REDDIT_SOURCE_WEIGHTS.get(str(subreddit).strip().lower(), DEFAULT_SOURCE_WEIGHT)


def _confidence_weight(neutral_prob: float) -> float:
    """``max(1 - neutral_prob, MIN_CONFIDENCE)``.

    FinBERT's ``neutral`` head close to 1 → text was vague, contribute less.
    Close to 0 → polarity is decisive (high pos or high neg), contribute more.
    Floor at MIN_CONFIDENCE so a perfectly neutral item still counts a little.
    """
    try:
        value = 1.0 - float(neutral_prob)
    except (TypeError, ValueError):
        return MIN_CONFIDENCE
    return max(value, MIN_CONFIDENCE)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


@dataclass
class FinBertScorer:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 32

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Sentiment scoring requires `transformers`. Install via requirements.txt."
            ) from exc
        import torch  # noqa: F401  (verify available)

        self._torch = __import__("torch")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        # FinBERT labels: 0=positive, 1=negative, 2=neutral (per ProsusAI repo)
        self._labels = self._model.config.id2label

    def score(self, texts: List[str]) -> pd.DataFrame:
        if not texts:
            return pd.DataFrame(columns=["positive", "negative", "neutral", "polarity"])
        rows: List[List[float]] = []
        torch = self._torch
        for start in range(0, len(texts), self.batch_size):
            batch = [t[:512] for t in texts[start : start + self.batch_size]]
            encoded = self._tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for prob in probs:
                pos = float(prob[self._index("positive")])
                neg = float(prob[self._index("negative")])
                neu = float(prob[self._index("neutral")])
                polarity = pos - neg
                rows.append([pos, neg, neu, polarity])
        return pd.DataFrame(rows, columns=["positive", "negative", "neutral", "polarity"])

    def _index(self, label: str) -> int:
        for idx, name in self._labels.items():
            if name.lower() == label:
                return int(idx)
        raise KeyError(label)


def _exp_weights(scores: pd.DataFrame, weight_col: str, half_life_days: float, asof_date: pd.Timestamp) -> pd.Series:
    """Recency decay × the precomputed importance weight.

    The composite importance built upstream is already non-zero (engagement
    floor 1.0 × source floor 0.5 × confidence floor 0.1 ⇒ ≈ 0.05) so we
    don't need the historical ``+ 1`` smoothing here.
    """
    deltas = (asof_date - scores["date"]).dt.days.clip(lower=0)
    decay = np.exp(-np.log(2) * deltas / half_life_days)
    weights = scores[weight_col].clip(lower=0).astype(float)
    return decay * weights


def _aggregate(
    scored: pd.DataFrame,
    source: str,
    weight_col: str,
) -> pd.DataFrame:
    """Compute per-(ticker, date) features from scored items."""
    if scored.empty:
        return pd.DataFrame()
    df = scored.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.tz_convert("Asia/Kolkata").dt.date
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"])

    # Build the (ticker, date) grid over the observed window
    out_rows: List[dict] = []
    for ticker, group in df.groupby("ticker"):
        start = group["date"].min()
        end = group["date"].max()
        if pd.isna(start) or pd.isna(end):
            continue
        date_index = pd.date_range(start=start, end=end, freq="D")
        group_sorted = group.sort_values("date")
        baseline_counts = group_sorted.groupby("date").size()

        for asof in date_index:
            window_7d = group_sorted[(group_sorted["date"] > asof - pd.Timedelta(days=7)) & (group_sorted["date"] <= asof)]
            window_3d = group_sorted[(group_sorted["date"] > asof - pd.Timedelta(days=3)) & (group_sorted["date"] <= asof)]

            n_3d = int(len(window_3d))
            n_7d = int(len(window_7d))

            if n_7d == 0:
                continue

            w7 = _exp_weights(window_7d, weight_col, half_life_days=3.0, asof_date=asof)
            w3 = _exp_weights(window_3d, weight_col, half_life_days=1.5, asof_date=asof) if n_3d else None

            sent_mean_7d = float(np.average(window_7d["polarity"], weights=w7)) if w7.sum() > 0 else float("nan")
            sent_mean_3d = (
                float(np.average(window_3d["polarity"], weights=w3)) if (n_3d and w3 is not None and w3.sum() > 0) else float("nan")
            )

            pos_share = float((window_7d["polarity"] > 0.1).mean()) if n_7d else float("nan")
            neg_share = float((window_7d["polarity"] < -0.1).mean()) if n_7d else float("nan")

            baseline_window = baseline_counts[
                (baseline_counts.index > asof - pd.Timedelta(days=BASELINE_DAYS))
                & (baseline_counts.index <= asof)
            ]
            baseline_mean = float(baseline_window.mean()) if len(baseline_window) else 0.0
            baseline_std = float(baseline_window.std(ddof=0)) if len(baseline_window) > 1 else 0.0
            current_count = float(baseline_counts.get(asof, 0))
            if baseline_std > 0:
                volume_z = (current_count - baseline_mean) / baseline_std
            else:
                volume_z = 0.0

            out_rows.append(
                {
                    "ticker": ticker,
                    "date": asof.date(),
                    "source": source,
                    "sent_mean_3d": sent_mean_3d,
                    "sent_mean_7d": sent_mean_7d,
                    "sent_pos_share": pos_share,
                    "sent_neg_share": neg_share,
                    "sent_volume_z": float(volume_z),
                    "n_3d": n_3d,
                    "n_7d": n_7d,
                }
            )

    return pd.DataFrame(out_rows)


def _prepare_news(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return news_df
    df = news_df.copy()
    df["text"] = df["title"].fillna("")
    df["engagement_raw"] = 0.0  # yfinance news has no engagement signal
    if "source" not in df.columns:
        df["source_label"] = None
    else:
        df["source_label"] = df["source"]
    return df[["ticker", "ts", "text", "engagement_raw", "source_label"]]


def _prepare_reddit(reddit_df: pd.DataFrame) -> pd.DataFrame:
    if reddit_df.empty:
        return reddit_df
    df = reddit_df.copy()
    df["text"] = (df["title"].fillna("") + "\n" + df["body"].fillna("")).str.strip()
    df["engagement_raw"] = df["score"].fillna(0).clip(lower=0).astype(float)
    df["source_label"] = df["subreddit"] if "subreddit" in df.columns else None
    return df[["ticker", "ts", "text", "engagement_raw", "source_label"]]


def _prepare_x(x_df: pd.DataFrame) -> pd.DataFrame:
    if x_df.empty:
        return x_df
    df = x_df.copy()
    df["text"] = df["body"].fillna("")
    df["engagement_raw"] = df["score"].fillna(0).clip(lower=0).astype(float)
    df["source_label"] = None  # X has no source_quality signal we trust yet
    return df[["ticker", "ts", "text", "engagement_raw", "source_label"]]


def _compose_importance(df: pd.DataFrame, source: str) -> pd.Series:
    """Return the composite per-row importance weight.

    Weight = ``engagement × source_quality × confidence``. ``recency_decay``
    is layered on inside ``_exp_weights`` at aggregation time.
    """
    if source == "news":
        quality_fn = _source_quality_news
    elif source == "reddit":
        quality_fn = _source_quality_reddit
    else:  # x or any future channel without per-source scoring
        quality_fn = lambda _: X_SOURCE_WEIGHT

    engagement = df["engagement_raw"].astype(float).apply(_engagement_weight)
    source_quality = df["source_label"].apply(quality_fn).astype(float)
    confidence = df["neutral"].astype(float).apply(_confidence_weight)
    return engagement * source_quality * confidence


def score_source(prepared: pd.DataFrame, scorer: FinBertScorer, source: str) -> pd.DataFrame:
    if prepared.empty:
        return pd.DataFrame()
    texts = prepared["text"].fillna("").tolist()
    scores = scorer.score(texts)
    if scores.empty:
        return pd.DataFrame()
    df = pd.concat([prepared.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    # Composite importance is computed *after* FinBERT scoring so we can use
    # the model's confidence as one of the weighting axes.
    df["weight"] = _compose_importance(df, source=source)
    aggregated = _aggregate(df, source=source, weight_col="weight")
    return aggregated


# ----------------------------------------------------------------------------
# Firestore writer — latest sentiment snapshot per (ticker, source)
# ----------------------------------------------------------------------------


def _company_name_for(ticker: str) -> str:
    entry = load_aliases().get(str(ticker).upper(), {})
    return entry.get("name") or str(ticker)


def _none_if_nan(value: Any) -> Any:
    """Firestore rejects NaN floats; coerce to None."""
    try:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_per_ticker_source(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent row per ``(ticker, source)`` group."""
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    idx = df.groupby(["ticker", "source"])["date"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def _firestore_payload(row: dict, scrape_date: str) -> dict:
    date_value = row.get("date")
    if isinstance(date_value, pd.Timestamp):
        as_of_date = date_value.date().isoformat()
    elif hasattr(date_value, "isoformat"):
        as_of_date = date_value.isoformat()
    else:
        as_of_date = str(date_value) if date_value is not None else None

    return {
        "company_name": _company_name_for(row.get("ticker", "")),
        "ticker": row.get("ticker"),
        "source": row.get("source"),
        "as_of_date": as_of_date,
        "sent_mean_3d": _none_if_nan(row.get("sent_mean_3d")),
        "sent_mean_7d": _none_if_nan(row.get("sent_mean_7d")),
        "sent_pos_share": _none_if_nan(row.get("sent_pos_share")),
        "sent_neg_share": _none_if_nan(row.get("sent_neg_share")),
        "sent_volume_z": _none_if_nan(row.get("sent_volume_z")),
        "n_3d": int(row.get("n_3d") or 0),
        "n_7d": int(row.get("n_7d") or 0),
        "scrape_date": scrape_date,
    }


def _firestore_doc_id(row: dict) -> str:
    return f"{row.get('ticker', '')}_{row.get('source', '')}"


def write_sentiment_to_firestore(
    df: pd.DataFrame,
    collection: str = FIRESTORE_COLLECTION,
    client: Optional[Any] = None,
) -> int:
    """Wipe ``collection`` then write the latest row per (ticker, source).

    "Wipe-then-write" guarantees the user's "no previous data should be
    there" requirement — a (ticker, source) pair that produced no rows in
    today's run will not have a stale doc carried over from yesterday.
    """
    if df.empty:
        return 0
    client = client or init_firestore_client()
    wipe_collection(client, collection)
    latest = _latest_per_ticker_source(df)
    if latest.empty:
        return 0
    scrape_date = datetime.now(timezone.utc).date().isoformat()
    docs = (
        (_firestore_doc_id(row), _firestore_payload(row, scrape_date))
        for row in latest.to_dict(orient="records")
        if row.get("ticker") and row.get("source")
    )
    return batch_write(client, collection, docs)


def refresh_sentiment(
    output_path: Path = DEFAULT_OUTPUT,
    news_path: Path = NEWS_PATH,
    reddit_path: Path = REDDIT_PATH,
    x_path: Path = X_PATH,
    scorer: Optional[FinBertScorer] = None,
    write_firestore: bool = False,
    firestore_client: Optional[Any] = None,
) -> pd.DataFrame:
    scorer = scorer or FinBertScorer()
    frames: List[pd.DataFrame] = []

    if news_path.exists():
        news_df = pd.read_parquet(news_path)
        prepared = _prepare_news(news_df)
        agg = score_source(prepared, scorer, source="news")
        if not agg.empty:
            frames.append(agg)
            _log(f"[sentiment] news: {len(agg)} (ticker, date) rows")
    else:
        _log(f"[sentiment] no news at {news_path}")

    if reddit_path.exists():
        reddit_df = pd.read_parquet(reddit_path)
        prepared = _prepare_reddit(reddit_df)
        agg = score_source(prepared, scorer, source="reddit")
        if not agg.empty:
            frames.append(agg)
            _log(f"[sentiment] reddit: {len(agg)} (ticker, date) rows")
    else:
        _log(f"[sentiment] no reddit at {reddit_path}")

    if x_path.exists():
        x_df = pd.read_parquet(x_path)
        prepared = _prepare_x(x_df)
        agg = score_source(prepared, scorer, source="x")
        if not agg.empty:
            frames.append(agg)
            _log(f"[sentiment] x: {len(agg)} (ticker, date) rows")
    else:
        _log(f"[sentiment] no x at {x_path}")

    if not frames:
        empty = pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "source",
                "sent_mean_3d",
                "sent_mean_7d",
                "sent_pos_share",
                "sent_neg_share",
                "sent_volume_z",
                "n_3d",
                "n_7d",
            ]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(output_path, index=False)
        _log("[sentiment] no sources had data; wrote empty frame")
        return empty

    combined = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    _log(f"[sentiment] wrote {len(combined)} rows to {output_path}")

    if write_firestore and not combined.empty:
        try:
            written = write_sentiment_to_firestore(combined, client=firestore_client)
            _log(
                f"[sentiment] wrote {written} latest-snapshot docs to Firestore "
                f"collection '{FIRESTORE_COLLECTION}' (prior contents wiped)"
            )
            output_path.unlink(missing_ok=True)
            _log(f"[sentiment] removed staged archive {output_path}")
        except Exception as exc:
            _log(f"[sentiment] Firestore write failed: {exc}")
    elif not write_firestore:
        _log("[sentiment] Firestore write skipped (--no-firestore); staged archive retained")

    return combined


def load_sentiment(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score sentiment for ingested news / reddit / x.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--news", default=str(NEWS_PATH))
    parser.add_argument("--reddit", default=str(REDDIT_PATH))
    parser.add_argument("--x", default=str(X_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--no-firestore",
        action="store_true",
        help="Skip the Firestore wipe-and-write step (parquet only).",
    )
    args = parser.parse_args(argv)

    scorer = FinBertScorer(model_name=args.model)
    write_firestore = not args.no_firestore and bool(
        os.environ.get("GOOGLE_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
    df = refresh_sentiment(
        output_path=Path(args.output),
        news_path=Path(args.news),
        reddit_path=Path(args.reddit),
        x_path=Path(args.x),
        scorer=scorer,
        write_firestore=write_firestore,
    )
    print(json.dumps({"rows": int(len(df))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
