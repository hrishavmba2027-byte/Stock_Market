"""Tests for sector detection + entity mapping (Phase 3 production change)."""
from __future__ import annotations

from ingestion import sectors, news_ingest
from features import trade_suggestions as ts


def test_sector_entity_id_roundtrips():
    eid = sectors.sector_entity_id("Oil & Gas")
    assert eid == "SECTOR__OIL_GAS"
    assert sectors.is_sector_entity(eid)
    assert sectors.sector_name_for_entity(eid) == "Oil & Gas"


def test_find_sectors_in_text_matches_curated_terms():
    assert "IT" in sectors.find_sectors_in_text("IT services exporters gain on demand")
    assert "Financials" in sectors.find_sectors_in_text("PSU banks lead the banking sector rally")
    assert sectors.find_sectors_in_text("weather update for the weekend") == []


def test_map_entities_company_implies_sector(monkeypatch):
    monkeypatch.setenv("ENABLE_SECTOR_NEWS", "true")
    ents = news_ingest._map_entities("TCS bags a record IT services deal")
    assert "TCS" in ents
    assert sectors.sector_entity_id("IT") in ents


def test_map_entities_macro_is_general():
    assert news_ingest._map_entities("RBI holds repo rate steady") == ["GENERAL"]


def test_map_entities_sector_news_toggle_off(monkeypatch):
    monkeypatch.setattr(news_ingest, "_sector_news_enabled", lambda: False)
    ents = news_ingest._map_entities("TCS wins IT services deal")
    assert ents == ["TCS"]  # no sector entity when disabled


def test_entity_kind_classification():
    assert news_ingest._entity_kind("TCS") == "company"
    assert news_ingest._entity_kind("GENERAL") == "general"
    assert news_ingest._entity_kind(sectors.sector_entity_id("IT")) == "sector"


def test_llm_sentiment_context_excludes_other_entities():
    it_entity = sectors.sector_entity_id("IT")
    # Build a sentiment map with the company, its sector, general, AND unrelated
    # noise (another company + another sector) that must NOT leak through.
    sentiment = {
        "TCS": [{"source": "news", "sent_mean_7d": 0.4}],
        it_entity: [{"source": "news", "sent_mean_7d": 0.2}],
        "GENERAL": [{"source": "news", "sent_mean_7d": -0.1}],
        "RELIANCE": [{"source": "news", "sent_mean_7d": 0.9}],
        sectors.sector_entity_id("Oil & Gas"): [{"source": "news", "sent_mean_7d": 0.9}],
    }
    ctx = ts.sentiment_context_for("TCS", sentiment)
    assert ctx["company"] == sentiment["TCS"]
    assert ctx["sector"]["name"] == "IT"
    assert ctx["sector"]["sentiment"] == sentiment[it_entity]
    assert ctx["general"] == sentiment["GENERAL"]
    # No other company / sector sentiment is present anywhere in the context.
    flat = str(ctx)
    assert "0.9" not in flat
    assert ts._has_sentiment(ctx) is True
