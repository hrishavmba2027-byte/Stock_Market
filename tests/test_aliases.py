from ingestion.aliases import find_tickers_in_text, list_tickers, sector_for


def test_all_nifty50_tickers_present():
    tickers = set(list_tickers())
    expected = {"RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC"}
    assert expected.issubset(tickers)


def test_sector_lookup_returns_known_sector():
    assert sector_for("RELIANCE") == "Oil & Gas"
    assert sector_for("TCS") == "IT"
    assert sector_for("UNKNOWN_TICKER") == "Unknown"


def test_find_tickers_matches_symbol_and_alias():
    hits = find_tickers_in_text("Reliance Industries beat estimates")
    assert "RELIANCE" in hits


def test_find_tickers_matches_dollar_cashtag():
    hits = find_tickers_in_text("Picking up $TCS at this level looks fine")
    assert "TCS" in hits


def test_find_tickers_ignores_substring_collisions():
    # "ITC" is contained in many words; alias matching must respect word boundaries
    hits = find_tickers_in_text("Switch to a fitc product")  # 'itc' inside 'fitc'
    assert "ITC" not in hits


def test_find_tickers_returns_empty_for_blank_text():
    assert find_tickers_in_text("") == []
    assert find_tickers_in_text(None) == []
