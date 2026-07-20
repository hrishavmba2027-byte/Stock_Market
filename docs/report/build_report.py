#!/usr/bin/env python3
"""Render the Signal Ledger project report to PDF via WeasyPrint.

Run from the outputs sandbox where report_assets/ holds the fonts.
"""
import os
from weasyprint import HTML

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.environ.get("REPORT_ASSETS", os.path.join(HERE, "report_assets"))


def font_face(family, filename, weight="normal", style="normal"):
    path = os.path.join(ASSETS, filename).replace("\\", "/")
    return (
        f"@font-face {{ font-family:'{family}'; font-weight:{weight}; "
        f"font-style:{style}; src:url('file://{path}'); }}"
    )


FONTS = "\n".join([
    font_face("Gloock", "Gloock-Regular.ttf"),
    font_face("Shoulders", "BigShoulders-Regular.ttf", "400"),
    font_face("Shoulders", "BigShoulders-Bold.ttf", "700"),
    font_face("Plex", "IBMPlexSerif-Regular.ttf", "400"),
    font_face("Plex", "IBMPlexSerif-Bold.ttf", "700"),
    font_face("Plex", "IBMPlexSerif-Italic.ttf", "400", "italic"),
    font_face("Mono", "IBMPlexMono-Regular.ttf", "400"),
    font_face("Mono", "IBMPlexMono-Bold.ttf", "700"),
])

# ── palette ──────────────────────────────────────────────────────────────────
INK = "#1a2233"        # deep ink navy
INK_SOFT = "#42506b"
PAPER = "#f4f0e6"      # warm paper
PAPER_DEEP = "#eae3d2"
GOLD = "#b5852a"       # muted saffron-gold accent
GOLD_SOFT = "#cda24a"
RULE = "#c9bfa6"
GREEN = "#3f6b4b"
RED = "#9c3b32"
LINE = "#d8cfb8"

CSS = f"""
{FONTS}
@page {{
  size: A4;
  margin: 20mm 18mm 18mm 18mm;
  @bottom-left {{
    content: "SIGNAL LEDGER";
    font-family: 'Mono'; font-size: 7pt; letter-spacing: 2px;
    color: {INK_SOFT};
  }}
  @bottom-center {{
    content: "Equity Decision Pipeline";
    font-family: 'Mono'; font-size: 7pt; letter-spacing: 1px; color: {INK_SOFT};
  }}
  @bottom-right {{
    content: counter(page) " / " counter(pages);
    font-family: 'Mono'; font-size: 7pt; color: {INK_SOFT};
  }}
}}
@page cover {{
  margin: 0;
  @bottom-left {{ content: none; }}
  @bottom-center {{ content: none; }}
  @bottom-right {{ content: none; }}
}}
@page :first {{
  margin: 0;
  @bottom-left {{ content: none; }}
  @bottom-center {{ content: none; }}
  @bottom-right {{ content: none; }}
}}

* {{ box-sizing: border-box; }}
html {{ -weasy-hyphens: none; }}
body {{
  font-family: 'Plex'; color: {INK}; background: {PAPER};
  font-size: 10pt; line-height: 1.5; margin: 0;
}}
p {{ margin: 0 0 8pt 0; text-align: justify; }}
strong {{ font-weight: 700; color: {INK}; }}
em {{ font-style: italic; color: {INK_SOFT}; }}

/* ── cover ─────────────────────────────────────────────────────────────── */
.cover {{
  page: cover; position: relative; width: 210mm; height: 297mm;
  background: {PAPER}; overflow: hidden; padding: 24mm 20mm;
}}
.cover .frame {{
  position: absolute; top: 12mm; left: 12mm; right: 12mm; bottom: 12mm;
  border: 1px solid {RULE};
}}
.cover .tick {{
  position: absolute; font-family:'Mono'; font-size: 7pt; color: {INK_SOFT};
  letter-spacing: 2px;
}}
.cover-tag {{
  font-family:'Mono'; font-size: 8.5pt; letter-spacing: 5px; color: {GOLD};
  text-transform: uppercase; margin-top: 6mm;
}}
.cover-rule {{ width: 46mm; height: 2px; background: {INK}; margin: 8mm 0 10mm 0; }}
.cover h1 {{
  font-family:'Gloock'; font-weight: 400; color: {INK};
  font-size: 52pt; line-height: 1.02; margin: 0; letter-spacing: -0.5px;
}}
.cover h1 .accent {{ color: {GOLD}; }}
.cover .sub {{
  font-family:'Plex'; font-style: italic; font-size: 14pt; color: {INK_SOFT};
  margin-top: 8mm; max-width: 130mm;
}}
.cover .meta {{
  position: absolute; left: 20mm; bottom: 26mm; right: 20mm;
  display: flex; justify-content: space-between; align-items: flex-end;
}}
.cover .meta .block {{ font-family:'Mono'; font-size: 8pt; color: {INK_SOFT}; line-height: 1.9; }}
.cover .meta .block b {{ color: {INK}; font-weight: 700; }}
.cover .plate {{
  font-family:'Shoulders'; font-weight: 700; font-size: 150pt; color: {PAPER_DEEP};
  position: absolute; right: 16mm; top: 92mm; line-height: 0.8; letter-spacing: -4px;
}}
.cover .flowmini {{ position: absolute; left: 20mm; bottom: 52mm; right: 20mm; }}

/* ── section headers ───────────────────────────────────────────────────── */
.section {{ margin-top: 4mm; }}
.kicker {{
  font-family:'Mono'; font-size: 8pt; letter-spacing: 3px; color: {GOLD};
  text-transform: uppercase; margin-bottom: 2mm;
}}
h2 {{
  font-family:'Gloock'; font-weight: 400; font-size: 21pt; color: {INK};
  margin: 0 0 3mm 0; line-height: 1.1; letter-spacing: -0.3px;
}}
h2 .num {{
  font-family:'Shoulders'; font-weight: 700; color: {GOLD}; font-size: 21pt;
  margin-right: 4mm;
}}
h3 {{
  font-family:'Plex'; font-weight: 700; font-size: 11pt; color: {INK};
  margin: 5mm 0 2mm 0;
}}
.lead {{ font-size: 11pt; line-height: 1.55; color: {INK}; }}
.lead .first {{
  font-family:'Gloock'; font-size: 15pt; color: {GOLD}; line-height: 1;
}}
hr.rule {{ border: none; border-top: 1px solid {RULE}; margin: 4mm 0; }}
.small {{ font-size: 8.5pt; color: {INK_SOFT}; }}
code, .mono {{ font-family:'Mono'; font-size: 8.5pt; color: {INK}; }}
.accentword {{ color: {GOLD}; font-weight: 700; }}

/* ── two-column facts ──────────────────────────────────────────────────── */
.facts {{ display: flex; gap: 6mm; margin: 3mm 0; }}
.fact {{
  flex: 1; border-top: 2px solid {INK}; padding-top: 2mm;
}}
.fact .n {{ font-family:'Shoulders'; font-weight: 700; font-size: 27pt; color: {INK}; line-height: 0.95; }}
.fact .n small {{ font-size: 12pt; color: {GOLD}; }}
.fact .l {{ font-family:'Mono'; font-size: 7.5pt; letter-spacing: 1px; color: {INK_SOFT}; text-transform: uppercase; margin-top: 1mm; }}

/* ── stage cards / lists ───────────────────────────────────────────────── */
.stagecard {{
  border: 1px solid {LINE}; border-left: 3px solid {GOLD};
  background: rgba(255,255,255,0.35); padding: 3mm 4mm; margin: 2.5mm 0;
}}
.stagecard .st-h {{ font-family:'Plex'; font-weight:700; font-size: 10pt; color: {INK}; margin-bottom: 1mm; }}
.stagecard .st-h .tag {{ font-family:'Mono'; font-size: 7pt; color: {GOLD}; letter-spacing:1px; margin-right: 3mm; }}
.stagecard p {{ margin: 0; font-size: 9pt; }}

ul.clean {{ margin: 2mm 0 3mm 0; padding-left: 0; list-style: none; }}
ul.clean li {{ padding: 1.4mm 0 1.4mm 7mm; position: relative; font-size: 9.5pt; border-bottom: 1px solid {LINE}; }}
ul.clean li:before {{ content: "→"; position: absolute; left: 0; color: {GOLD}; font-family:'Mono'; }}
ul.clean li b {{ color: {INK}; }}

/* ── gate table ────────────────────────────────────────────────────────── */
table.gates {{ width: 100%; border-collapse: collapse; margin: 3mm 0; }}
table.gates td, table.gates th {{ text-align: left; padding: 2.2mm 3mm; border-bottom: 1px solid {LINE}; font-size: 9pt; vertical-align: top; }}
table.gates th {{ font-family:'Mono'; font-size: 7.5pt; letter-spacing: 1px; text-transform: uppercase; color: {INK_SOFT}; border-bottom: 1.5px solid {INK}; }}
table.gates td.g {{ font-family:'Mono'; font-weight: 700; color: {GOLD}; white-space: nowrap; }}

.callout {{
  border: 1px solid {RULE}; background: rgba(255,255,255,0.4);
  padding: 4mm 5mm; margin: 4mm 0;
}}
.callout .ttl {{ font-family:'Mono'; font-size: 7.5pt; letter-spacing: 2px; color: {GOLD}; text-transform: uppercase; margin-bottom: 1.5mm; }}
.callout p {{ margin: 0; font-size: 9.5pt; }}

.pipeline {{ margin: 3mm 0; }}
.legendbar {{ display:flex; gap: 5mm; font-family:'Mono'; font-size: 7.5pt; color: {INK_SOFT}; margin-top: 2mm; }}
.legendbar span b {{ color: {INK}; }}
.avoid-note {{ font-family:'Plex'; font-style: italic; color: {INK_SOFT}; font-size: 9.5pt; }}
.tradebar {{ display:flex; align-items:stretch; margin: 3mm 0 1mm 0; height: 12mm; border:1px solid {LINE}; }}
.tradebar div {{ display:flex; align-items:center; justify-content:center; font-family:'Mono'; font-size: 7.5pt; letter-spacing:1px; }}
.tradebar .stop {{ background: rgba(156,59,50,0.14); color:{RED}; width: 22%; }}
.tradebar .entry {{ background: rgba(26,34,51,0.08); color:{INK}; width: 30%; border-left:1px solid {LINE}; border-right:1px solid {LINE};}}
.tradebar .tgt {{ background: rgba(63,107,75,0.16); color:{GREEN}; width: 48%; }}
"""


def svg_flow():
    """Horizontal pipeline schematic used on the cover."""
    stages = ["MARKET DATA", "FEATURES", "FORECAST", "SENTIMENT", "LLM GATE", "SIZING"]
    w, h = 170, 26
    n = len(stages)
    gap = w / n
    dots = ""
    for i, s in enumerate(stages):
        cx = gap * i + gap / 2
        fill = GOLD if s == "LLM GATE" else INK
        r = 3.4 if s == "LLM GATE" else 2.2
        dots += f'<circle cx="{cx:.1f}" cy="10" r="{r}" fill="{fill}"/>'
        dots += (f'<text x="{cx:.1f}" y="22" font-family="Mono" font-size="3.1" '
                 f'fill="{INK_SOFT}" text-anchor="middle" letter-spacing="0.5">{s}</text>')
        if i < n - 1:
            x1 = cx + 4
            x2 = gap * (i + 1) + gap / 2 - 4
            dots += f'<line x1="{x1:.1f}" y1="10" x2="{x2:.1f}" y2="10" stroke="{RULE}" stroke-width="0.5"/>'
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg">'
            f'{dots}</svg>')


def svg_pipeline_big():
    """Vertical detailed pipeline used inside the report (System at a glance)."""
    rows = [
        ("01", "Ingestion", "yfinance OHLCV · quarterly fundamentals · news · Reddit · X", INK),
        ("02", "Feature engineering", "32 technical indicators + cross-sectional / regime features", INK),
        ("03", "Forecast ensemble", "Dense · LSTM · Transformer → q10 / q50 / q90 price path", INK),
        ("04", "Sentiment", "FinBERT polarity, importance + recency weighted per channel", INK),
        ("05", "LLM decision gate", "GLM-4.7 → BUY / HOLD / AVOID with target + stop-loss", GOLD),
        ("06", "Entry gates", "deterministic re-check of momentum, volume, stop, reward:risk", INK),
        ("07", "Capital allocation", "vol-targeted Markowitz sizing, conviction-scaled deployment", INK),
    ]
    W = 176
    rh = 15
    H = rh * len(rows) + 4
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    for i, (num, title, desc, col) in enumerate(rows):
        y = i * rh + 2
        cy = y + rh / 2
        heart = col == GOLD
        out.append(f'<rect x="0" y="{y}" width="{W}" height="{rh-2}" fill="{"rgba(181,133,42,0.08)" if heart else "rgba(255,255,255,0.35)"}" stroke="{LINE}" stroke-width="0.4"/>')
        out.append(f'<text x="6" y="{cy+3.2}" font-family="Shoulders" font-weight="700" font-size="11" fill="{col}">{num}</text>')
        out.append(f'<text x="22" y="{cy-0.5}" font-family="Plex" font-weight="700" font-size="5.2" fill="{INK}">{title}</text>')
        out.append(f'<text x="22" y="{cy+5}" font-family="Mono" font-size="3.5" fill="{INK_SOFT}">{desc}</text>')
        if i < len(rows) - 1:
            out.append(f'<line x1="10" y1="{y+rh-2}" x2="10" y2="{y+rh}" stroke="{RULE}" stroke-width="0.5"/>')
    out.append('</svg>')
    return "".join(out)


# ── document body ────────────────────────────────────────────────────────────
BODY = f"""
<div class="cover">
  <div class="frame"></div>
  <div class="tick" style="top:14mm;left:14mm;">NSE · EQUITY · SWING</div>
  <div class="tick" style="top:14mm;right:14mm;">REV. JUL 2026</div>
  <div class="plate">₹</div>
  <div class="cover-tag">Project Report</div>
  <div class="cover-rule"></div>
  <h1>The Signal<br>to <span class="accent">Trade.</span></h1>
  <div class="sub">A machine-learning pipeline that reads the market — fundamentals,
  news, crowd sentiment, and price — and passes its evidence through an LLM gate
  to decide what to buy, what to avoid, and where to set the stop.</div>

  <div class="flowmini">{svg_flow()}</div>

  <div class="meta">
    <div class="block">
      SYSTEM &nbsp;<b>Stock Market ML Automation</b><br>
      SCOPE &nbsp;&nbsp;&nbsp;<b>Indian equities (NSE), swing horizon</b><br>
      UNIVERSE <b>49 NIFTY-50 stocks</b>
    </div>
    <div class="block" style="text-align:right;">
      PREPARED FOR <b>Kingshuk</b><br>
      AUDIENCE <b>Technical + stakeholder</b><br>
      STATUS <b>Live · backtest pending</b>
    </div>
  </div>
</div>

<!-- ═══════════ EXECUTIVE SUMMARY ═══════════ -->
<div class="section">
  <div class="kicker">01 — Executive summary</div>
  <h2>What this system is</h2>
  <p class="lead"><span class="first">A</span>t its core the project answers one
  question every trading day: <em>of the stocks we follow, which are worth
  buying tomorrow, and on what terms?</em> It does not guess. It assembles four
  independent readings of each stock — how the business is doing, what the news
  and the crowd are saying, and where a forecasting ensemble thinks the price is
  headed — and hands that dossier to a large language model that acts as a
  disciplined analyst, returning a <span class="accentword">BUY / HOLD / AVOID</span>
  call with an entry price, a target, a sell day, and a protective stop.</p>

  <p>The design bias throughout is capital preservation before activity. The
  forecasting models never predict a single number; they predict a band of
  outcomes. The LLM is instructed that outputting zero buys across every stock is
  often the correct answer. And a separate deterministic layer re-checks every
  buy the LLM proposes against hard rules learned from trade history, so a good
  narrative can never override bad arithmetic. This is a system built to say
  <em>no</em> cheaply and <em>yes</em> only when the reward genuinely pays for the
  risk.</p>

  <div class="facts">
    <div class="fact"><div class="n">4</div><div class="l">Evidence channels</div></div>
    <div class="fact"><div class="n">32</div><div class="l">Model features</div></div>
    <div class="fact"><div class="n">3</div><div class="l">Ensemble models</div></div>
    <div class="fact"><div class="n">q10<small>·</small>q90</div><div class="l">Forecast band</div></div>
    <div class="fact"><div class="n">16</div><div class="l">Pipeline stages</div></div>
  </div>

  <div class="callout">
    <div class="ttl">The edge, in one line</div>
    <p>Four uncorrelated signals feed one gate, the gate is forced to justify
    every buy against a costed reward-to-risk test, and a rule engine guarantees
    the discipline the prompt only asks for. The advantage is not any single
    model — it is the refusal to act on a thesis that the numbers do not survive.</p>
  </div>
</div>

<!-- ═══════════ SYSTEM AT A GLANCE ═══════════ -->
<div class="section" style="page-break-before: always;">
  <div class="kicker">02 — System at a glance</div>
  <h2>Seven stages, one decision</h2>
  <p>Market data flows left to right through a fixed contract. Each stage depends
  on the one before it; a failure upstream stops the chain rather than letting a
  half-formed signal reach the trading layer. The LLM gate sits at the centre —
  everything above it gathers evidence, everything below it turns a decision into
  a sized, risk-bounded position.</p>

  <div class="pipeline">{svg_pipeline_big()}</div>

  <div class="facts">
    <div class="fact"><div class="n">1<small>d</small></div><div class="l">Data cadence</div></div>
    <div class="fact"><div class="n">15<small>d</small></div><div class="l">Forecast path used by gate</div></div>
    <div class="fact"><div class="n">Mo.</div><div class="l">Incremental fine-tune</div></div>
    <div class="fact"><div class="n">3</div><div class="l">Storage surfaces</div></div>
  </div>
  <p class="small">Storage is split by purpose: real prices and forecasts live in
  <span class="mono">Google Sheets</span>; news, sentiment, fundamentals and the
  final trade suggestions live in <span class="mono">Firebase / Firestore</span>;
  and a local <span class="mono">Parquet</span> archive retains history for
  training and drift monitoring. Orchestration, scheduling (GitHub Actions), a
  FastAPI server, and Docker parity wrap the whole thing.</p>
</div>

<!-- ═══════════ DATA & SIGNAL LAYER ═══════════ -->
<div class="section" style="page-break-before: always;">
  <div class="kicker">03 — Data &amp; signal layer</div>
  <h2>Reading the market four ways</h2>
  <p>A price forecast alone is a thin basis for a trade. The system deliberately
  triangulates, and it keeps the channels <em>separate</em> because they behave
  differently — news lags, Reddit runs contrarian, X is fast and noisy, and a
  combined average would smear those distinct signals into mush.</p>

  <h3>Fundamentals &amp; price</h3>
  <ul class="clean">
    <li><b>OHLCV</b> — daily bars pulled via <span class="mono">yfinance</span>,
    appended to the operational sheet, mirrored to a local store for integrity checks.</li>
    <li><b>Quarterly fundamentals</b> — the last four quarters of revenue, net
    income and related lines per ticker, refreshed weekly and stored one document
    per company.</li>
  </ul>

  <h3>News &amp; social sentiment</h3>
  <ul class="clean">
    <li><b>News</b> — headlines from the last 7 days, article bodies scraped in
    full, mapped to companies by an alias matcher; macro items flow through as a
    market-level bucket.</li>
    <li><b>Reddit &amp; X</b> — posts scraped from NIFTY-focused subreddits and X,
    deduplicated, matched to tickers.</li>
    <li><b>FinBERT scoring</b> — each item is condensed to a polarity in
    <span class="mono">[-1, 1]</span>, then aggregated per ticker and day.</li>
  </ul>

  <div class="callout">
    <div class="ttl">Not every headline counts the same</div>
    <p>Each item's weight is <span class="mono">recency × engagement ×
    source_quality × confidence</span>. A viral post cannot drown out the day
    through log-scaled engagement; trusted outlets outweigh unknown blogs; and
    mushy, neutral text is discounted. Recency decays on a configurable half-life
    (default 3 days), so the latest headline dominates and a week-old one barely
    registers.</p>
  </div>
</div>

<!-- ═══════════ FORECASTING ENGINE ═══════════ -->
<div class="section" style="page-break-before: always;">
  <div class="kicker">04 — Forecasting engine</div>
  <h2>A band, never a point</h2>
  <p>From the engineered features, a three-model PyTorch ensemble produces a
  multi-horizon <em>quantile</em> forecast: for each day ahead it emits the 10th,
  50th and 90th percentile of the log-return, so the output is an explicit
  cone of uncertainty rather than a single deceptive line. Predictions are bounded
  by the exchange's daily circuit limit, so no horizon can imply a physically
  impossible move.</p>

  <div class="facts">
    <div class="fact"><div class="n">Dense</div><div class="l">MLP baseline</div></div>
    <div class="fact"><div class="n">LSTM</div><div class="l">Sequential memory</div></div>
    <div class="fact"><div class="n">Trans.</div><div class="l">Attention encoder</div></div>
    <div class="fact"><div class="n">20</div><div class="l">Day lookback window</div></div>
  </div>

  <h3>Features the models see</h3>
  <p>32 indicators per bar — momentum (RSI, MACD, ROC, Williams %R), trend
  (SMA / EMA families, ADX), volatility (ATR plus a GARCH-style conditional-vol
  family), and volume (OBV, VWAP, MFI) — extended with cross-sectional and regime
  features that place each stock <em>relative</em> to the NIFTY index and the
  India VIX. Forward-return labels for horizons up to 30 days are targets, never
  inputs, and are stripped before inference.</p>

  <div class="callout">
    <div class="ttl">Learning that never forgets the whole book</div>
    <p>The models are never retrained from scratch. A gated monthly fine-tune
    warm-starts the saved checkpoints on only the rows they have never seen —
    triggered by a new calendar month, or early once the median stock accumulates
    ~30 fresh rows — and atomically overwrites the weights in place. Retraining is
    incremental by design: cheap, frequent, and never destabilising.</p>
  </div>
</div>

<!-- ═══════════ THE LLM GATE ═══════════ -->
<div class="section" style="page-break-before: always;">
  <div class="kicker">05 — The decision gate</div>
  <h2>Where evidence becomes a trade</h2>
  <p>The heart of the system. Each stock's 15-day forecast path, its sentiment
  snapshot and its recent fundamentals are assembled into one dossier and sent to
  <span class="mono">GLM-4.7</span> under a strict system prompt. The model is
  told to think like a portfolio manager — capital preservation first, asymmetric
  reward second, activity last — and to pass every candidate buy through five
  gates <em>in order</em>. Fail any gate, and the answer is HOLD or AVOID.</p>

  <table class="gates">
    <tr><th>Gate</th><th>Rule</th></tr>
    <tr><td class="g">1 · Trend</td><td>Price above its close ~15 days ago <b>and</b> the forecast path ends above today. A bullish forecast on a falling stock is catching a knife.</td></tr>
    <tr><td class="g">2 · Entry</td><td>Momentum orderly, not climactic — never a steep oversold slide, never a volume/price spike day (those are distribution, not accumulation).</td></tr>
    <tr><td class="g">3 · Stop</td><td>Stop below recent swing support and at least 2× the typical daily move below entry. A stop inside one day's range is noise, not protection.</td></tr>
    <tr><td class="g">4 · Reward</td><td>After 0.9% round-trip cost, reward must be ≥ 1.5× the risk. If the maths does not clear, there is no trade.</td></tr>
    <tr><td class="g">5 · No chase</td><td>Entry within 3% of the latest close.</td></tr>
  </table>

  <p>Confidence is <em>computed, not felt</em>: it starts at 0.5 and moves on an
  explicit rubric (recent price change, forecast shape, sentiment, fundamentals,
  volatility), and only a score ≥ 0.6 permits a BUY. The model returns a single
  JSON object — action, buy price, sell day, target, stop-loss, confidence, and a
  rationale that must cite each gate. Runs are incremental: a stock is only
  re-sent when its forecast or sentiment actually changes, fingerprinted by a hash.</p>

  <div class="callout">
    <div class="ttl">Discipline the prompt asks for, the code guarantees</div>
    <p>A deterministic <span class="mono">entry_gates</span> layer re-checks every
    BUY against the same rules in plain arithmetic — momentum-and-forecast
    agreement, a 1.5× average-volume ceiling, an RSI window, an ATR stop floor,
    and the costed reward:risk bar. A buy that fails a hard gate is downgraded to
    HOLD with the reason recorded; a too-tight stop is first widened to the ATR
    floor and only rejected if the reward no longer clears the bar. The thresholds
    are drawn from a walk-forward analysis of 668 closed round trips — so the
    rules are earned, not assumed.</p>
  </div>

  <p class="small">A separate position-review prompt manages <em>open</em> trades —
  letting winners run, trailing stops upward, and cutting broken theses — so the
  gate governs both entry and exit.</p>
</div>

<!-- ═══════════ CAPITAL ALLOCATION ═══════════ -->
<div class="section" style="page-break-before: always;">
  <div class="kicker">06 — Capital allocation</div>
  <h2>From a call to a position size</h2>
  <p>A BUY is only the beginning of a decision. The allocation engine turns the
  discrete calls into concrete rupee sizing using a volatility-targeted Markowitz
  optimisation, tilted mildly by the LLM's confidence, and subject to guardrails
  that make a catastrophic drawdown structurally difficult.</p>

  <ul class="clean">
    <li><b>Cash floor</b> — a fraction of equity is always retained; the book can
    never be fully drawn down.</li>
    <li><b>Per-name cap</b> — hard ceiling per stock, shrunk further for volatile
    names via a volatility target.</li>
    <li><b>Conviction-scaled deployment</b> — how much of the book to deploy is not
    fixed; weak-but-still-BUY sets deploy only a floor, ramping to full deployment
    only when the weighted reward:risk is genuinely excellent.</li>
    <li><b>No invented capital</b> — cash never goes negative; a BUY is funded by
    rotating out of an incumbent only if it beats that holding on reward:risk by a
    configured edge, otherwise it is downsized to the cash actually available.</li>
  </ul>

  <div class="tradebar">
    <div class="stop">STOP-LOSS</div>
    <div class="entry">ENTRY</div>
    <div class="tgt">TARGET · ≥ 1.5× RISK</div>
  </div>
  <p class="small">Every position the system advises carries this shape — a stop
  it can survive and a target that, net of costs, pays at least one and a half
  times what it puts at risk.</p>
</div>

<!-- ═══════════ ENGINEERING & OPS ═══════════ -->
<div class="section" style="page-break-before: always;">
  <div class="kicker">07 — Engineering &amp; operations</div>
  <h2>Built to run itself</h2>
  <p>The pipeline is production-shaped, not a notebook. A 16-stage orchestrator
  runs the daily path with a strict contract, hard gates that abort on failure,
  and dry-run as the default so no live sheet is ever touched by accident.</p>

  <h3>Automation surface</h3>
  <ul class="clean">
    <li><b>Scheduled jobs</b> — GitHub Actions for daily data + prediction, weekly
    fundamentals, scheduled news/sentiment, and the monthly gated fine-tune.</li>
    <li><b>Serving</b> — a FastAPI server plus a Google-Sheets polling watcher for
    event-driven runs; Docker Compose for deployment parity.</li>
    <li><b>MLOps</b> — model artifacts versioned and uploaded to GitHub Releases;
    metadata tracked per run.</li>
    <li><b>Tests</b> — a pytest suite guards the contract; the entry-gate maths is
    pure and unit-tested independently of any network.</li>
  </ul>

  <div class="callout">
    <div class="ttl">Backtest — running now, arriving next</div>
    <p>A walk-forward backtest across 2020–2026 is currently in progress and is
    deliberately left out of this edition. It is the layer that will measure
    realised P&amp;L, win rate and drawdown, and validate the same allocation
    engine used in production. This report will be updated with those results once
    the run completes.</p>
  </div>

  <hr class="rule">
  <p class="small" style="text-align:center;">
    <span class="mono">SIGNAL LEDGER</span> &nbsp;·&nbsp; Stock Market ML
    Automation &nbsp;·&nbsp; prepared for Kingshuk &nbsp;·&nbsp; July 2026
    &nbsp;·&nbsp; backtest section pending
  </p>
</div>
"""

HTML_DOC = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{BODY}</body></html>"

OUT = os.environ.get("REPORT_OUT", os.path.join(HERE, "Stock_Market_Project_Report.pdf"))
HTML(string=HTML_DOC, base_url=HERE).write_pdf(OUT)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
