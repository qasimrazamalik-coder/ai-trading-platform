# Look-through Shariah screening of ETFs

## Why funds need their own screen

The Shariah gate blocks any ETF that has no screening ratios (rule R6), so the diversified
core could not trade Shariah ETFs on placeholder numbers. The fix builds fund-level AAOIFI
Standard 21 ratios from the fund's real holdings. Code:
`packages/core/src/quant_os_core/shariah/fund_screen.py`; run it with
`python -m quant_os_core.shariah.fund_screen SPUS HLAL UMMA SPRE SPSK AAAU --user-agent "Name email" --config examples/paper_core.json --write`.

## Look-through method

1. Find the fund's latest N-PORT-P filing on SEC EDGAR and read every holding.
2. For each company held, pull its XBRL figures and SIC code from EDGAR (Yahoo fundamentals,
   looked up by ISIN, are a fallback for holdings that do not file with the SEC; those are
   listed for review).
3. Add up the fund's *share* of each company (value held / company market cap):
   interest-bearing debt and interest-bearing cash and securities, plus any conventional debt
   or cash the fund holds directly; non-compliant revenue is the fund's share of each
   company's interest income plus all revenue of companies in an excluded sector (SIC-mapped:
   banks, insurers, alcohol, tobacco, gambling and similar).
4. Market cap is the fund's equity holdings at market value. The result is one fund-level set
   of financials that the gate screens like a single company: interest-bearing debt < 30% of
   market cap, interest-bearing cash and securities < 30%, non-compliant revenue < 5%,
   tangible assets at least one third of total assets.

It fails closed: if less than 90% of the fund's equity value can be matched to EDGAR data, no
financials are produced and the fund stays blocked. SIC 6798 (all REITs) is not mapped to
conventional finance; mortgage REITs are caught by their interest-income share of revenue.

## Screening results (holdings filed 2026-08-31)

| Fund | Result | Why |
|---|---|---|
| SPUS | PASS | look-through ratios inside every AAOIFI limit |
| HLAL | PASS | look-through ratios inside every AAOIFI limit |
| SPRE | PASS | look-through ratios inside every AAOIFI limit |
| UMMA | blocked | only 74% of its equity value could be matched to SEC data (needs 90%) |
| SPSK | blocked | a sukuk fund holds debt-like certificates, not equities; unscreenable by this method |
| AAAU | blocked | a physical-gold trust files no N-PORT; unscreenable by this method |

The ratios for SPUS, HLAL and SPRE are stored in `examples/paper_core.json` with
`financials_source` "SEC N-PORT 2026-08-31, look-through". UMMA, SPSK and AAAU stay blocked
until the owner decides how to treat them (for example a manual review or the fund's own
Shariah board certificate).

## Keeping screens fresh

Holdings change every quarter. The signal agent raises `SCREEN_STALE` when a fund's financials
are older than 120 days or missing; re-run `fund_screen` with `--write` to refresh them. A
stale screen is information only: the gate still decides on whatever financials are on file.
