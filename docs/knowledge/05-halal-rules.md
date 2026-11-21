# Halal trading rules used by the system

Summary of the rulings behind `quant_os_core.shariah`. This is not a fatwa; the owner should
confirm the rules with a mufti they follow.

## Blocked in every profile

| Product | Ruling and source |
|---|---|
| Futures and forwards (stocks, commodities, oil, currencies, crypto) | IIFA Resolution 63 (1/7), 1992: postponement of both counter-values; AAOIFI Shariah Standard 1 forbids forward and futures currency deals even for hedging; AAOIFI SS21 3/12-3/14 |
| Options, including binary options | IIFA Resolution 63; binary options are treated as gambling (qimar) and gross uncertainty (gharar) by IslamWeb and Hanafi fatwa services |
| Index trading (the index number itself) | IIFA Resolution 63 calls it gambling |
| CFDs and leveraged retail forex | No ownership, interest-based overnight swaps |
| Margin and short selling | Interest-bearing loans; selling what one does not own |
| Perpetual contracts and leveraged tokens | Futures with funding payments; derivatives underneath |
| Prediction markets and betting | Gambling |

## Permitted with conditions

- **Shares** of companies whose main business is halal and which pass AAOIFI SS21 ratios:
  interest-bearing debt < 30% of market cap, interest-bearing cash and securities < 30%,
  non-compliant revenue < 5%, tangible assets at least one third. Dividends and gains are
  purified by giving away the non-compliant share: non-compliant income / shares outstanding
  x shares held. In Pakistan, the KMI All-Share index list is the screened universe.
- **Spot currency exchange** with immediate possession of both sides (AAOIFI SS1). Retail online
  forex with leverage is not this. Pakistani residents may not use offshore forex/CFD platforms
  (SBP FE circular 08/2022), so forex stays paper-only in the code.
- **Physical gold and silver** bought outright with immediate delivery.
- **Sukuk** and screened halal ETFs and funds.

## Disputed - owner's choice

- **Spot crypto.** Darul Uloom Karachi's fatwa of 10 June 2026 holds crypto is not *maal*, so
  dealing in it is impermissible; Saylani Welfare's scholars hold it permissible; Pakistan's
  regulator PVARA asked for asset categories to be judged separately. The owner enabled spot
  crypto. The token screen still blocks lending/yield tokens, interest-bearing stablecoins,
  gambling, leveraged tokens, derivatives, meme coins and tokens with no utility.
- **Day trading shares.** Many muftis allow same-day resale of a compliant share; the stricter
  Hanafi view requires settlement (possession) before resale. The owner enabled day trading;
  `allow_pre_settlement_resale=False` switches to the strict view.

## Why futures stay blocked

The owner asked twice to add futures if they are halal. Checked against IIFA Resolution 63 and
AAOIFI Standards 1 and 21: they are not. The only permitted forward-type sale is *salam* (full
price paid now, a defined commodity delivered later), which retail futures markets do not offer.
