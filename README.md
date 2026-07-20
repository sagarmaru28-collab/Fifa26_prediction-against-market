# World Cup 2026: ML Forecast vs Prediction Market

A probabilistic forecasting model for the 2026 FIFA World Cup, run at four checkpoints across the tournament and compared against Polymarket's prediction market at each stage. Spain won the tournament. The model had priced Spain above the market at every checkpoint.

## Method

Three components, kept deliberately simple and fully reproducible:

**Elo ratings.** Computed from scratch over ~49,000 international matches (1872-2026) from the [martj42 international results dataset](https://github.com/martj42/international_results). K-factor varies by match importance (World Cup 60, qualifiers and continental championships 40, friendlies 20), with a margin-of-victory multiplier and 80 points of home advantage. Host nations (USA, Canada, Mexico) receive the home bonus in tournament simulation.

**Match model.** Two Poisson regressions (scikit-learn) predict expected goals for each side from the Elo difference (scaled), venue neutrality, and rolling 10-match attack/defense form. Simulated scores handle draws, group tiebreakers, and knockout extra time (draws resolved by an Elo-weighted coin flip standing in for penalties). A gradient boosting classifier (win/draw/loss) is trained alongside for backtest evaluation.

**Monte Carlo simulation.** 10,000-100,000 simulations of the remaining tournament at each checkpoint. Pre-tournament this means the full 48-team format: all 12 groups, tiebreakers, best-eight third-place qualification, and the Round-of-32 bracket. At later checkpoints, actual results replace simulated ones and only the remaining rounds are simulated.

## Backtests

Trained strictly on data available before each tournament:

| Tournament | Log-loss (model vs baseline) | Brier (model vs baseline) |
|---|---|---|
| 2018 World Cup | 0.988 vs 1.096 | 0.587 vs 0.668 |
| 2022 World Cup | 1.085 vs 1.074 | beat baseline |

The model lost to the naive baseline on log-loss in 2022, the tournament of Saudi Arabia over Argentina and Morocco's semifinal run. That result is reported, not hidden. Upset-heavy tournaments humble every model.

## The four checkpoints

Model title probability vs Polymarket price at each stage:

| | Pre-tournament | Post-groups | Post-R16 | Pre-semis | Result |
|---|---|---|---|---|---|
| **Spain** | **20% / 16%** | **15% / 12%** | **22% / 18%** | **27% / 20%** | **Champions** |
| Argentina | 17% / 8% | 26% / 18% | 28% / 17% | 30% / 18% | Runners-up |
| France | 10% / 16% | 17% / 24% | 22% / 33% | 28% / 40% | Fourth |
| England | 6% / 11% | 6% / 13% | 12% / 8% | 15% / 21% | Third |

Format: model % / market %.

## Scorecard

**Model over market.** Spain was the model's pre-tournament favorite and was priced above the market at all four checkpoints. The market's biggest conviction, France at 40% before the semifinals (double Spain's price, driven by Mbappé's Golden Boot run), busted: the model had that semifinal as a 51.5/48.5 coin flip, Spain won it, then beat Argentina 1-0 in the final with an xG of 1.94 to 0.20.

**Market over model.** The model's top pick from the group stage onward was Argentina, who lost the final. The market's skepticism about Argentina (three straight extra-time knockout wins) contained real information about a team winning ugly.

**Individual match calls.** Argentina 61% over England in the semifinal: hit. Brazil 63% over Norway in the Round of 16: missed. Spain vs Argentina was the model's second most likely final (29%).

## Repository structure

```
01_pretournament.py   Full 48-team simulation + 2018/2022 backtests (run Jun 10)
02_postgroup.py       Knockout-only sim seeded with the real R32 bracket (run Jun 28)
03_quarterfinals.py   Final-eight simulation (run Jul 8)
04_semifinals.py      Final-four simulation (run Jul 12)
requirements.txt
```

Each script is self-contained. Download the dataset and run:

```
pip install -r requirements.txt
curl -sL https://raw.githubusercontent.com/martj42/international_results/master/results.csv -o results.csv
python 01_pretournament.py
```

Note that re-running today reproduces the pipeline but not the historical outputs verbatim: the dataset now contains results that postdate each checkpoint. The probabilities each script produced at its run date are recorded in its docstring.

## Known limitations

- The pre-tournament script approximates FIFA's third-place allocation table for the Round of 32; the real allocation is more intricate. Effect on title probabilities is small.
- Knockout bracket progression pairs matches in listed order, approximating the official bracket tree.
- Penalty shootouts are modeled as an Elo-weighted coin flip shrunk toward 50/50.
- The 48-team format had never been played; all pre-tournament forecasts, model and market alike, were extrapolating.
- Elo sees results, not squads. Injuries, suspensions, and a hot striker are invisible to it. That gap versus the market's information set was the entire point of the comparison.

## Disclaimer

This is a forecasting and calibration exercise, not betting advice.

## Data

Match data: [martj42/international_results](https://github.com/martj42/international_results) (CC0). Market prices: Polymarket, recorded manually at each checkpoint.
