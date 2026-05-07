# TrafficAI Holdout Report (quick)

Scenarios evaluated: 4
Models compared: Baseline, Gen 9, Gen 10, Gen 11

## Executive Takeaway

Gen 11 vs Gen 10: final waiting +6.0%, 95th-percentile waiting -32.2%, average speed -4.5%.
Quick holdout is a regression screen, not final fairness proof. Full daily evaluation remains decisive.

## Scorecard

| Agent | Final wait | P95 wait | Stopped burden | Mean speed | Signal changes |
|---|---:|---:|---:|---:|---:|
| Baseline | 2,503 | 2,483 | 134,626 | 3.80 | 449 |
| Gen 9 | 2,247 | 3,183 | 100,003 | 5.19 | 1,252 |
| Gen 10 | 1,630 | 2,266 | 84,498 | 5.75 | 1,290 |
| Gen 11 | 1,532 | 2,996 | 88,168 | 5.49 | 1,296 |

## Baseline Improvement

| Agent | Final wait | P95 wait | Stopped burden | Mean speed |
|---|---:|---:|---:|---:|
| Baseline | 0.0% | 0.0% | 0.0% | 0.0% |
| Gen 9 | +10.2% | -28.2% | +25.7% | +36.7% |
| Gen 10 | +34.9% | +8.7% | +37.2% | +51.5% |
| Gen 11 | +38.8% | -20.7% | +34.5% | +44.7% |
