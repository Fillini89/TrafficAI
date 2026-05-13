# TrafficAI Holdout Report (full)

Scenarios evaluated: 14
Models compared: Baseline, Gen 12, Gen 13, Gen 14, Gen 15, Gen 16

## Executive Takeaway

Gen 16 vs Gen 15: final waiting +26.7%, 95th-percentile waiting +2.9%, average speed +1.9%.
Full holdout is the main evidence for long-horizon starvation and fairness.

## Scorecard

| Agent | Final wait | P95 wait | Stopped burden | Mean speed | Signal changes |
|---|---:|---:|---:|---:|---:|
| Baseline | 3,228 | 4,500 | 817,386 | 2.70 | 1,735 |
| Gen 12 | 1,317 | 3,429 | 515,967 | 3.40 | 8,029 |
| Gen 13 | 2,596 | 4,158 | 645,538 | 3.29 | 2,061 |
| Gen 14 | 1,286 | 3,526 | 494,837 | 3.68 | 3,110 |
| Gen 15 | 1,790 | 3,543 | 492,589 | 3.71 | 3,244 |
| Gen 16 | 1,312 | 3,440 | 479,275 | 3.78 | 3,230 |

## Hidden Starvation Check

| Agent | Worst lane wait | P95 worst-lane wait | Max starved lanes | Max starvation excess | Fairness violations |
|---|---:|---:|---:|---:|---:|
| Baseline | 11,504 | 9,439 | 11 | 24,365 | 11,445 |
| Gen 12 | 6,474 | 4,180 | 9 | 15,394 | 10,337 |
| Gen 13 | 10,823 | 8,263 | 10 | 22,538 | 10,837 |
| Gen 14 | 8,969 | 5,725 | 9 | 17,816 | 10,291 |
| Gen 15 | 10,172 | 5,995 | 9 | 20,201 | 10,274 |
| Gen 16 | 9,514 | 5,737 | 9 | 17,701 | 10,234 |

## Signal Smoothness Check

| Agent | Actual switches | Policy switches | Seconds between actual switches | Actual switches/hour | Cadence suppressed | Max service age | Avg hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1,735 | 1,735 | 32.0 | 112.4 | 0 | 0.0 | 14.0 |
| Gen 12 | 8,029 | 5,194 | 6.8 | 533.2 | 0 | 0.0 | 6.8 |
| Gen 13 | 2,061 | 5,100 | 26.7 | 134.9 | 6,474 | 140.9 | 12.2 |
| Gen 14 | 3,110 | 5,657 | 17.6 | 204.9 | 4,828 | 145.7 | 9.2 |
| Gen 15 | 3,244 | 5,724 | 16.8 | 213.9 | 4,635 | 196.9 | 9.1 |
| Gen 16 | 3,230 | 5,662 | 16.8 | 214.4 | 4,654 | 207.7 | 9.1 |

## Service-Age Budget Check

| Agent | Max service age | >150s steps | >210s steps | >150s burden |
|---|---:|---:|---:|---:|
| Baseline | 0.0 | 0 | 0 | 0 |
| Gen 12 | 0.0 | 10,054 | 9,627 | 85,559,555 |
| Gen 13 | 140.9 | 0 | 0 | 1 |
| Gen 14 | 145.7 | 2 | 0 | 51 |
| Gen 15 | 196.9 | 49 | 1 | 3,358 |
| Gen 16 | 207.7 | 65 | 4 | 5,373 |

## Baseline Improvement

| Agent | Final wait | P95 wait | Stopped burden | Mean speed |
|---|---:|---:|---:|---:|
| Baseline | 0.0% | 0.0% | 0.0% | 0.0% |
| Gen 12 | +59.2% | +23.8% | +36.9% | +25.7% |
| Gen 13 | +19.6% | +7.6% | +21.0% | +21.8% |
| Gen 14 | +60.1% | +21.7% | +39.5% | +36.3% |
| Gen 15 | +44.5% | +21.3% | +39.7% | +37.4% |
| Gen 16 | +59.4% | +23.6% | +41.4% | +40.0% |
