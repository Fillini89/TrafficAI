# TrafficAI Holdout Report (full)

Scenarios evaluated: 14
Models compared: Baseline, Gen 10, Gen 11, Gen 12

## Executive Takeaway

Gen 12 vs Gen 11: final waiting +51.5%, 95th-percentile waiting +47.7%, average speed -15.7%.
Full holdout is the main evidence for long-horizon starvation and fairness.

## Scorecard

| Agent | Final wait | P95 wait | Stopped burden | Mean speed | Signal changes |
|---|---:|---:|---:|---:|---:|
| Baseline | 3,291 | 4,497 | 817,867 | 2.70 | 1,735 |
| Gen 10 | 2,539,105 | 2,211,394 | 918,718 | 3.29 | 3,222 |
| Gen 11 | 2,701 | 6,511 | 490,638 | 4.03 | 4,910 |
| Gen 12 | 1,311 | 3,406 | 515,918 | 3.40 | 5,189 |

## Hidden Starvation Check

| Agent | Worst lane wait | P95 worst-lane wait | Max starved lanes | Max starvation excess | Fairness violations |
|---|---:|---:|---:|---:|---:|
| Baseline | 11,343 | 9,412 | 11 | 24,502 | 11,467 |
| Gen 10 | 21,487 | 20,365 | 10 | 94,421 | 10,499 |
| Gen 11 | 12,395 | 6,868 | 10 | 25,039 | 10,279 |
| Gen 12 | 6,539 | 4,111 | 9 | 15,540 | 10,324 |

## Baseline Improvement

| Agent | Final wait | P95 wait | Stopped burden | Mean speed |
|---|---:|---:|---:|---:|
| Baseline | 0.0% | 0.0% | 0.0% | 0.0% |
| Gen 10 | -77042.9% | -49072.5% | -12.3% | +21.8% |
| Gen 11 | +17.9% | -44.8% | +40.0% | +49.4% |
| Gen 12 | +60.2% | +24.3% | +36.9% | +25.9% |
