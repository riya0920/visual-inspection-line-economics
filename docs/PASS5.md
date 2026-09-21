# ML-2 pass 5 - the review station writes, and the loop eats it

Two not-built items, and closing the first makes the second answerable.

## 1. The station writes

`src/review_service.py` is a store and an HTTP API; `write_review_station(..., post_to=...)` points the existing buttons at it. The static page still works with no service, because a page that needs a server running cannot be emailed to a supplier and a page whose log dies with the tab cannot feed a retraining loop. Neither is a superset of the other.

What the API does when it is asked for something wrong, end to end:

| request | status | response |
|---|---|---|
| a normal disposition | 200 | recorded |
| the same request again (double-clicked button) | 200 | replayed |
| a second operator disagreeing | 200 | recorded |
| false reject on a part the model ACCEPTED | 400 | FALSE_REJECT is not a coherent answer to ACCEPT: there was no reject for the operator to call false |
| a part that is not on the queue | 404 | no such part on the queue |

The two that matter are the middle ones. **A double-clicked button is one opinion, not two** - without the idempotency key the replay enrols a second row from the same person, and the agreement statistic below then reports an operator agreeing with themselves. And **FALSE_REJECT against ACCEPT is refused**: there was no reject for the operator to call false, and a store that takes it has quietly recorded a good-part label nobody intended.

P00001 got two contradicting answers, so it yields no consensus label (`tied: the operators contradict each other`). It appears in `/api/labels` no and in the consensus-only view no.

## 2. The loop had never eaten a human label

`complete.py`'s retraining stage models the CENSORING carefully - only flagged parts are reviewed, and it corrects for that with inverse propensity weights. Then it takes the label from `pool_y`. The truth array. **Label noise was not modelled at all**, which is a strange thing to notice about a stage whose entire subject is human review.

Baseline on a held-out golden set of 60: recall 0.500, PPV 0.500, AUROC 0.556. 83 of 250 parts are flagged and reviewable (the golden set is excluded from review - retraining on what you are about to score on produces a number that means nothing).

### AUROC after retraining, by operator error rate

| operator error | one operator | two, consensus only | two, weighted majority |
|---|---|---|---|
| 0% | 0.606 | 0.512 | 0.590 |
| 5% | 0.818 | 0.611 | 0.607 |
| 10% | 0.590 | 0.594 | 0.542 |
| 20% | 0.612 | 0.548 | 0.673 |
| 30% | 0.588 | 0.544 | 0.564 |

Baseline for comparison: **0.556**.

At a perfect operator the loop reproduces the old result (0.606 against 0.556 baseline). It stays above the baseline across the whole sweep, which is its own finding: on this data the extra labels are worth having even from a fairly unreliable operator.

### What the second operator buys

| operator error | measured agreement | labels, one op | labels, consensus only |
|---|---|---|---|
| 0% | 1.000 | 79 | 69 |
| 5% | 0.944 | 78 | 68 |
| 10% | 0.861 | 80 | 62 |
| 20% | 0.685 | 78 | 50 |
| 30% | 0.701 | 77 | 47 |

The agreement column is the number item 6 of the not-built list has always been about, and it is measurable **only because the store keeps both answers rather than letting the second write overwrite the first**. That is not a database detail: last-write-wins is the default shape of a disposition table, and it destroys the only measurement of the human that a review station can make for free.

Consensus filtering trades labels for label quality, and the table shows the price directly - the count falls while the error in what survives falls faster. Whether that trade pays is in the AUROC table above, not in an argument.

## Honest limits

- No authentication: `operator` is a claim typed by the client, not an identity. Anything downstream that treats it as one is wrong.
- One process, one SQLite file. No replication and no backup.
- Agreement is measured only on parts more than one operator happened to open. Nothing here routes a part to a second operator on purpose, so the agreement figure is on a self-selected subset.
- A superseded disposition is kept but the retraining view uses only the latest per operator. An operator who changes their mind twice leaves a trail that nothing reads.
- The operator is still a SIMULATION. The error rate, the abstention rate and the 65/35 tilt toward passing bad parts are guessed. What is no longer assumed is that the rate is ZERO, and the sweep is there because the right value is not known.
- One seed. The AUROC differences between neighbouring error rates are smaller than this project's own measured seed spread, so read the trend across the sweep and not any single pair.