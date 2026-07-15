# Problem-first research roadmap

## Why the previous direction was weak

Adding attention, SSIM/F1-style mixed losses, replacing Inception with Mamba, or importing an LLM post-training idea starts from a tool. It does not establish which SEVIR failure requires that tool. A stronger paper must begin with a repeatable empirical contradiction.

## Feasibility constraints

Until the baseline is stable, a candidate project must satisfy all of the following:

- use SEVIR VIL only;
- reuse a maintained SimVP implementation;
- train at 128 × 128 on one A4000;
- require no manual labels, large teacher or external foundation model;
- add at most one independently testable change;
- have a small-data stop test before a full run;
- preserve inference cost unless efficiency is the research question.

## Candidate questions, not committed methods

### Q1. High-intensity attenuation

Does MSE-trained SimVP systematically underestimate high VIL because zero and weak pixels dominate the optimization, or because high-intensity futures are genuinely less predictable?

First evidence required:

- intensity-bin bias and MAE by lead time;
- gradient contribution by intensity bin;
- comparison with a simple persistence baseline;
- repeatability across seeds.

Only after this is established should tail-aware, ordinal or sampling-based solutions be considered.

### Q2. Horizon conflict

Do near- and far-lead-time losses produce conflicting gradients in the shared SimVP translator, causing one-shot multi-frame training to compromise both regimes?

First evidence required:

- cosine similarity between per-horizon gradients;
- performance of short-only, long-only and joint models under matched budgets;
- conflict frequency over training and event intensity.

Only after conflict is demonstrated should gradient balancing or horizon-specific lightweight heads be considered.

### Q3. Active-region gradient dilution

Do large zero/background areas dilute the learning signal from active storm regions beyond what intensity weighting alone can correct?

First evidence required:

- foreground fraction distribution;
- metric and gradient contribution from background versus active regions;
- controlled crop or sampler experiment without changing the model.

Only after the sampler alone shows value should a patch-aware training method be developed.

### Q4. Event-distribution mismatch

Does the standard training set underrepresent the event types that dominate high-threshold test errors?

First evidence required:

- train/test distributions of maximum VIL, active area, growth rate and motion proxy;
- error concentration by automatically computed event attributes;
- comparison between event-uniform and window-uniform sampling.

Only after a stable mismatch is found should event-aware sampling or robust optimization be attempted.

## Initial decision order

1. Establish a reproducible baseline and environment.
2. Run inexpensive offline dataset statistics for Q1, Q3 and Q4.
3. Run a short baseline training and collect per-horizon diagnostics for Q1 and Q2.
4. Select exactly one question whose evidence is strongest.
5. Write the method only after recording the null and stop conditions.

## Stop rule

If a claimed failure is not stable across at least three seeds or disappears under a matched simple control, it is not a suitable thesis problem.

