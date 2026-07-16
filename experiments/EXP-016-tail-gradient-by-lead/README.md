# EXP-016: tail-gradient saturation by lead

Status: `completed`

## Question

After fixed-temperature tail-area training, do long-lead predictions fall far
enough below severe thresholds that useful upward gradients become too narrow
or vanish?

This gate tests the mechanism required for a possible second component:
lead-adaptive soft-threshold continuation. It does not authorize that component
unless a wider temperature restores long-lead gradient support while keeping
most upward gradient near observed severe regions.

## Method

On the frozen seed-0 tail-area checkpoint, compare raw-VIL temperatures 5, 10,
20 and 30 at thresholds 160, 181 and 219. For every lead, record:

- predicted and target soft exceedance counts;
- upward and downward gradient mass;
- effective gradient-support fraction;
- fraction of upward gradient lying within 0, 2 and 4 pixels of target severe
  regions.

## Command

```bash
python scripts/probe_tail_gradient_by_lead.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --checkpoint artifacts/local/exp010_bf16_tail_area_seed0/last.pt \
  --output artifacts/local/exp016_tail_gradient_by_lead.json \
  --resolution 128 \
  --batch-size 8 \
  --max-batches 8 \
  --thresholds-raw 160 181 219 \
  --temperatures-raw 5 10 20 30 \
  --radii 0 2 4 \
  --amp-dtype bfloat16 \
  --workers 2
```

## Decision rule

- Continue only if temperature 10 shows deteriorating long-lead gradient
  support and a wider temperature restores it without moving most upward
  gradient outside the radius-4 target envelope.
- Reject adaptive temperature if fixed-temperature gradients remain healthy,
  or if widening mainly activates unrelated background.
- A positive result authorizes one bounded training comparison before any
  multi-seed or native-resolution run.

## Result

The gate rejects wider lead-adaptive temperature as component two.

At thresholds 160 and 181, temperature 10 does not lose upward-gradient mass
or effective support at long lead. From 5 to 60 minutes, upward L1 gradient
increases from 2.06 to 5.11 at threshold 160 and from 1.32 to 3.53 at threshold
181. Gradient saturation therefore does not explain their remaining deficit.

At threshold 219, temperature 5 nearly saturates at 60 minutes, but temperature
10 retains upward gradient (0.492 versus 0.688 at 5 minutes) with similar
effective support. Its localization has nevertheless collapsed: only 1.02% of
the 60-minute upward gradient lies within four pixels of target severe regions.

Temperatures 20 and 30 broaden effective support, but they do not solve this
localization failure. At threshold 219 and 60 minutes, radius-4 fractions are
only 1.99% and 2.06%. Wider temperatures primarily activate background rather
than restore target-local severe support. The same delocalization trend appears
at thresholds 160 and 181.

## Decision

- Reject lead-adaptive temperature widening; no training run is authorized.
- Fixed temperature 10 is not proven optimal, but its failure is not simple
  long-lead gradient saturation at thresholds 160 and 181.
- Threshold 219 requires spatial/representation information capable of
  identifying where an extinct core could reappear. A scalar area objective or
  wider sigmoid alone cannot provide that information.
- Stop adding loss terms until the publication strategy is re-audited. The
  next method, if any, must alter representation or conditioning and must be
  compared with recent refinement and physics/motion-guided methods.

See `result-analysis.json` for the compact numerical record.
