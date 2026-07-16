# EXP-016: tail-gradient saturation by lead

Status: `planned`

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
