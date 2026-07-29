# EXP-026: native-resolution paper SimVP seed-1 replication

Status: `completed`

Current decision: the frozen native-resolution result replicates. The model
configuration is frozen and the one-time GRSL test evaluation is authorized.

## Purpose

Test whether EXP-025's severe-echo benefit survives training randomness under
the identical native 384x384 protocol.

## Frozen pair

- official OpenSTL SimVP IncepU;
- native 384x384, 13 input and 12 output frames;
- full 35,718-window training and 9,060-window validation splits;
- BF16, batch size 1 and seed 1;
- three epochs and 107,154 optimizer updates;
- independent checkpoint selection by validation `mcsi_global`;
- MSE baseline versus MSE plus the frozen tail-area term;
- no test access.

## Result

The baseline selected epoch 2; the method selected epoch 3.

| Metric | Baseline | Tail area | Relative change |
|---|---:|---:|---:|
| validation mCSI global | 0.387278 | 0.407487 | +5.22% |
| validation mCSI lead average | 0.382699 | 0.406921 | +6.33% |
| validation MSE | 0.003425 | 0.003262 | -4.76% |
| validation MAE | 0.027758 | 0.026251 | -5.43% |

Global CSI changes by +0.24%/+1.14%/+4.89%/+12.05%/+20.81%/+21.45% at
raw VIL thresholds 16/74/133/160/181/219. Mean CSI improves at every lead,
from +2.48% at 5 minutes to approximately +10.45% at 60 minutes.

At severe thresholds 160/181/219, POD rises by
19.76%/33.38%/38.50%, while SUCR falls by 10.90%/17.11%/29.07%. At 60
minutes, baseline forecast/observed area ratios are 6.27%, 0.92% and 0%;
the method raises them to 24.30%, 21.42% and 17.61%.

## Replication conclusion

Both native seeds show:

- positive overall and severe-threshold CSI changes;
- increasing benefit with threshold severity and forecast lead;
- nonzero 60-minute VIL-219 forecasts where the baseline predicts none;
- higher severe recall with lower severe success ratio.

Seed 1 also improves MSE and MAE, whereas seed 0 incurred a small MSE cost.
The error tradeoff is therefore seed-dependent and must be reported using the
two-seed mean and sample standard deviation.

The configuration is now frozen. Test evaluation may run exactly once using
the independently selected `best.pt` files; test results cannot trigger
hyperparameter or checkpoint changes.
