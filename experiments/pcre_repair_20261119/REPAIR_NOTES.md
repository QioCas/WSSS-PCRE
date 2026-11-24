# PCRE repair: diagnosis and controlled changes

## Observed results
Both original 20k checkpoints were evaluated on all 1,449 VOC val images with identical scales 1.0/1.25/1.5, horizontal flip, float32 and CRF.
Baseline: raw 73.789488%, CRF 74.224284%. Legacy adaptive: raw 72.958834%, CRF 73.444794%.

The regression is mixed: cat/dog lose recall; boat/bicycle/pottedplant lose precision. It cannot be explained as only foreground over-expansion.

## Verified implementation problem and hypothesis
The legacy CAM rescaling multiplies class c by 0.7/tau[c] before softmax and PAR. This changes foreground class rankings, foreground/background competition, and both effective high and low thresholds. Thus it is not an isolated confidence-threshold intervention.
Example: two class CAM scores 0.65 and 0.64 rank class 1 first; thresholds 0.75 and 0.55 rescale them to 0.6067 and 0.8145, reversing the class.
Its global confidence product can lower every class threshold when all CAM activation scores drop equally; CAM activations are not calibrated class probabilities.

Across 20 classes in this one seed, final thresholds correlate with raw precision change at r=+0.673 and recall change at r=-0.671. This supports the mechanism as a hypothesis, not proof that it explains all training differences.
The exported threshold_error_diagnostics.csv and PNG retain the per-class evidence.

## Single-model repair: adaptive_consensus
- Do not rescale CAMs or alter the baseline PAR calls.
- Preserve every baseline foreground and background pseudo-label.
- Only admit an ignored pixel if main CAM class, prototype CAM class and the low-background PAR label agree; prototype score must be at least 0.7.
- Center class-relative thresholds on 0.7 instead of multiplying by global CAM confidence. Bound them to [0.65,0.75]. Only classes with threshold below 0.7 can admit additional pixels.
- Keep the baseline exactly through 8,000 steps, then ramp the threshold adjustment through step 10,000.
- No extra ViT forward, no validation mask or training pixel ground truth enters the admission gate.
- Train fresh for 20,000 iterations, batch 4, seed 0, same lr, crop, CAM scales, validation cadence and optimizer as the paired baseline.

The pipeline performs a full batch/crop smoke run, fresh training and identical full VOC val inference automatically. Old checkpoints are immutable. A worse candidate is not adopted. Training logs record any changed baseline foreground/background labels (both must stay zero) and the admitted-pixel fraction.

## Separate deployment experiment: fixed probability ensemble
The two existing models correct different pixels: after CRF, the adaptive model alone corrects 3,201,193 pixels missed by baseline; baseline alone corrects 3,467,580 pixels missed by adaptive. This motivates an ensemble, but does not prove it improves mIoU.
Use fixed 50:50 averaging of normalized probabilities after each model's unchanged TTA fusion, followed by the same CRF. There are no class-wise weights or validation-fitted thresholds.
This is explicitly a two-model method, with approximately twice the forward compute. Its result is not evidence that the single-model repair improved.

## Status and interpretation
- Fourteen unit tests passed, including old tracker/checkpoint regressions, baseline preservation, padding, absent classes, prototype conflicts and warm-up identity.
- A 4-step integration run with batch 1 completed both training phases and checkpoint saving.
- single_model_v2/supervisor.log and queue_events.jsonl track the batch-4 smoke, 20k training and subsequent full inference.
- ensemble_full/progress.json tracks the fixed ensemble inference; metrics.json and summary.json are written after all 1,449 images finish.

VOC val has already informed method development. Any gain is a development-set result for one seed; independent evaluation and additional seeds would be needed to establish generalization.

## Completed ensemble evaluation

All 1,449 images: ensemble raw 73.879539%, CRF 74.321040%. CRF gain vs baseline +0.096757 percentage points.

Paired whole-image bootstrap (1,000 resamples, seed 0) 95% percentile interval for the CRF gain: [-0.3287236414023553, 0.5455073193430897]. This includes zero, excludes training-seed variation and is only a development-val diagnostic.

The two-model ensemble is not adopted as a convincing improvement: gain is small, uncertainty includes no gain, and forward compute approximately doubles. Keep the original baseline while evaluating the single-model repair.

## Runtime investigation and replacement attempt
The first full repair attempt (`single_model`) produced no first 100-step log after more than 10 minutes, although short tests completed. It was stopped without deleting its files; it had no completed full checkpoint. The exact stall cause was not established.
For the replacement (`single_model_v2`), the repair mode uses a direct single-device module wrapper, preserving the `module.` checkpoint key prefix. Single-rank all-reduce/all-gather identities are skipped. A CPU Gloo world-size-1 comparison with alternating unused heads and AdamW produced bitwise-equal parameter updates over three steps. This check is not a full-network or GPU-equivalence claim.
The integration smoke now exercises both classification-only warm-up and full segmentation loss at batch 4. It completed all four steps and saved a paired checkpoint. Step-level progress and a faulthandler signal handler make future stalls inspectable without elevated attach permissions.
The replacement is a fresh 20k run, not a continuation of the interrupted attempt. After training, full VOC val inference runs automatically via `tools/infer_voc_run.py`; `single_model_v2/inference/summary.json` and `single_model_v2/decision.json` are produced at completion.

## Completed single-model evaluation

Fresh 20k adaptive_consensus: raw 73.146305%, CRF 73.718852%. CRF delta vs baseline -0.505432 pp; delta vs legacy adaptive +0.274058 pp. Retain baseline. The final policy fixes only a local pseudo-label invariant, not the learned model's final output.

Important attribution limit: at 5k, before the consensus gate activates at 8k, the repair run already scored 50.9592% vs baseline 51.9176% in training-time validation. Runtime/wrapper and stochastic numerical differences have not been isolated; do not attribute the whole final difference solely to the gate.
