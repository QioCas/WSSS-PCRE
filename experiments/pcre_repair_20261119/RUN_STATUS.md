# PCRE repair: completed evaluation

The fresh adaptive_consensus run completed all 20,000 iterations. Its automatic full VOC val inference completed on 2026-11-19 at 15:25 UTC. All results below use 1,449 images, identical multi-scale/flip inference and CRF.

| Method | Raw mIoU (%) | CRF mIoU (%) | CRF delta vs baseline (pp) |
|---|---:|---:|---:|
| Baseline | 73.7895 | 74.2243 | +0.0000 |
| Legacy adaptive | 72.9588 | 73.4448 | -0.7795 |
| Fixed two-model ensemble | 73.8795 | 74.3210 | +0.0968 |
| Repaired adaptive_consensus | 73.1463 | 73.7189 | -0.5054 |

Retained single-model choice: **baseline**. The repair improves the legacy adaptive result but remains below baseline. The two-model ensemble gain is small, its paired image-bootstrap interval includes zero, and its forward compute approximately doubles.

Completed results: `single_model_v2/inference/summary.json`, `single_model_v2/inference/adaptive_consensus/metrics.json`, and `single_model_v2/decision.json`.
The old `single_model` attempt was stopped for diagnosis; `single_model_v2` is the completed replacement.

No training job in this repair pipeline remains active. These are one-seed development-validation results, not independent-test or multi-seed evidence.
