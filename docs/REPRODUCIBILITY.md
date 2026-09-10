# Reproducibility notes

## What is fixed in this snapshot

- main trainer class and network construction;
- Task 850 architecture plan and batch size;
- dataset IDs, input-channel counts, and label vocabularies;
- exact prompt strings;
- raw, non-normalized CLIP text-feature convention;
- joint sampling rule;
- TI loss construction and default weight;
- inherited optimizer, schedule, deep-supervision, and checkpoint defaults;
- training, prediction, and evaluation entry points.

## What is not distributed

- source medical images and labels;
- subject or case identifiers;
- exact historical `splits_final.json` files;
- CLIP checkpoint or generated prompt cache;
- trained TPUS checkpoints;
- predictions, training logs, or reported result tables;
- a complete exported personal Conda environment.

The absence of those artifacts means this repository alone cannot reproduce a numerical paper result. It can reproduce the released method implementation and prepare a newly authorized experiment under the documented contract.

## Randomness

When a split file is absent, split generation uses seed 12345. Training still contains stochastic augmentation, case sampling, CUDA kernels, and cuDNN behavior. The inherited training entry point enables cuDNN benchmarking and does not request deterministic kernels. Record all seeds, hardware, software versions, generated splits, and checkpoint hashes for a new experiment.

## Checkpoints

The inherited trainer saves:

- `checkpoint_latest.pth` every 50 epochs and for continuation;
- `checkpoint_best.pth` when the validation EMA improves; and
- `checkpoint_final.pth` at training completion.

Inference defaults to `checkpoint_final.pth`. Always pass `-chk` explicitly in reported evaluations. `--val_best` writes into the same validation directory as final-checkpoint validation, so use separate copied output folders or provenance records if both are evaluated.

## Extended metric definitions

`scripts/evaluate_segmentation.py` computes metrics per case and foreground label:

- Dice: `2TP / (2TP + FP + FN)`;
- Jaccard: `TP / (TP + FP + FN)`;
- precision: `TP / (TP + FP)`;
- recall: `TP / (TP + FN)`;
- ASSD: symmetric average surface distance;
- HD95: symmetric 95th-percentile Hausdorff distance.

Distance metrics use physical spacing from the reference image. If both masks are empty, overlap, precision, and recall are 1 and distances are 0. If exactly one mask is empty, overlap, precision, and recall are 0 and distances are represented as `null` with status `one_empty`; those undefined distances are excluded from finite means and counted separately.

Macro values average all available case-label observations equally. If a study uses case means, class means, foreground-union metrics, or another empty-mask policy, its aggregate is not directly comparable.

## Validation levels

The release distinguishes:

1. syntax and import checks;
2. synthetic dynamic-head forward/backward and checkpoint tests;
3. topology-loss behavior and gradients;
4. trainer discovery and architecture invariant checks;
5. source-to-release output and gradient comparison under identical synthetic inputs;
6. full training and medical test-set evaluation.

Only completed checks should be cited as evidence. A successful import or single synthetic forward pass does not establish segmentation quality or methodological equivalence to an upstream paper.
