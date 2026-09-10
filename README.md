# TPUS

TPUS is research code for joint 3D segmentation of heterogeneous medical-image datasets with one model. This repository contains the implementation snapshot used for the TPUS method audit; it does not assign an expansion to the project name, claim a publication status, or report experimental results.

The supported main trainer is:

```text
MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems
```

It combines:

- dataset-specific input stems;
- a shared MultiTalent encoder-decoder with NexToU Pool GNN and Swin GNN blocks;
- frozen, raw OpenAI CLIP ViT-B/32 text embeddings for dataset-specific class prompts;
- a shared controller that generates per-class dynamic `1x1x1` convolution parameters;
- a dynamic full-resolution segmentation head on the final decoder feature;
- static dataset-specific auxiliary heads for deep supervision; and
- Dice + cross-entropy + topology-interaction (TI) loss.

No medical images, case identifiers, dataset split lists, checkpoints, predictions, or paper result tables are distributed here.

## Method scope

For each sample, the dataset ID selects one input stem and the corresponding class vocabulary. All features after the stem use the shared GNN encoder-decoder. The dynamic primary head receives:

1. the shared bottleneck feature, transformed by GroupNorm, ReLU, global average pooling, and a `320 -> 256` projection; and
2. one frozen raw CLIP text vector per class, transformed by a shared `512 -> 256` projection.

The two 256-dimensional vectors are concatenated and passed through a shared controller. It emits 153 parameters per class for an `8 -> 8 -> 8 -> 1` sequence of dynamic `1x1x1` convolutions. Those convolutions act on an `8`-channel projection of the final decoder feature, not on precomputed logits. The full output order during deep-supervision training is:

```text
[dynamic D4, static D3, static D2, static D1, static D0]
```

Dataset 801 has one foreground class, so its pairwise TI exclusion list is empty and its topology term is zero. Dataset 802 has four foreground classes and six pairwise exclusion constraints. See [docs/METHOD.md](docs/METHOD.md) for the code-level mapping and limitations.

## Repository layout

```text
multitalent/                       MultiTalent framework and TPUS model/trainer code
configs/                           Task 850 plan, prompt text, and portable environment templates
scripts/                           Prompt-cache, architecture-audit, synthetic-data, and metric tools
tests/                             Synthetic unit and smoke tests
examples/synthetic_data/           Documentation for generated, non-medical test data
docs/                              Method, data, and reproducibility notes
LICENSES/                          Licenses for incorporated or referenced upstream components
THIRD_PARTY_NOTICES.md             Source and modification inventory
```

## Installation

Python 3.10 or newer is required. The release was validated with Python 3.11.15 and PyTorch 2.6.0+cu124. Install a PyTorch build appropriate for your CUDA driver first, then install TPUS and the pinned OpenAI CLIP source:

```bash
git clone https://github.com/YonghengSun1997/TPUS.git
cd TPUS

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the correct PyTorch build for your platform first.
python -m pip install -e '.[clip]'
```

The `clip` extra pins OpenAI CLIP commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. The first ViT-B/32 prompt-cache generation downloads the public CLIP checkpoint. Its expected SHA-256 is:

```text
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
```

Versions observed in the validation environment are listed in [requirements-tested.txt](requirements-tested.txt). That file is an environment record, not a universal lock file.

## Data format

TPUS follows the nnU-Net v2 raw-data layout:

```text
nnUNet_raw/
  Dataset801_UterUS/
    dataset.json
    imagesTr/<case>_0000.nii.gz
    labelsTr/<case>.nii.gz
    imagesTs/<case>_0000.nii.gz      # optional
    labelsTs/<case>.nii.gz           # optional, evaluation only
  Dataset802_UMDMyomaMRI/
    ...
```

The expected experimental schemas are:

| Dataset | Input channel | Integer labels |
| --- | --- | --- |
| `Dataset801_UterUS` | `0: US` | `0: background`, `1: uterus` |
| `Dataset802_UMDMyomaMRI` | `0: T2_MRI` | `0: background`, `1: uterine_wall`, `2: uterine_cavity`, `3: myoma`, `4: nabothian_cyst` |

Input images and labels must be spatially aligned NIfTI volumes. The release does not redistribute the datasets or assert data-use rights. Obtain every dataset from its authorized provider and follow its license or data-use agreement. Do not place protected health information, direct identifiers, or private split files in a public clone.

See [docs/DATA.md](docs/DATA.md) for complete `dataset.json` templates, naming rules, split behavior, and the synthetic test-data generator.

## Paths

Set the three nnU-Net paths before preprocessing, training, or inference:

```bash
export TPUS_DATA_ROOT=/path/to/your/tpus_data
export nnUNet_raw="$TPUS_DATA_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$TPUS_DATA_ROOT/nnUNet_preprocessed"
export nnUNet_results="$TPUS_DATA_ROOT/nnUNet_results"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

Alternatively, inspect and source [configs/paths.env.example](configs/paths.env.example) after setting `TPUS_DATA_ROOT`.

## Preprocessing

The Task 850 configuration combines source datasets 801 and 802. The checked configuration uses 1 mm isotropic target spacing, a `192 x 192 x 192` 3D patch, z-score normalization, and batch size 1.

```bash
tpus_prepare UterusMT_Task509_UMD 850 \
  -d 801 802 \
  -p "$PWD/configs/nnUNetResEncUNetLPlansIso1x1x1.json" \
  -batch_size 1 \
  --verify_dataset_integrity
```

This creates `Dataset850_UterusMT_Task509_UMD` metadata under `nnUNet_preprocessed`, writes compatible per-dataset plans, and preprocesses both source datasets. Add `--omit_preprocessing` only when compatible preprocessed data already exist.

The MultiTalent loader samples cases from the combined source datasets. Its dataset-level probability is proportional to the square root of each source dataset's case count. With the observed source counts, this does not mean alternating datasets 1:1.

### Data splits

Each source dataset uses its own `splits_final.json` from its preprocessed directory. If that file is absent, the trainer creates deterministic five-fold splits using seed 12345. Exact experimental case lists are not present in the selected source snapshot and therefore are not distributed here. A newly generated deterministic split must not be described as the original experimental split unless its case list is verified against the original record.

## CLIP prompts

Load the checked prompt configuration and generate the raw, non-normalized CLIP embedding cache:

```bash
source configs/task850_clipdynfeature.env

python scripts/generate_task850_raw_clip_prompt_cache.py \
  --prompts "$MT_CLIP_PROMPT_TEXTS_JSON" \
  --output "$MT_FAITHFUL_CLIP_PROMPT_CACHE" \
  --model "$MT_CLIP_PROMPT_MODEL" \
  --device "$MT_CLIP_PROMPT_DEVICE"
```

The environment-variable name `MT_FAITHFUL_CLIP_PROMPT_CACHE` is retained for checkpoint and configuration compatibility. It is not a claim of method equivalence to another implementation.

## Training

Train fold 0 with the main TPUS trainer:

```bash
source configs/task850_clipdynfeature.env

tpus_train 850 3d_fullres 0 \
  -p nnUNetResEncUNetLPlansIso1x1x1 \
  -tr MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems \
  -num_gpus 1
```

The inherited default schedule is 1000 epochs, 250 training iterations and 50 validation iterations per epoch, initial learning rate `1e-3`, weight decay `3e-5`, foreground oversampling fraction `0.33`, gradient clipping at norm 12, and a checkpoint every 50 epochs. The trainer writes `checkpoint_latest.pth`, `checkpoint_best.pth`, and `checkpoint_final.pth` according to the inherited MultiTalent checkpoint rules.

The `*_1ep` trainer is supplied only for pipeline smoke tests; it is not a paper training setting.

## Inference

Run inference separately for each target dataset because each input folder has its own modality and label vocabulary. The `--multichannel` flag is required even when each current dataset has one channel: it tells model construction to restore the dataset-ID-to-input-channel mapping needed by the separate stems.

```bash
MODEL_FOLDER="${nnUNet_results}/Dataset850_UterusMT_Task509_UMD/MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems__nnUNetResEncUNetLPlansIso1x1x1__3d_fullres"

tpus_predict \
  -i "$nnUNet_raw/Dataset801_UterUS/imagesTs" \
  -o predictions/uterus_us \
  -m "$MODEL_FOLDER" \
  -f 0 \
  -target_id 801 \
  --multichannel \
  -chk checkpoint_final.pth

tpus_predict \
  -i "$nnUNet_raw/Dataset802_UMDMyomaMRI/imagesTs" \
  -o predictions/umd_mri \
  -m "$MODEL_FOLDER" \
  -f 0 \
  -target_id 802 \
  --multichannel \
  -chk checkpoint_final.pth
```

Predictions are placed in an ID subdirectory (`.../801` or `.../802`). The checkpoint name is explicit above; change it only when intentionally evaluating another checkpoint.

## Evaluation

The inherited evaluator reports overlap metrics in `summary.json`:

```bash
tpus_evaluate \
  "$nnUNet_raw/Dataset801_UterUS/labelsTs" \
  predictions/uterus_us/801 \
  -djfile "$MODEL_FOLDER/801_dataset.json" \
  -pfile "$MODEL_FOLDER/801_plans.json" \
  -o predictions/uterus_us/metrics.json
```

For Dice, Jaccard, ASSD, HD95, precision, and recall with explicit empty-mask handling, use:

```bash
python scripts/evaluate_segmentation.py \
  --reference "$nnUNet_raw/Dataset801_UterUS/labelsTs" \
  --prediction predictions/uterus_us/801 \
  --labels 1 \
  --output predictions/uterus_us/extended_metrics.json
```

Distance metrics are in physical units from image spacing. See the script help and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before comparing aggregates across studies.

## Verification

Lightweight tests use only synthetic tensors and generated non-medical volumes:

```bash
python -m pytest -q
python scripts/generate_synthetic_datasets.py --output-root /tmp/tpus_synthetic
```

The architecture audit can use either the included plan plus synthetic dataset JSON files or an authorized prepared-data directory:

```bash
python scripts/audit_task850_clipdynfeature_architecture.py \
  --plans configs/nnUNetResEncUNetLPlansIso1x1x1.json \
  --dataset-json 801=/path/to/Dataset801_UterUS/dataset.json \
  --dataset-json 802=/path/to/Dataset802_UMDMyomaMRI/dataset.json \
  --prompt-texts configs/task850_prompt_texts.json \
  --prompt-cache /path/to/task850_clip_prompt_cache_raw_vitb32.pt \
  --output architecture_audit.json
```

The selected source snapshot contains no checkpoint and no completed full training run for this exact trainer. Passing the included smoke tests verifies interfaces, tensor shapes, loss/gradient flow, prompt-cache handling, trainer discovery, and architecture invariants; it does not establish medical accuracy or reproduce a paper result.

## Upstream code and licensing

TPUS-original contributions are released under Apache License 2.0, as stated in [LICENSE](LICENSE). Third-party and adapted components retain their own terms and are not relicensed by the repository-level license. In particular, the retained MAE-derived utility is noncommercial, and the CLIP-Driven-derived adaptations are published under separate written permission confirmed by the TPUS maintainer. The permission text is not distributed here and does not amend the upstream public license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES](LICENSES/) before using or redistributing the code.

Citation metadata for TPUS is intentionally omitted until verified publication information is supplied.

## Acknowledgements

Please cite the original projects and papers appropriate to the components and datasets you use:

- [MultiTalent](https://github.com/MIC-DKFZ/MultiTalent/tree/public_MT)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [NexToU](https://github.com/PengchengShi1220/NexToU)
- [Vision GNN](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [CLIP-Driven Universal Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model)

The links above identify upstream projects; they do not imply endorsement of TPUS.
