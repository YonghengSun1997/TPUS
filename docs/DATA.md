# Data contract

TPUS uses nnU-Net v2 directory and filename conventions. This repository does not contain real medical data, labels, subject identifiers, or experimental split lists.

## Directory layout

```text
${nnUNet_raw}/
  Dataset801_UterUS/
    dataset.json
    imagesTr/
      CASE_ID_0000.nii.gz
    labelsTr/
      CASE_ID.nii.gz
    imagesTs/                 # optional inference set
      CASE_ID_0000.nii.gz
    labelsTs/                 # optional evaluation reference
      CASE_ID.nii.gz
  Dataset802_UMDMyomaMRI/
    ...
```

Do not use patient names, medical-record numbers, dates, or other direct identifiers as `CASE_ID`. Files in `labelsTs` are consumed only by evaluation tools; inference does not require them.

## Dataset 801 schema

```json
{
  "channel_names": {"0": "US"},
  "labels": {"background": 0, "uterus": 1},
  "numTraining": 138,
  "file_ending": ".nii.gz",
  "overwrite_image_reader_writer": "SimpleITKIO"
}
```

The `numTraining` value must be changed if a reader prepares a different authorized cohort. It is shown here only to document the checked experiment schema.

## Dataset 802 schema

```json
{
  "channel_names": {"0": "T2_MRI"},
  "labels": {
    "background": 0,
    "uterine_wall": 1,
    "uterine_cavity": 2,
    "myoma": 3,
    "nabothian_cyst": 4
  },
  "numTraining": 266,
  "file_ending": ".nii.gz",
  "overwrite_image_reader_writer": "SimpleITKIO"
}
```

Labels must be integer-valued and spatially aligned to the corresponding image. Dataset 802 is a multiclass task, not four independent binary masks.

## Combined training metadata

`tpus_prepare UterusMT_Task509_UMD 850 -d 801 802 ...` creates:

```text
${nnUNet_preprocessed}/Dataset850_UterusMT_Task509_UMD/
  datasets.json
  nnUNetResEncUNetLPlansIso1x1x1.json
```

The source datasets remain separately preprocessed. The combined directory records which dataset IDs belong to the joint training.

## Splits

For each source dataset, the trainer first looks for `splits_final.json` in the source dataset's preprocessed directory. If no file exists, it generates five folds with seed 12345. The selected release source did not contain the original case-level split files, so they are deliberately absent from the public repository.

To publish a split for an authorized public dataset, release a script or de-identified manifest only when the dataset terms allow it. Record the generation rule and hash. Do not replace a missing historical split with a newly generated list and call it the original split.

## Synthetic example

Generate two small, clearly synthetic datasets for file-layout and I/O tests:

```bash
python scripts/generate_synthetic_datasets.py --output-root /tmp/tpus_synthetic
```

The output uses IDs 901 and 902 to avoid colliding with real experiment folders. Its ellipsoids and intensity patterns are generated mathematically and are not derived from a patient image. It is unsuitable for accuracy claims.

## Data acquisition

No authoritative download URL or redistribution grant for Dataset 801 or Dataset 802 was stored in the selected source snapshot. Readers must obtain data directly from the respective authorized provider and independently verify access, consent, license, and publication conditions. The code does not bypass restricted access.
