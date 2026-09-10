# Method implementation

This document maps the supported TPUS trainer to the released code. It describes implementation behavior only and does not claim a paper title or experimental result.

## Main trainer

`MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems` is defined in:

```text
multitalent/training/nnUNetTrainer/project_specific/MultiTalent_native/
  MultiTalent_clip_driven_feature.py
```

The trainer declares five method switches:

```python
prompt_style = "clip"
topology_loss_kind = "ti"
use_nextou_gnn = True
enable_dynamic_conv = True
stem_mode = "dataset_specific"
```

These attributes are metadata; the concrete behavior is established by the class's architecture builder, the multistem initializer, and the inherited TI-loss builder.

## Forward path

### Dataset-specific stems

`MultiTalentNexToUEncoder` stores the stems in an `nn.ModuleDict`, keyed by dataset ID. In `forward`, every sample is passed through the stem selected by its ID before the batch is concatenated. Later encoder and decoder stages are shared.

Relevant files:

```text
multitalent/network_architecture/nextou/multitalent_nextou.py
multitalent/training/nnUNetTrainer/project_specific/MultiTalent_native/
  MultiTalent_multistem.py
```

### Shared NexToU encoder-decoder

`_build_nextou_gnn_network` constructs `MultiTalentNexToU`. The checked six-stage 3D plan uses channels `[32, 64, 128, 256, 320, 320]`. Pool GNN and Swin GNN blocks are placed inside the shared encoder and decoder according to the NexToU-derived stage logic.

Relevant files:

```text
multitalent/network_architecture/nextou/encoder_decoder_blocks.py
multitalent/network_architecture/nextou/multitalent_nextou.py
multitalent/network_architecture/nextou/torch_edge.py
multitalent/network_architecture/nextou/torch_nn.py
multitalent/network_architecture/nextou/pos_embed.py
```

### CLIP-conditioned dynamic feature head

`ClipDrivenFeatureMultiTalentWrapper.forward` obtains all encoder skips, retains the `320`-channel bottleneck, and calls `decoder.forward_features`. The latter returns the final `32`-channel D4 feature and four lower-resolution static outputs per sample.

`ClipDrivenFeatureDynamicHead` then computes:

```text
bottleneck 320
  -> GroupNorm(16) -> ReLU -> global average pooling -> Conv3d(320, 256)

raw CLIP class embedding 512
  -> Linear(512, 256) -> ReLU

concat(image 256, text 256)
  -> shared Conv3d controller(512, 153)

final decoder feature 32
  -> GroupNorm(16) -> ReLU -> Conv3d(32, 8)
  -> dynamic Conv3d(8, 8) -> ReLU
  -> dynamic Conv3d(8, 8) -> ReLU
  -> dynamic Conv3d(8, 1)
```

One 153-parameter vector is generated per class and sample. Grouped convolution evaluates all class heads for that sample. The controller, image projection, text projection, and pre-classification projection are shared across datasets. CLIP vectors are persistent frozen buffers; the `512 -> 256` projection remains trainable.

Relevant file:

```text
multitalent/utilities/MultiTalent/clip_driven_feature_wrapper.py
```

### Output semantics

Each sample retains its dataset-specific multiclass vocabulary, including background. The primary output is a dense softmax-compatible logit tensor:

```text
Dataset801: [N, 2, D, H, W]
Dataset802: [N, 5, D, H, W]
```

With deep supervision enabled, each sample returns five tensors ordered from full to lower resolution:

```text
[dynamic D4, static D3, static D2, static D1, static D0]
```

Static full-resolution heads are removed after the wrapper is created. There is no logit-level FiLM adapter, residual logit fusion, prompt gate, or dynamic gate in this trainer.

## Prompt handling

Class text defaults can be generated from `dataset.json`, but the checked Task 850 run uses the explicit strings in `configs/task850_prompt_texts.json`. `update_prompt_texts_from_dataset_jsons` verifies that the number of prompts equals the number of output classes for every dataset.

The raw output of `clip.encode_text` is used without L2 normalization. Embeddings can be loaded from a cache with this schema:

```python
{
    "embeddings_by_id": {
        "801": {
            "clip_model_name": "ViT-B/32",
            "prompt_texts": ["..."],
            "normalized": False,
            "embeddings": torch.Tensor,
        }
    }
}
```

A cache entry is rejected when its model name, text list, normalization flag, or tensor shape does not match.

## Loss

`_NexToUTopologyLossMixin._build_loss` constructs Dice + cross-entropy + TI loss for non-region label managers. For 3D training, connectivity is 26 and the default TI weight is `1e-6`; `MT_NEXTOU_TOPOLOGY_WEIGHT` can override it. Foreground labels are combined pairwise into exclusion constraints.

For two output classes, there is only one foreground label, so no pair exists and the topology term is exactly zero. For five output classes, labels 1 through 4 produce six pairwise exclusions.

Relevant files:

```text
multitalent/training/nnUNetTrainer/project_specific/MultiTalent_native/
  MultiTalent_nextou_ablation.py
multitalent/training/loss/nextou_topology_losses.py
```

## Joint sampling

The trainer builds one combined case list while preserving each case's dataset ID. Dataset weights are derived from inverse square-root case-count weights at the case level. Summed over each source dataset, its expected selection probability is therefore proportional to the square root of its case count.

Augmentation and label handling remain dataset-specific. Batch size 1 was used by the checked Task 850 plan, so one sample and one target vocabulary are active per training iteration.

## Deliberate adaptation

The CLIP-conditioned dynamic feature concept was adapted to MultiTalent's dataset-specific multiclass setting. Unlike the referenced CLIP-Driven Universal Model's foreground-oriented output convention, TPUS generates one dynamic logit for every class including background, then uses the existing MultiTalent softmax loss and prediction path. This is a material adaptation and must not be described as an exact reproduction of the reference implementation.

## Included ablations

The release retains composable prompt, topology-loss, GNN, dynamic-convolution, and multistem trainer modules because they are reachable through string-based trainer discovery and support method ablations. The main supported public command uses only the trainer named at the top of this document. Other trainers should be treated as research variants unless separately verified.
