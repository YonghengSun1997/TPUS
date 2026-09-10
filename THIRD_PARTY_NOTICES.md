# Third-party notices

TPUS combines original modifications with code derived from or dependent on several upstream research projects. This inventory is about source provenance and license boundaries; it is not legal advice and does not replace the license texts in [`LICENSES/`](LICENSES/).

The root [`LICENSE`](LICENSE) applies Apache License 2.0 to TPUS-original contributions. It does not relicense third-party or adapted components identified below.

## MultiTalent

- Project: <https://github.com/MIC-DKFZ/MultiTalent/tree/public_MT>
- Audited revision: `be8243fd16f7d262be2497fe4264dfc576008528`
- License: Apache License 2.0
- License copy: `LICENSES/MultiTalent-Apache-2.0.txt`
- Scope: most files under `multitalent/`, including planning, preprocessing, training, inference, postprocessing, evaluation, model sharing, data loading, and command entry points.

TPUS changes include new trainers and wrappers, NexToU integration, topology loss integration, dataset-specific stem handling, CLIP prompt handling, dynamic feature heads, release-path parameterization, and packaging/documentation changes. Files changed from MultiTalent retain their existing attribution where present and are identified in this notice and/or a source-file modification notice.

## nnU-Net

- Project: <https://github.com/MIC-DKFZ/nnUNet>
- Current revision checked during release audit: `0b47823c2d5c80e9db62e838c56f79ba0622fa2e`
- License: Apache License 2.0
- License copy: `LICENSES/nnUNet-Apache-2.0.txt`
- Scope: nnU-Net-derived framework behavior inherited through MultiTalent.

The selected TPUS source did not retain a separate exact nnU-Net source commit, so the current revision above is a license-check reference rather than a claim that every inherited line came from that commit.

## NexToU

- Project: <https://github.com/PengchengShi1220/NexToU>
- Official revision matched during release audit: `9671941a424d3e2fe1ff6f6c65c53a5c32b9dedd`
- License: Apache License 2.0
- License copy: `LICENSES/NexToU-Apache-2.0.txt`
- Scope: `multitalent/network_architecture/nextou/`, `multitalent/training/loss/nextou_topology_losses.py`, and the NexToU/TI trainer integration.

The local source archive did not contain Git metadata. Hash comparison established that its `pos_embed.py`, `torch_nn.py`, and `torch_edge.py` match the files at the official revision above. TPUS retains `pos_embed.py` and `torch_nn.py` unchanged from that NexToU snapshot; other network and loss files are adaptations for MultiTalent's multi-dataset interfaces.

## Vision GNN

- Project: <https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch>
- Revision checked during release audit: `f90e129b645c3b1684fe07cd361cd557d0ad71f7`
- License: Apache License 2.0
- License copy: `LICENSES/Vision-GNN-Apache-2.0.txt`
- Upstream notice: `LICENSES/Vision-GNN-THIRD-PARTY-NOTICE.txt`
- Scope: graph-convolution and dynamic-kNN foundations carried through the NexToU-derived `torch_nn.py` and `torch_edge.py` code.

## MAE positional embedding utility

- Project: <https://github.com/facebookresearch/mae>
- Revision checked during release audit: `efb2a8062c206524e35e47d04501ed4f544c0ae8`
- License: Creative Commons Attribution-NonCommercial 4.0 International
- License copy: `LICENSES/MAE-CC-BY-NC-4.0.txt`
- Scope: `multitalent/network_architecture/nextou/pos_embed.py`, whose retained header explicitly states that it was modified from MAE's positional-embedding utility.

This component carries a noncommercial restriction. NexToU's repository-level Apache declaration does not erase the retained MAE notice or its separate license terms.

## Topology Interaction

- Project: <https://github.com/TopoXLab/TopoInteraction>
- Revision checked during release audit: `1d8d6eb3e844c28ce0f1d3fbff69863a6b9fdcaf`
- License: MIT License
- License copy: `LICENSES/TopoInteraction-MIT.txt`
- Scope: conceptual and implementation lineage of TI loss, incorporated through the NexToU-derived loss implementation.

## OpenAI CLIP

- Project: <https://github.com/openai/CLIP>
- Pinned dependency revision: `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`
- License: MIT License
- License copy: `LICENSES/OpenAI-CLIP-MIT.txt`
- Scope: optional runtime dependency used to tokenize prompts and generate frozen ViT-B/32 text embeddings.

The CLIP source and checkpoint are not vendored in this repository. The prompt-cache generator downloads or uses them through the official package interface.

## CLIP-Driven Universal Model

- Project: <https://github.com/ljwztc/CLIP-Driven-Universal-Model>
- Revision checked during release audit: `43c12399cc98a40f447342a8dc2cecae5edae84c`
- License: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
- License copy: `LICENSES/CLIP-Driven-CC-BY-NC-ND-4.0.txt`
- Scope: method and code lineage for CLIP-conditioned controllers, dynamic-parameter parsing, and grouped dynamic-convolution heads in `multitalent/utilities/MultiTalent/clip_driven_feature_wrapper.py`, `multitalent/utilities/MultiTalent/prompt_dynamic_wrapper.py`, and `multitalent/utilities/MultiTalent/ablation_feature_wrappers.py`.

The TPUS maintainer confirmed on 2026-09-10 that written permission from the CLIP-Driven rights holder authorizes publication of the adapted material in this repository. The authorization text and any private correspondence are not distributed here. This statement does not amend the upstream public license, imply endorsement, or grant downstream rights beyond those provided by the applicable licenses and separate authorization. Readers should not assume that Apache License 2.0 relicenses these adapted files.

## Runtime Python packages

Additional runtime packages are declared in `pyproject.toml`. Their source code is not vendored here. Users remain responsible for complying with the licenses shipped by those packages and their transitive dependencies.

## TPUS-specific work

TPUS-specific glue, configuration, documentation, tests, and scripts are distinguishable from the upstream components above and are released under Apache License 2.0. The exceptions and additional terms for upstream and adapted components remain as described above.
