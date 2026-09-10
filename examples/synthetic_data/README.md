# Synthetic data example

Run:

```bash
python scripts/generate_synthetic_datasets.py --output-root /tmp/tpus_synthetic
```

This creates nnU-Net-style `Dataset901_SyntheticUS` and `Dataset902_SyntheticMRI` folders with mathematical ellipsoids, integer labels, and matching image geometry. The files contain no patient-derived content and are intended only for I/O, schema, and smoke tests.

The generator refuses to overwrite an existing dataset directory unless `--force` is passed. Never point it at a real `nnUNet_raw` directory with IDs 901 or 902 already in use.
