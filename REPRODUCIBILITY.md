# Reproducibility notes

## Data

The source data are available from OpenNeuro as [`ds006104`, version 1.0.0](https://openneuro.org/datasets/ds006104/versions/1.0.0). The original team repository used the EEGLAB-formatted recordings under `derivatives/eeglab/sub-XXX/`.

Raw and processed EEG files are excluded from this repository by `.gitignore`. This keeps participant data and large intermediate files out of the portfolio mirror.

## MATLAB / EEGLAB preprocessing

Requirements:

- MATLAB
- EEGLAB
- ICLabel plugin
- `fillgaps` used by the TMS interpolation step

Place raw `.set/.fdt` data under `data/raw_data/` and the experiment event tables under `data/events_information/`. Run `preprocessing_matlab/preprocessing_main.m` from the repository after adding EEGLAB to the MATLAB path.

The pipeline writes stage-numbered outputs. Important checkpoints are:

- stage 1: structured events;
- stage 2: TMS-interpolated continuous data;
- stage 3: 256 Hz filtered continuous data;
- stage 4: `stim_1`-locked epochs;
- stage 5: automatic epoch QC;
- stage 6: ICA decomposition;
- stage 7: ICLabel-cleaned data;
- stage 8: task/condition/category subsets;
- stage 9: MATLAB and compressed NumPy export.

## Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Some modeling scripts additionally require PyTorch and, for older archived experiments, `pyriemann`. They are not pinned in the original top-level requirements file.

## Analysis / Figure 3

The Figure 3 workflow uses the continuous stage-3 `.set/.fdt` pair and aligns trials to `stim_1`.

Example ERP command from the original analysis script:

```bash
python analysis/average_stage3_erp.py \
  --set data/processed/S02_3_filtered_data.set \
  --task single_phoneme \
  --tms 0 \
  --target control \
  --category vowels \
  --event-role stim_1 \
  --channel FCz \
  --tmin -0.6 \
  --tmax 1.0 \
  --baseline-start -0.5 \
  --baseline-end -0.25 \
  --reject-ptp 300 \
  --output figures/S02_control_vowels_FCz_ERP.png
```

## Modeling

The main model scripts expect the original `npz2/` structure at the repository root. The key commands and evaluation distinctions are documented in `model/model/README.md` and `model/SUMMARY.md`.

Always distinguish:

- trial-level splits, where participants occur in both train and test sets but trials do not overlap;
- token and time-block controls;
- leave-one-subject-out evaluation, where the test participant is unseen.

## Known portability limits

- The archive contains exploratory scripts retained for provenance; not every attic script is maintained as a standalone entry point.
- The complete numerical results require the processed EEG arrays, which are not included.
- Hardware/software versions from the original runs were not fully pinned.
- The final poster and repository are the evidence sources; claims in this portfolio should not be extended beyond them.

