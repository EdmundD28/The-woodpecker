# Adopted RMS-normalized log-spectrum model

- Version: `REAL-RMSLOG-fb6b2bf9d8ef`.
- Python model: `spectrum_model.json`; generated C++ weights: `spectrum_weights.h`.
- Input: three ordered hits (endpoint, middle, endpoint), 8 kHz, 2400 ADC samples per hit.
- Preparation: whole-window mean removal and /2048; per-hit RMS normalization; symmetric Hann; magnitude RFFT /2400; fitted-scale log1p compression. Endpoint parameters are shared.
- Architecture: stride 2/2/2 convolutions with 8/16/32 channels, kernels 9/5/3; flatten 151x32 features; two-class softmax; 12,114 parameters.
- Training: fixed seed 42, both table and carpet, 174/56/65 unique hits in training/validation/development holdout. Five previously identified abnormal hits excluded. Preparation fitted only on unique training hits. No additional balanced-validation change was combined with this preparation.
- Development checks: 630 holdout combinations and 16 non-reusing triplets all classified correctly. These recordings were previously inspected and are not independent deployment acceptance data.
- Six source-split method study (not the mixed-support deployment weights): raw spectrum CNN 82.35%/92.70% mean balanced accuracy; RMS+log CNN 93.10%/99.81%, for carpet-to-table/table-to-carpet. Targets were reused across splits.
- The repository's existing public raw oracle inputs are retained; spectrum features and scores were regenerated. Full acquisition data, experimental runs, caches and binaries are not included.
- Default firmware environment: `esp32dev`; `esp32dev_spectrum` is an alias; legacy time model: `esp32dev_time`.
- Validation: Python unit tests, firmware host simulation, exact-length FFT comparison and ESP32 builds. No physical hardware flashing or new field acceptance in this change.
