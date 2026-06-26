
# Overview
This repository provides non-proprietary analysis code, model-architecture definitions, configuration files, and selected model-development scripts associated with the study, "Artificial Intelligence–Enhanced Auscultation to Identify Left Ventricular Dysfunction: Development and External Validation in Prospective Multicenter Studies."

# Repository scope
- This repository is intended to support methodological transparency.
- It includes code illustrating the model-development, inference, ECG-feature extraction, model ensemble, and statistical-analysis workflows. 
- Because the original patient-level physiological recordings and trained model weights cannot be released, the provided training and inference scripts use dummy data and randomly initialized models to demonstrate the workflow and expected file structure. 
- The repository is therefore not intended to directly reproduce the reported AUROC or other study results without access to the original data and trained model weights.

# Not included
- This repository does not include raw patient-level recordings, trained model weights, proprietary preprocessing code, production configuration files, or the complete device-specific inference pipeline.

# Installation
Execute the following commands to initialize and launch the Docker container:
```
>>> git clone …
>>> cd …
>>> docker build -f ./env/inference/Dockerfile -t inference .
>>> docker run -it --gpus all --shm-size=32g \
        -v $PWD:/app \
        -v $PWD/env/inference/pyproject.toml:/app/pyproject.toml \
        -v $PWD/env/inference/poetry.lock:/app/poetry.lock \
        inference bash

# >>> export PYTHONPATH=$PWD
# >>> poetry run python ...
```

# Code
## Dummy Model Training Recipe
Scripts for creating and running inference with dummy models are stored in the `script/` directory.

### step 01: `script/s01_train_models.py`
- This script creates randomly-initialized dummy CNN and wav2vec models.
- The network architecture for the CNN is defined in `config/cnn.yaml`, and the architecture for the wav2vec model is defined in `config/wav2vec`.
- This script performs k-fold cross-validation and saves multiple models:
```
{EXPERIMENT_DIRECTORY}/
`-- exp
    `-- 01_train_models
        |-- cnn  # CNN models
        |   |-- 0
        |   |   `-- model.pth  # fold-0 CNN model
        |   |-- …
        |   `-- 4
        |       `-- model.pth  # fold-0 CNN model
        `-- wav2vec  # wav2vec models
            |-- 0
            |   `-- model.pth  # fold-0 wav2vec model
            |-- …
            |-- 4
            |   `-- model.pth  # fold-0 wav2vec model
            `-- pretrained.ckpt  # Pretrained encoder networks
```

### step 02: `script/s02_inference.py`
- This script loads the models saved in step-01 to make predictions.
- The prediction results calculated by the CNN models are saved under `exp/cnn/`, and the results calculated by the wav2vec models are saved under `exp/w2v/`.
```
{EXPERIMENT_DIRECTORY}/
|-- data  # Reference labels and PCG/ECG data
|   |-- cv0_develop.csv  # fold-0 development set
|   |-- cv0_external.csv  # fold-0 external test set
|   `-- …
`-- exp
    |-- cnn  # Single model (CNN)
    |   |-- …
    |   `-- external
    |       |-- prediction.0.csv  # Predicted probabilities (fold-0)
    |       `-- …
    `-- w2v  # Single model (wav2vec)
        |   `-- …
        `-- external
            |-- prediction.0.csv  # Predicted probabilities (fold-0)
            `-- …
```

### step 03: `script/s03_ensemble.py`
- This script ensembles the inference results derived from multiple models and multiple auscultation sites.
- The final ensembled results are saved in the `exp/ensemble/`:
```
{EXPERIMENT_DIRECTORY}
`-- exp
    `-- ensemble
        |-- prediction.test.0.csv  # Prediction results (external test set)
        |-- prediction.valid.0.csv  # Prediction results (development set, fold-0)
        |-- …
        `-- prediction.valid.4.csv  # Prediction results (development set, fold-4)
```

### misc: `script/s99_extract_ecg_features.py`
- Code used for electrocardiogram (ECG) waveform analysis.
- This script utilizes neurokit2 to detect P-wave and R-wave onsets.

### misc: `script/s99_roc_curve.py`
- Code used to create ROC curves.


## Analysis Code
- Analysis code is stored in `analysis/script/`.

# License
- Unless otherwise noted, AMI-authored code in this repository is licensed under the MIT License.
- The R analysis environment is specified in renv.lock. Third-party R packages are not included in this repository and retain their respective licenses. Users installing the R environment via renv::restore() are responsible for complying with the licenses of those third-party packages. See THIRD_PARTY_NOTICES.md for a summary of major third-party dependencies.

# Citation
If you use this repository, please cite:
```
Sato et al. Artificial Intelligence–Enhanced Auscultation to Identify Left Ventricular Dysfunction: Development and External Validation in Prospective Multicenter Studies. [Journal/status to be updated].
```


