# Third-party notices

This repository uses third-party open-source software. Unless otherwise noted, AMI-authored code in this repository is licensed under the license specified in `LICENSE`.

Third-party packages are not vendored or redistributed in this repository. R packages are installed by users through `renv::restore()` based on `renv.lock`; Python packages are installed based on `pyproject.toml` and `poetry.lock`. Third-party packages retain their respective licenses.

## Major direct R dependencies

| Package | License | Purpose |
|---|---|---|
| tidyverse | MIT | Meta-package for data science workflows |
| dplyr | MIT | Data manipulation and transformation |
| pROC | GPL (>= 3) | ROC/AUC analysis and DeLong's test |
| ggplot2 | MIT | Data visualization and plotting |
| tableone | GPL-2 | Baseline characteristics table generation |
| readxl | MIT | Reading Excel files (.xls/.xlsx) |
| writexl | BSD 2-Clause | Writing Excel files (.xlsx) |
| mice | GPL (>= 2) | Multiple imputation by chained equations |
| boot | Unlimited | Bootstrap confidence interval estimation |

For the full R dependency list and exact versions, see `renv.lock`.

## Major direct Python dependencies

| Package | License | Purpose |
|---|---|---|
| logging| Python Software Foundation | Logging |
| collections| Python Software Foundation | Utilities |
| contextlib| Python Software Foundation | Utilities |
| copy| Python Software Foundation | Utilities |
| dataclasses| Python Software Foundation | Hyperparameter management |
| errno| Python Software Foundation | Utilities |
| functools| Python Software Foundation | Utilities |
| math| Python Software Foundation | Numerical computation |
| matplotlib| Python Software Foundation | Visualization |
| os| Python Software Foundation | Utilities |
| pathlib| Python Software Foundation | Utilities |
| pprint| Python Software Foundation | Logging |
| random| Python Software Foundation | Random number generation |
| re| Python Software Foundation | Utilities |
| typing| Python Software Foundation | Utilities |
| fairseq| MIT | Model Training |
| neurokit2| MIT | ECG waveform analysis |
| numpy| BSD | Numerical computation |
| omegaconf| BSD | Hyperparameter management |
| pandas| BSD | Data processing |
| scikit-learn| BSD | Numerical computation |
| scipy| BSD | Signal processing |
| torch| BSD | Model Training |

For the full Python dependency list and exact versions, see `pyproject.toml` and `poetry.lock`.

## Notes
This repository does not include any patient-level data, fine-tuned model weights, proprietary pre-processing code or production configuration files.