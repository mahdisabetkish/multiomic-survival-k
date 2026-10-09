# Multi-omic subtyping with survival-guided choice of the number of clusters

Code and results for *A Pipeline for Multi-Omic Subtyping of Breast Cancer: Choosing the Number of Clusters by Survival Improves Held-Out Discrimination over Silhouette* (M. Sabet Kish, C. Hennig, M. Yazdi).

The pipeline summarises METABRIC mRNA, copy-number and mutation data as pathway, chromosome-arm and gene features, learns a joint representation with MOFA, and chooses the number of clusters by survival separation on inner validation folds. It is evaluated on held-out patients, over ten repeated train/test splits, on simulated data, and on TCGA-BRCA.

## Setup (Python 3.12)

```
python -m venv envs/venv
source envs/venv/bin/activate
pip install -r envs/requirements.txt
python scripts/01_download_data.py
```

The last command downloads METABRIC and TCGA-BRCA from cBioPortal and the gene sets from Enrichr (about 0.7 GB, 1.5 GB once unpacked) and checks them against the files used for the paper.

## Run

Every stage is a script in `scripts/` that reads `config/config.yaml`, for example `python scripts/02_preprocess_omics.py --config config/config.yaml`. Run the stages in the order listed in `slurm/submit_all.sh`, or submit them all to a SLURM cluster with `bash slurm/submit_all.sh`. The full run takes several hours on four CPU cores. `scripts/make_*.py` rebuild the paper's tables and diagrams from `outputs/` (the workflow diagram needs the Graphviz `dot` program).

## Results

`outputs/` contains the tables, figures, logs and fitted models from our run, so every number in the paper can be checked without rerunning the pipeline.
