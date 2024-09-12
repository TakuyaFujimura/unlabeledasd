# Preparation
## PANNs and OpenL3
- Make venv (`unlabeledasd/pseudoattr/venv_ext`)
- `cd pseudoattr`
- `python3 -m venv venv_ext`
- `source venv_ext/bin/activate`
- `pip install -r requirements_ext.txt`

## PANNs
- Download `https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1`
- Store it in `pseudoattr` (i.e., `pseudoattr/Cnn14_mAP=0.431.pth`)

## Triplet
- `cd jobs/exp` and execute `./exp3_triplet.sh`

## Class
- For DCASE2023, `pattr_macdom_subloss_0256_4096_${seed}` should be executed
- For DCASE2024, `subloss_0256_4096_${seed}` should be executed
- These experiments are included in `jobs/exp/exp1.sh`
    - `cd jobs/exp` and execute `./exp1.sh`
