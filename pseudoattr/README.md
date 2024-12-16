# Preparation
After completing the following preparations, please run `pseudoattr/pseudo_attr.sh` to generate pseudo-labels.

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
- In DCASE2023, Class requires that `pattr_macdom_subloss_0256_4096_${seed}` has already been executed.
    - `pattr_macdom_subloss_0256_4096_${seed}` is included in `jobs/exp/exp3_NA23.sh`. Please execute it first.
- In DCASE2024, Class requires that `subloss_0256_4096_${seed}` has already been executed.
    - `subloss_0256_4096_${seed}` is included in `jobs/exp/exp1.sh`. Please execute it first.
