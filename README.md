# Improvements of Discriminative Feature Space Training for Anomalous Sound Detection in Unlabeled Conditions

## venv
- Make venv (`unlabledasd/venv`)
  - python version was `3.10.10`
  - `python3 -m venv venv`
  - `source venv/bin/activate`
  - `pip install -r requirements.txt`

## Data
- Download DCASE2023 and DCASE2024 data
- Modify the data structure as follows: (You can also use symbolic links)
  - `<data_dir>/dcase2023/all/raw/bandsaw/train`
  - ...
  - `<data_dir>/dcase2023/all/raw/valve/test`
  - `<data_dir>/dcase2024/all/raw/3DPrinter/train`
  - ...
  - `<data_dir>/dcase2024/all/raw/valve/test`
- Add ground-truth information
  - **This process renames the filenames** of the test set using ground-truth labels.
  - Set `data_dir` to `<data_dir>` in the `preprocess/add_gt_info.sh`.
  - `cd preprocess` and execute `./add_gt_info.sh`
  
## Training and Testing
- The full shell script of the experiments is provided in the `jobs/exp`
- `exp1.sh`, `exp2.sh`, and `exp3.sh` correspond to Tables 1, 2, and 3, respectively.
- Set `data_dir` in each shell script and execute it
- These shell scripts run experiments with five different seeds and several methods using a simple `for` loop. If you want to parallelize the process or run it selectively, you can split the `for` loop.
- When you execute the shell scripts, the training will strat, and the results will be stored in `results/exp/<dcase>/<method>/<machine>/infer/version_epoch<epoch>/*_test_result.csv`
  - `infer/version_epoch_12_14_16` is the result of an ensemble of anomaly scores for the 12, 14, and 16 epochs
  - `hmean_official` of `infer/versoin_epoch<epoch>/*_test_result.csv` is the official score of DCASE Task2 Challenge
  - `hmean_official` is the harmonic mean of the AUC of the source domain (`0_source_auc_all`), the AUC of the target domain (`0_target_auc_all`), and the pAUC of both domains (`0_all_pauc_all`)

- Before executing `exp3.sh`, pseudo labels should be generated (see next section).

## Pseudo-labeling
- `pseudoattr/pseudo_attr.sh` generates several types of pseudo-labels using a simple `for` loop. If you want to parallelize the process or run it selectively, you can split the `for` loop.
- Some pseudo-labeling methods require some preparation. See `pseudoattr/README.md` for more details.

<!-- ## Ensemble -->
<!-- - `ensemble/ensemble.sh` executes an ensemble of anomaly scores of 12, 14, and 16 epochs -->
<!-- - `cd ensemble`, set `${method}` in `ensemble.sh`, and execute `./ensemble.sh` -->


## Citation
```
@article{fujimura2024improvements,
  title={Improvements of Discriminative Feature Space Training for Anomalous Sound Detection in Unlabeled Conditions},
  author={Fujimura, Takuya and Kuroyanagi, Ibuki and Toda, Tomoki},
  journal={arXiv preprint arXiv:2409.09332},
  year={2024}
}
```
