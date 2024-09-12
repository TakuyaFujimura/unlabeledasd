data_dir=$1
dcase=$2
method=$3
seed=$4
gpu_id=$5


###############################################################

function extract () {
dcase=$1
method=$2
seed=$3

if [ "${dcase}" = "dcase2023" ]; then
    pseudo_attr_machines=("bearing" "fan" "gearbox" "slider" "ToyCar" "ToyTrain" "valve" "Vacuum" "ToyTank" "ToyNscale" "ToyDrone" "bandsaw" "grinder" "shaker")
elif [ "${dcase}" = "dcase2024" ]; then
    pseudo_attr_machines=(ToyTrain gearbox slider AirCompressor BrushlessMotor HoveringDrone ToothBrush)
fi

echo "Start Extraction"
if [ "${method}" = "panns" ]; then
    source "venv_ext/bin/activate"
    python -u extract_external.py --data_dir ${data_dir}/${dcase}/all/raw --model_name "panns" \
    --model_hp "CNN14" --machines "${pseudo_attr_machines[@]}"
    python -u umap_trans.py --model_path "${dcase}/panns" --ckpt_cond "CNN14"
elif [ "${method}" = "openl3" ]; then
    source "venv_ext/bin/activate"
    python -u extract_external.py --data_dir ${data_dir}/${dcase}/all/raw --model_name "openl3" \
    --model_hp "env512mel128" --machines "${pseudo_attr_machines[@]}"
    python -u umap_trans.py --model_path "${dcase}/openl3" --ckpt_cond "env512mel128"
elif [ "${method}" = "triplet" ]; then
    source "../venv/bin/activate"
    python -u extract_scratch.py --data_dir ${data_dir}/${dcase}/all/raw --model_path "../results/exp/${dcase}/triplet_${seed}" \
    --ckpt_cond "interval_epoch=5-*.ckpt" --machines "${pseudo_attr_machines[@]}" --device "cuda:${gpu_id}"
    python -u umap_trans.py --model_path "../results/exp/${dcase}/triplet_${seed}" --ckpt_cond "interval_epoch=5-*"
elif [ "${method}" = "class" ]; then
    source "../venv/bin/activate"
    if [ "${dcase}" = "dcase2023" ]; then
        python -u extract_scratch.py --data_dir ${data_dir}/${dcase}/all/raw --model_path "../results/exp/${dcase}/pattr_macdom_subloss_0256_4096_${seed}" \
        --ckpt_cond "interval_epoch=13-*.ckpt" --machines "${pseudo_attr_machines[@]}" --device "cuda:${gpu_id}" --all_ckpt
        python -u umap_trans.py --model_path "../results/exp/${dcase}/pattr_macdom_subloss_0256_4096_${seed}" --ckpt_cond "interval_epoch=13-*"
    elif [ "${dcase}" = "dcase2024" ]; then
        python -u extract_scratch.py --data_dir ${data_dir}/${dcase}/all/raw --model_path "../results/exp/${dcase}/subloss_0256_4096_${seed}" \
        --ckpt_cond "interval_epoch=13-*.ckpt" --machines "${pseudo_attr_machines[@]}" --device "cuda:${gpu_id}" --all_ckpt
        python -u umap_trans.py --model_path "../results/exp/${dcase}/subloss_0256_4096_${seed}" --ckpt_cond "interval_epoch=13-*"
    fi
fi
}


###############################################################

function labeling () {
dcase=$1
method=$2
seed=$3

source "../venv/bin/activate"
if [ "${dcase}" = "dcase2023" ]; then
    pseudo_attr_machines=("bearing" "fan" "gearbox" "slider" "ToyCar" "ToyTrain" "valve" "Vacuum" "ToyTank" "ToyNscale" "ToyDrone" "bandsaw" "grinder" "shaker")
elif [ "${dcase}" = "dcase2024" ]; then
    pseudo_attr_machines=("ToyTrain" "gearbox" "slider" "AirCompressor" "BrushlessMotor" "HoveringDrone" "ToothBrush")
fi

if [ "${method}" = "panns" ]; then
    python pseudo_attr_gmm.py --data_dir ${data_dir}/${dcase}/all/raw --save_dir "${dcase}/gmm_bic_panns/CNN14" \
    --ckpt_path "embed/${dcase}/panns/CNN14" --pseudo_attr_machines "${pseudo_attr_machines[@]}" \
    --compression --ic "bic"
elif [ "${method}" = "openl3" ]; then
    python pseudo_attr_gmm.py --data_dir ${data_dir}/${dcase}/all/raw --save_dir "${dcase}/gmm_bic_openl3/env512mel128" \
    --ckpt_path "embed/${dcase}/openl3/env512mel128" --pseudo_attr_machines "${pseudo_attr_machines[@]}" \
    --compression --ic "bic"
elif [ "${method}" = "triplet" ]; then
    python pseudo_attr_gmm.py --data_dir ${data_dir}/${dcase}/all/raw --save_dir "${dcase}/gmm_bic_triplet/5epoch_${seed}" \
    --ckpt_path "embed/${dcase}/triplet_${seed}/interval_epoch=5-*" --pseudo_attr_machines "${pseudo_attr_machines[@]}" \
    --compression --ic "bic"
elif [ "${method}" = "class" ]; then
    if [ "${dcase}" = "dcase2023" ]; then
        python pseudo_attr_gmm.py --data_dir ${data_dir}/${dcase}/all/raw --save_dir "${dcase}/gmm_bic_pattr_macdom_subloss_0256_4096/13epoch_${seed}" \
        --ckpt_path "embed/${dcase}/pattr_macdom_subloss_0256_4096_${seed}/interval_epoch=13-*" --pseudo_attr_machines "${pseudo_attr_machines[@]}" \
        --compression --ic "bic"
    elif [ "${dcase}" = "dcase2024" ]; then
        python pseudo_attr_gmm.py --data_dir ${data_dir}/${dcase}/all/raw --save_dir "${dcase}/gmm_bic_subloss_0256_4096/13epoch_${seed}" \
        --ckpt_path "embed/${dcase}/subloss_0256_4096_${seed}/interval_epoch=13-*" --pseudo_attr_machines "${pseudo_attr_machines[@]}" \
        --compression --ic "bic"
    fi
fi
}
###############################################################

if [ "${data_dir}" = "" ]; then
    echo "Please specify the data_dir"
    exit
fi

extract ${dcase} ${method} ${seed}
labeling ${dcase} ${method} ${seed}
