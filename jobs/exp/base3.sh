############TRAINING############
name=$1
version=$2
seed=$3
data_dir=$4
gpu_id=$5
num_workers=$6


cd "../.."
if [[ $version == "pattr_bic_pre_subloss_0256_4096" ]]; then
	pattr_epoch="13epoch"
	model_name="${v1}_${v3}"
	if [[ $name == "dcase2024" ]]; then
		path_command='datamodule.pretrained_single_idx_dict_path_dict.onehot_pattr='"pseudoattr/label/${name}/gmm_bic_subloss_0256_4096/13epoch_${seed}/idx.json"''
	elif [[ $name == "dcase2023" ]]; then
		path_command='datamodule.pretrained_single_idx_dict_path_dict.onehot_pattr='"pseudoattr/label/${name}/gmm_bic_pattr_macdom_subloss_0256_4096/13epoch_${seed}/idx.json"''
	else
		echo "Invalid DCASE"
		exit 1
	fi
elif [[ $version == "pattr_bic_triplet_subloss_0256_4096" ]]; then
	path_command='datamodule.pretrained_single_idx_dict_path_dict.onehot_pattr='"pseudoattr/label/${name}/gmm_bic_triplet/5epoch_${seed}/idx.json"''
elif [[ $version == "pattr_bic_openl3_subloss_0256_4096" ]]; then
	path_command='datamodule.pretrained_single_idx_dict_path_dict.onehot_pattr='"pseudoattr/label/${name}/gmm_bic_openl3/env512mel128/idx.json"''
elif [[ $version == "pattr_bic_panns_subloss_0256_4096" ]]; then
	path_command='datamodule.pretrained_single_idx_dict_path_dict.onehot_pattr='"pseudoattr/label/${name}/gmm_bic_panns/CNN14/idx.json"''
elif [[ $version == "pattr_gt_subloss_0256_4096" ]]; then
	path_command='datamodule.pretrained_single_idx_dict_path_dict.onehot_pattr='"pseudoattr/label/${name}/gt/idx.json"''
fi

exp_yaml="${name}/pattr_single_subloss_0256_4096"
version="${version}_${seed}"
############TESTING############
config_name="cosine"
umap_metric="cosine"
umap_stage=false
epoch_list=(12 14 16)
################################


echo "num_workers was set to ${num_workers}"
source "venv/bin/activate"
job_dir_name=$(basename "$(dirname "$(dirname "$(pwd)")")")
exp_root="results/exp"
echo ${exp_root}



python train.py experiments="${exp_yaml}" \
'path.exp_root='${exp_root}'' 'name='${name}'' \
'num_workers='${num_workers}'' 'version='${version}'' 'pos_machine=all' \
'datamodule.data_dir='${data_dir}'' \
'refresh_rate=1' 'seed='${seed}'' 'trainer.devices='"[${gpu_id}]"'' \
$path_command


if [[ $name == "dcase2024" ]]; then
	machines=("bearing" "fan" "gearbox" "slider" "ToyCar" "ToyTrain" "valve" "3DPrinter" "AirCompressor" "BrushlessMotor" "HairDryer" "HoveringDrone" "RoboticArm" "Scanner" "ToothBrush" "ToyCircuit")
elif [[ $name == "dcase2023" ]]; then
	machines=("bearing" "fan" "gearbox" "slider" "ToyCar" "ToyTrain" "valve" "Vacuum" "ToyTank" "ToyNscale" "ToyDrone" "bandsaw" "grinder" "shaker")
else
	echo "Invalid DCASE"
	exit 1
fi



for epoch in "${epoch_list[@]}"
do
	ckpt_name="interval_epoch=$((epoch - 1))-*"
	feat_ver="epoch${epoch}"
	checkpoint="checkpoints/${ckpt_name}.ckpt"
	for pos_machine in "${machines[@]}"
	do
		echo "#            pos_machine is ${pos_machine}           #"
		python test/test.py backend=${config_name} \
		'pos_machine='${pos_machine}'' 'exp_root='${exp_root}'' \
		'checkpoint="'${checkpoint}'"' 'name='${name}'' 'version='${version}'' \
		'feat_ver='${feat_ver}'' 'dataloader_cfg.num_workers='${num_workers}'' \
		'all_mode='true'' 'device='"cuda:${gpu_id}"''

		score_path="${exp_root}/${name}/${version}/${pos_machine}/infer/version_${feat_ver}/${ckpt_name}_test_score.csv"
		python evaluate23.py --score_path "${score_path}" --all_mode

		if ${umap_stage}; then
		python umap_vis.py --valid_df_path "${exp_root}/${name}/${version}/${pos_machine}/infer/version_${feat_ver}/${ckpt_name}_train.csv" \
		--eval_df_path "${exp_root}/${name}/${version}/${pos_machine}/infer/version_${feat_ver}/${ckpt_name}_test.csv" --all_mode --metric ${umap_metric}
		fi
	done
done

cd ensemble
./base.sh ${name} ${version}
