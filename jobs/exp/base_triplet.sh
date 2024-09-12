############TRAINING############
name=$1
version="triplet"
seed=$2
data_dir=$3
gpu_id=$4
num_workers=$5
exp_yaml="${name}/${version}"
version="${version}_${seed}"
################################
cd ../..
echo "num_workers was set to ${num_workers}"
source "venv/bin/activate"
exp_root="results/exp"
echo ${exp_root}


if [[ "${name}" == "dcase2024" ]]; then
    machines=("ToyTrain" "gearbox" "slider" "AirCompressor" "BrushlessMotor" "HoveringDrone" "ToothBrush")
elif [[ "${name}" == "dcase2023" ]]; then
    machines=("bearing" "fan" "gearbox" "slider" "ToyCar" "ToyTrain" "valve" "Vacuum" "ToyTank" "ToyNscale" "ToyDrone" "bandsaw" "grinder" "shaker")
else
    echo "Invalid DCASE"
    exit 1
fi

for pos_machine in "${machines[@]}"
do

echo "################################################"
echo "#            pos_machine is ${pos_machine}           #"
echo "################################################"

python train.py experiments="${exp_yaml}" \
'path.exp_root='${exp_root}'' 'name='${name}'' \
'num_workers='${num_workers}'' 'version='${version}'' 'pos_machine='${pos_machine}'' \
'datamodule.data_dir='${data_dir}'' \
'refresh_rate=1' 'seed='${seed}'' 'trainer.devices='"[${gpu_id}]"''
done
