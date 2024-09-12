################################
dcase=$1
method=$2
################################
result_dir="../results/exp/${dcase}"
cfg_path="epoch_12_14_16.yaml"
infer_ver="epoch_12_14_16"
################################
if [ "${dcase}" = "dcase2023" ]; then
    machines=("bearing" "fan" "gearbox" "slider" "ToyCar" "ToyTrain" "valve" "Vacuum" "ToyTank" "ToyNscale" "ToyDrone" "bandsaw" "grinder" "shaker")
elif [ "${dcase}" = "dcase2024" ]; then
    machines=("bearing" "fan" "gearbox" "slider" "ToyCar" "ToyTrain" "valve" "3DPrinter" "AirCompressor" "BrushlessMotor" "HairDryer" "HoveringDrone" "RoboticArm" "Scanner" "ToothBrush" "ToyCircuit")
fi
################################
source "../venv/bin/activate"

method_dir="${result_dir}/${method}"
echo ${method_dir}
python ensemble.py ${cfg_path} --base_dir ${method_dir} --machines "${machines[@]}"

echo "Evaluating ensemble results"
for pos_machine in "${machines[@]}"
do
    score_path="${method_dir}/${pos_machine}/infer/version_${infer_ver}/ensemble_test_score.csv"
    python ../evaluate23.py --score_path "${score_path}" --all_mode
done

echo "Finished"
