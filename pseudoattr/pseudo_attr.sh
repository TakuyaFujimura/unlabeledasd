data_dir=""
gpu_id=0

for dcase in "dcase2023" "dcase2024"
do
    for method in "panns" "openl3"
    do
        ./base.sh ${data_dir} ${dcase} ${method} 0 ${gpu_id}
    done

    for method in "triplet" "class"
    do
        for seed in {1..5}
        do
            ./base.sh ${data_dir} ${dcase} ${method} ${seed} ${gpu_id}
        done
    done
done

# GT for dcase2024
source "../venv/bin/activate"
pseudo_attr_machines=("ToyTrain" "gearbox" "slider" "AirCompressor" "BrushlessMotor" "HoveringDrone" "ToothBrush")
python pseudo_attr_gt24.py --data_dir ${data_dir}/dcase2024/all/raw --save_dir "dcase2024/gt" --pseudo_attr_machines "${pseudo_attr_machines[@]}"

