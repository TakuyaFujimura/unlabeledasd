data_dir=""
gpu_id=0
num_workers=6

if [ "${data_dir}" = "" ]
then
    echo "Please specify data_dir in the script"
    exit 1
fi

name=$(basename "$(pwd)")
for dcase in "dcase2023" "dcase2024"
do
    for seed in {1..5}
    do
        ./base_triplet.sh "${dcase}" "${seed}" "${data_dir}/${dcase}/all/raw" "${gpu_id}" "${num_workers}"
    done
done
