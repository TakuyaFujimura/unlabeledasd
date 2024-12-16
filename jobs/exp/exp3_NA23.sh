data_dir=""
gpu_id=0
num_workers=6

if [ "${data_dir}" = "" ]
then
    echo "Please specify data_dir in the script"
    exit 1
fi


dcase="dcase2023"
for seed in {1..5}
do
    # N/A for dcase2023
    version="pattr_macdom_subloss_0256_4096"
    ./base1.sh "${dcase}" "${version}" "${seed}" "${data_dir}/${dcase}/all/raw" "${gpu_id}" "${num_workers}"
done
