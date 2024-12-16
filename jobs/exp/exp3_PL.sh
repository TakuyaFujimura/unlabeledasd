data_dir=""
gpu_id=0
num_workers=6

if [ "${data_dir}" = "" ]
then
    echo "Please specify data_dir in the script"
    exit 1
fi


for dcase in "dcase2023" "dcase2024" 
do
    for seed in {1..5}
    do
        for version in "pattr_bic_pre_subloss_0256_4096" "pattr_bic_triplet_subloss_0256_4096" "pattr_bic_openl3_subloss_0256_4096" "pattr_bic_panns_subloss_0256_4096" 
        do
            ./base3.sh "${dcase}" "${version}" "${seed}" "${data_dir}/${dcase}/all/raw" "${gpu_id}" "${num_workers}"
        done
    done
done
