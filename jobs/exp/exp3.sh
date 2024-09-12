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
        ######################################################
        for version in "pattr_bic_pre_subloss_0256_4096" "pattr_bic_triplet_subloss_0256_4096" "pattr_bic_openl3_subloss_0256_4096" "pattr_bic_panns_subloss_0256_4096" 
        do
            ./base3.sh "${name}" "${version}" "${seed}" "${data_dir}/${dcase}/all/raw" "${gpu_id}" "${num_workers}"
        done
        ######################################################
        if [ ${dcase} == "dcase2023" ]
        then
            # N/A for dcase2023
            version="pattr_macdom_subloss_0256_4096"
            ./base1.sh "${name}" "${version}" "${seed}" "${data_dir}/${dcase}/all/raw" "${gpu_id}" "${num_workers}"
        elif [ ${dcase} == "dcase2024" ]
        then
            # GT for dcase2024
            version="pattr_gt_subloss_0256_4096"
            ./base3.sh "${dcase}" "${version}" "${seed}" "${data_dir}/${dcase}/all/raw" "${gpu_id}" "${num_workers}"
        else
            echo "Invalid dcase"
            exit 1
        fi
        ######################################################
    done
done
