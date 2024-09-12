source ../venv/bin/activate
data_dir=""

if [ "${data_dir}" = "" ]; then
    echo "Please specify the data directory"
    exit 1
fi

python add_gt_info.py --data_dir ${data_dir}
