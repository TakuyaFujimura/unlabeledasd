dcase="dcase2024"
for method in "subloss_0256_4096_1" "subloss_0256_4096_2" "subloss_0256_4096_3" "subloss_0256_4096_4" "subloss_0256_4096_5"
do
./base.sh ${dcase} ${method}
done
