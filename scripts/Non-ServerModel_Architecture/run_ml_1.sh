#!/bin/bash

######################### Weights Share Receiver ############################################################################################
build_path=${BASE_DIR}/build_debwithrelinfo_gcc

 for(( m = 1; m <= 1; m++ )); do 

$build_path/bin/training_all_labels --my-id 1 --party 0,::1,7000 --party 1,::1,7001 --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits 13 --config-file-input Sample_shares --config-file-model file_config_model1 --actual-labels Actual_all_labels --current-path ${BASE_DIR}/build_debwithrelinfo_gcc --sample-size 20 --theta0 Theta_all_labels
pid1=$!

wait $pid1

cp $build_path/server1/outputshare_1 $build_path/server1/Image_shares/Theta_all_labels
wait 

done 

echo "100 iterations done for 20 samples"

