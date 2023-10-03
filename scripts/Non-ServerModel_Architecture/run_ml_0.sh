#!/bin/bash

######################### Weights Share Receiver ############################################################################################
build_path=${BASE_DIR}/build_debwithrelinfo_gcc

 for(( m = 1; m <= 1; m++ )); do 

$build_path/bin/training_all_labels --my-id 0 --party 0,::1,7000 --party 1,::1,7001 --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits 13 --config-file-input Sample_shares --config-file-model file_config_model0 --actual-labels Actual_all_labels --current-path ${BASE_DIR}/build_debwithrelinfo_gcc --sample-size 20 --theta0 Theta_all_labels
pid2=$!

wait $pid2  

cp $build_path/server0/outputshare_0 $build_path/server0/Image_shares/Theta_all_labels

wait

done 

echo "10 iterations done for 20 samples"

