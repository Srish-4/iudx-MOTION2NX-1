#! /bin/bash
check_exit_statuses() {
   for status in "$@";
   do
      if [ $status -ne 0 ]; then
         echo "Exiting due to error."
         exit 1  # Exit the script with a non-zero exit code
      fi
   done
}
# paths required to run cpp files
image_config=${BASE_DIR}/config_files/file_config_input_remote
build_path=${BASE_DIR}/build_debwithrelinfo_gcc
image_provider_path=${BASE_DIR}/data/ImageProvider/Final_Output_Shares
debug_0=${BASE_DIR}/logs/server0/
scripts_path=${BASE_DIR}/scripts
smpc_config_path=${BASE_DIR}/config_files/smpc-helpernode-config.json
smpc_config=`cat $smpc_config_path`
# #####################Inputs##########################################################################################################
# Do dns reolution or not 
cs0_dns_resolve=`echo $smpc_config | jq -r .cs0_dns_resolve`
cs1_dns_resolve=`echo $smpc_config | jq -r .cs1_dns_resolve`
helpernode_dns_resolve=`echo $smpc_config | jq -r .helpernode_dns_resolve`


# cs0_host is the ip/domain of server0, cs1_host is the ip/domain of server1
cs0_host=`echo $smpc_config | jq -r .cs0_host`
cs1_host=`echo $smpc_config | jq -r .cs1_host`
helpernode_host=`echo $smpc_config | jq -r .helpernode_host`

if [[ $cs0_dns_resolve == "true" ]];
then 
cs0_host=`dig +short $cs0_host | grep '^[.0-9]*$' | head -n 1`
fi
if [[ $cs1_dns_resolve == "true" ]];
then 
cs1_host=`dig +short $cs1_host | grep '^[.0-9]*$' | head -n 1`
fi
if [[ $helpernode_dns_resolve == "true" ]];
then
helpernode_host=`dig +short $helpernode_host | grep '^[.0-9]*$' | head -n 1`
fi


# Ports on which weights provider  receiver listens/talks
cs0_port_model_receiver=`echo $smpc_config | jq -r .cs0_port_model_receiver`
cs1_port_model_receiver=`echo $smpc_config | jq -r .cs1_port_model_receiver`

# Ports on which image provider  receiver listens/talks
cs0_port_image_receiver=`echo $smpc_config | jq -r .cs0_port_image_receiver`
cs1_port_image_receiver=`echo $smpc_config | jq -r .cs1_port_image_receiver`

# Ports on which Image provider listens for final inference output
cs0_port_cs0_output_receiver=`echo $smpc_config | jq -r .cs0_port_cs0_output_receiver`
cs0_port_cs1_output_receiver=`echo $smpc_config | jq -r .cs0_port_cs1_output_receiver`

# Ports on which server0 and server1 of the inferencing tasks talk to each other
cs0_port_inference=`echo $smpc_config | jq -r .cs0_port_inference`
cs1_port_inference=`echo $smpc_config | jq -r .cs1_port_inference`
helpernode_port_inference=`echo $smpc_config | jq -r .helpernode_port_inference`
relu0_port_inference=`echo $smpc_config | jq -r .relu0_port_inference`
relu1_port_inference=`echo $smpc_config | jq -r .relu1_port_inference`

number_of_layers=`echo $smpc_config | jq -r .number_of_layers`
fractional_bits=`echo $smpc_config | jq -r .fractional_bits`

# Index of the image for which inferencing task is run
image_id=`echo $smpc_config | jq -r .image_id`


# echo all input variables
#echo "cs0_host $cs0_host"
#echo "cs1_host $cs1_host"
#echo "cs0_port_model_receiver $cs0_port_model_receiver"
#echo "cs1_port_model_receiver $cs1_port_model_receiver"
#echo "cs0_port_cs0_output_receiver $cs0_port_cs0_output_receiver"
#echo "cs0_port_cs1_output_receiver $cs0_port_cs1_output_receiver"
#echo "cs0_port_inference $cs0_port_inference"
#echo "cs1_port_inference $cs1_port_inference"
#echo "fractional bits: $fractional_bits"
#echo "no. of splits: $splits"
##########################################################################################################################################


if [ ! -d "$debug_0" ];
then
	# Recursively create the required directories
	mkdir -p $debug_0
fi


cd $build_path

if [ -f finaloutput_0 ]; then
   rm finaloutput_0
   # echo "final output 0 is removed"
fi

if [ -f MemoryDetails0 ]; then
   rm MemoryDetails0
   # echo "Memory Details0 are removed"
fi

if [ -f AverageMemoryDetails0 ]; then
   rm AverageMemoryDetails0
   # echo "Average Memory Details0 are removed"
fi

if [ -f AverageMemory0 ]; then
   rm AverageMemory0
   # echo "Average Memory Details0 are removed"
fi

if [ -f AverageTimeDetails0 ]; then
   rm AverageTimeDetails0
   # echo "AverageTimeDetails0 is removed"
fi

if [ -f AverageTime0 ]; then
   rm AverageTime0
   # echo "AverageTime0 is removed"
fi

#########################Weights Share Receiver ############################################################################################
echo "Weight shares receiver starts"
$build_path/bin/Weights_Share_Receiver_CNN --my-id 0 --port $cs0_port_model_receiver --current-path $build_path > $debug_0/Weights_Share_Receiver.txt &
pid1=$!

#########################Image Share Receiver ############################################################################################
echo "Image shares receiver starts"

$build_path/bin/Image_Share_Receiver_CNN --my-id 0 --port $cs0_port_image_receiver --fractional-bits $fractional_bits --file-names $image_config --current-path $build_path > $debug_0/Image_Share_Receiver.txt &
pid2=$!

wait $pid1
check_exit_statuses $? 
wait $pid2
check_exit_statuses $?
echo "Weight shares received"
echo "Image shares received"


#########################Share generators end ############################################################################################

echo "Inferencing task of the image shared starts"


############################Inputs for inferencing tasks #######################################################################################
layer_id=1
layer_types=($(cat layer_types0))
# input_config=" "
image_share="remote_image_shares"
if [ $layer_id -eq 1 ];
then
    input_config="remote_image_shares"
fi
start=$(date +%s)
#######################################Matrix multiplication layer 1 ###########################################################################
for((; layer_id<$number_of_layers; layer_id++))
do

if [ ${layer_types[layer_id]} -eq 1 ] && [ $layer_id -eq 1 ];then 
   input_config="remote_image_shares"
   
   $build_path/bin/server0_cnn --WB_file file_config_model0 --input_file $input_config  --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --helper_node $helpernode_host,$helpernode_port_inference --current-path $build_path --layer-id $layer_id --fractional-bits $fractional_bits > $debug_0/server0_layer${layer_id}.txt &
   pid1=$!
   wait $pid1
   check_exit_statuses $?
   echo "Layer $layer_id: Convolution is done"
   
   $build_path/bin/tensor_gt_relu --my-id 0 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input0 --current-path $build_path > $debug_0/tensor_gt_relu0_layer${layer_id}.txt &
   pid2=$!
   wait $pid2
   check_exit_statuses $?
   echo "Layer $layer_id: ReLU is done"

   tail -n +2 server0/outputshare_0 >> server0/cnn_outputshare_0
   

elif [ ${layer_types[layer_id]} -eq 1 ] && [ $layer_id -gt 1 ];then
   
   input_config="cnn_outputshare"

   $build_path/bin/server0_cnn --WB_file file_config_model0 --input_file $input_config  --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --helper_node $helpernode_host,$helpernode_port_inference --current-path $build_path --layer-id $layer_id --fractional-bits $fractional_bits > $debug_0/server0_layer${layer_id}.txt &
   pid1=$!
   wait $pid1
   check_exit_statuses $?
   echo "Layer $layer_id: Convolution is done"

   $build_path/bin/tensor_gt_relu --my-id 0 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input0 --current-path $build_path > $debug_0/tensor_gt_relu0_layer${layer_id}.txt &
   pid2=$!
   wait $pid2
   check_exit_statuses $?
   echo "Layer $layer_id: ReLU is done"
   tail -n +2 server0/outputshare_0 >> server0/cnn_outputshare_0

elif [ ${layer_types[layer_id]} -eq 0 ] && [[ $layer_id -lt $number_of_layers ]];then

      input_config="outputshare"
       
      $build_path/bin/server0 --WB_file file_config_model0 --input_file $input_config  --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --helper_node $helpernode_host,$helpernode_port_inference  --current-path $build_path --layer-id $layer_id --fractional-bits $fractional_bits > $debug_0/server0_layer${layer_id}.txt &
      pid1=$!
      wait $pid1
      check_exit_statuses $?
      echo "Layer $layer_id: Matrix multiplication and addition is done"
      
      $build_path/bin/tensor_gt_relu --my-id 0 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input0 --current-path $build_path > $debug_0/tensor_gt_relu0_layer${layer_id}.txt &
      pid2=$!
      wait $pid2
      check_exit_statuses $?
      echo "Layer $layer_id: ReLU is done"
   fi
done
      
      input_config="outputshare"

      $build_path/bin/server0 --WB_file file_config_model0 --input_file $input_config  --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --helper_node $helpernode_host,$helpernode_port_inference  --current-path $build_path --layer-id $layer_id --fractional-bits $fractional_bits > $debug_0/server0_layer${layer_id}.txt &
      pid1=$!
      wait $pid1
      check_exit_statuses $?
      echo "Layer $layer_id: Matrix multiplication and addition is done"

      $build_path/bin/tensor_argmax --my-id 0 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input0 --current-path $build_path > $debug_0/argmax0_layer2.txt &

      pid2=$!
      wait $pid2
      check_exit_statuses $?
      echo "Layer $layer_id: Argmax is done"

         end=$(date +%s)
      

      $build_path/bin/final_output_provider --my-id 0 --connection-port $cs0_port_cs0_output_receiver --config-input $image_share --current-path $build_path > $debug_0/final_output_provider.txt &
      pid3=$!
      wait $pid3
      echo "Output shares of server 0 sent to the image provider"
      wait $pid5 $pid6 

      echo "Output shares of server 0 received by the Image provider"
      echo "Output shares of server 1 received by the Image provider"
      
      echo "Reconstruction Starts"
      #$build_path/bin/Reconstruct --current-path $image_provider_path 
      $build_path/bin/Reconstruct --current-path $image_provider_path --fractional-bits $fractional_bits
      wait 

awk '{ sum += $1 } END { print sum }' AverageTimeDetails0 >> AverageTime0
#  > AverageTimeDetails0 #clearing the contents of the file

sort -r -g AverageMemoryDetails0 | head  -1 >> AverageMemory0
#  > AverageMemoryDetails0 #clearing the contents of the file

echo -e "\nInferencing Finished"

Mem=`cat AverageMemory0`
Time=`cat AverageTime0`

Mem=$(printf "%.2f" $Mem)
Convert_KB_to_GB=$(printf "%.14f" 9.5367431640625E-7)
Mem2=$(echo "$Convert_KB_to_GB * $Mem" | bc -l)

Memory=$(printf "%.3f" $Mem2)

echo "Memory requirement:" `printf "%.3f" $Memory` "GB"
echo "Time taken by inferencing task:" $Time "ms"
echo "Elapsed Time: $(($end-$start)) seconds"

cd $scripts_path


