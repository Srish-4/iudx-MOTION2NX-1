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

in_rows=148
# strides=1
# kernel=2

# out_rows=$(( (in_rows-kernel)/strides +1 ))

# echo $out_rows

# no_user_split=2
# no_of_out_rows_per_split=$(( out_rows/no_user_split ))

# temp=0

###############################################################################
image_config="remote_image_shares"
model_config=${BASE_DIR}/config_files/model_helpernode_config.json
build_path=${BASE_DIR}/build_debwithrelinfo_gcc
debug_1=${BASE_DIR}/logs/server1/
smpc_config_path=${BASE_DIR}/config_files/smpc-pn-helpernode-config.json
smpc_config=`cat $smpc_config_path`
scripts_path=${BASE_DIR}/scripts

# #####################################################################################################################################
cd $build_path

if [ -f MemoryDetails1 ]; then
   rm MemoryDetails1
   # echo "Memory Details0 are removed"
fi

if [ -f AverageMemoryDetails1 ]; then
   rm AverageMemoryDetails1
   # echo "Average Memory Details0 are removed"
fi

if [ -f AverageMemory1 ]; then
   rm AverageMemory1
   # echo "Average Memory Details0 are removed"
fi

if [ -f AverageTimeDetails1 ]; then
   rm AverageTimeDetails1
   # echo "AverageTimeDetails0 is removed"
fi

if [ -f AverageTime1 ]; then
   rm AverageTime1
   # echo "AverageTime0 is removed"
fi

# #####################Inputs##########################################################################################################
# Do dns reolution or not 
cs0_dns_resolve=`echo $smpc_config | jq -r .cs0_dns_resolve`
cs1_dns_resolve=`echo $smpc_config | jq -r .cs1_dns_resolve`
helpernode_dns_resolve=`echo $smpc_config | jq -r .helpernode_dns_resolve`


# cs0_host is the ip/domain of server0, cs1_host is the ip/domain of server1
cs0_host=`echo $smpc_config | jq -r .cs0_host`
cs1_host=`echo $smpc_config | jq -r .cs1_host`
helpernode_host=`echo $smpc_config | jq -r .helpernode_host`
reverse_ssh_host=`echo $smpc_config | jq -r .reverse_ssh_host`


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

# Port on which final output talks to image provider 
cs0_port_cs1_output_receiver=`echo $smpc_config | jq -r .cs0_port_cs1_output_receiver`


# Ports on which server0 and server1 of the inferencing tasks talk to each other
cs0_port_inference=`echo $smpc_config | jq -r .cs0_port_inference`
cs1_port_inference=`echo $smpc_config | jq -r .cs1_port_inference`
helpernode_port_inference=`echo $smpc_config | jq -r .helpernode_port_inference`
relu0_port_inference=`echo $smpc_config | jq -r .relu0_port_inference`
relu1_port_inference=`echo $smpc_config | jq -r .relu1_port_inference`

number_of_layers=`echo $smpc_config | jq -r .number_of_layers`
fractional_bits=`echo $smpc_config | jq -r .fractional_bits`


if [ -f $build_path/server1/maxpool_temp1 ]; then
   rm $build_path/server1/maxpool_temp1
   # echo "AverageTime0 is removed"
fi

if [ -f $build_path/server1/maxpool_output1 ]; then
   rm $build_path/server1/maxpool_output1
   # echo "AverageTime0 is removed"
fi

ch=$(awk 'NR==1 {print $1}' server1/cnn_outputshare_1)
in_rows=$(awk 'NR==1 {print $2}' server1/cnn_outputshare_1)
echo "in_rows : " $in_rows
strides=2
kernel=2

out_rows=$(( (in_rows-kernel)/strides +1 ))

echo $out_rows

no_user_split=5
no_of_out_rows_per_split=$(( out_rows/no_user_split ))

temp=0

#########################################################################################################
# touch server1/maxpool_output1

for ((i=1; i<=$no_user_split; i++)); do 
    if [ "$i" -eq "$no_user_split" ] && [ $((out_rows % no_user_split)) -ne 0 ]; then
        no_of_out_rows_per_split=$((no_of_out_rows_per_split+out_rows % no_user_split))
        echo $no_of_out_rows_per_split
    fi

    start=$((temp + 1))
    end=$((start + strides * (no_of_out_rows_per_split - 1) + kernel - 1))
    temp=$((temp + strides * no_of_out_rows_per_split))

    echo $start $end



   $build_path/bin/tensor_maxpool_split --my-id 1 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath cnn_outputshare --current-path $build_path --strides $strides --pool-size $kernel --start_row $start --end_row $end> $debug_1/tensor_gt_relu1_layer${layer_id}.txt &
   pid1=$!
   wait $pid1
   check_exit_statuses $?

   echo "Layer $layer_id: Maxpool is done" 

   $build_path/bin/tensor_gt_relu --my-id 1 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input1 --current-path $build_path > $debug_1/tensor_gt_relu1_layer${layer_id}.txt &
   pid2=$!
   wait $pid2
   check_exit_statuses $?
   echo "Layer $layer_id: ReLU is done"
   
   
   tail -n +2 server1/outputshare_1 >> server1/maxpool_temp1
done 

# cat server1/maxpool_temp1 >> server1/maxpool_output1
# cp server1/maxpool_output1 server1/cnn_outputshare_1