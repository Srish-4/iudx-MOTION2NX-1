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

# in_rows=8


# start_arr=()
# end_arr=()
###############################################################################
image_config="remote_image_shares"
build_path=${BASE_DIR}/build_debwithrelinfo_gcc
image_provider_path=${BASE_DIR}/data/ImageProvider/Final_Output_Shares
debug_0=${BASE_DIR}/logs/server0/
scripts_path=${BASE_DIR}/scripts
smpc_config_path=${BASE_DIR}/config_files/smpc-pn-helpernode-config.json
smpc_config=`cat $smpc_config_path`
# #####################Inputs##########################################################################################################
cd $build_path
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


if [ -f $build_path/server0/maxpool_output0 ]; then
   rm $build_path/server0/maxpool_output0
   # echo "AverageTime0 is removed"
fi

if [ -f $build_path/server0/maxpool_temp0 ]; then
   rm $build_path/server0/maxpool_temp0
   # echo "AverageTime0 is removed"
fi

ch=$(awk 'NR==1 {print $1}' server0/cnn_outputshare_0)
in_rows=$(awk 'NR==1 {print $2}' server0/cnn_outputshare_0)
echo "in_rows : " $in_rows
strides=1
kernel=2

out_rows=$(( (in_rows-kernel)/strides +1 ))

echo "out_rows : " $out_rows

no_user_split=4
no_of_out_rows_per_split=$(( out_rows/no_user_split ))

temp=0

##############################################################################
# touch $build_path/server0/maxpool_output0

for((i=1;i<=$no_user_split;i++))
do 

    if [ "$i" -eq "$no_user_split" ] && [ $(( out_rows % no_user_split )) -ne 0 ];then
    no_of_out_rows_per_split=$((no_of_out_rows_per_split+out_rows % no_user_split))
    echo $no_of_out_rows_per_split
    fi

    start=$((temp + 1))
    end=$((start + strides * (no_of_out_rows_per_split - 1) + kernel - 1))
    temp=$((temp + strides * no_of_out_rows_per_split))


   echo "start end" $start $end

   $build_path/bin/tensor_maxpool_split --my-id 0 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath cnn_outputshare --current-path $build_path --strides $strides --pool-size $kernel --start_row $start --end_row $end  > $debug_0/tensor_gt_relu0_layer${layer_id}.txt &
   pid1=$!
   wait $pid1
   check_exit_statuses $?

   echo "Layer $layer_id: Maxpool is done" 

   $build_path/bin/tensor_gt_relu --my-id 0 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input0 --current-path $build_path > $debug_0/tensor_gt_relu0_layer${layer_id}.txt &
   pid2=$!
   wait $pid2
   check_exit_statuses $?
   echo "Layer $layer_id: ReLU is done"

    echo "no of out rows per split" $no_of_out_rows_per_split
   tail -n +2 server0/outputshare_0 >> server0/maxpool_temp0
done

# cat server0/maxpool_temp0 >> server0/maxpool_output0
# cp server/maxpool_output0 server0/cnn_outputshare_0
