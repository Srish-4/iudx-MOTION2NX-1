#!/bin/bash
while : 
do
check_exit_statuses() {
   for status in "$@";
   do
      if [ $status -ne 0 ]; then
         echo "Exiting due to error."
         exit 1  # Exit the script with a non-zero exit code
      fi
   done
}
image_config="remote_image_shares"
build_path=${BASE_DIR}/build_debwithrelinfo_gcc
model_provider_path=${BASE_DIR}/data/ModelProvider
debug_1=${BASE_DIR}/logs/server1
smpc_config_path=${BASE_DIR}/config_files/mnist-smpc-cnn-config.json
smpc_config=`cat $smpc_config_path`
# #####################Inputs##########################################################################################################
# Do dns resolution or not 
cs0_dns_resolve=`echo $smpc_config | jq -r .cs0_dns_resolve`
cs1_dns_resolve=`echo $smpc_config | jq -r .cs1_dns_resolve`
reverse_ssh_dns_resolve=`echo $smpc_config | jq -r .reverse_ssh_dns_resolve`

# cs0_host is the ip/domain of server0, cs1_host is the ip/domain of server1
cs0_host=`echo $smpc_config | jq -r .cs0_host`
cs1_host=`echo $smpc_config | jq -r .cs1_host`
reverse_ssh_host=`echo $smpc_config | jq -r .reverse_ssh_host`

if [[ $cs0_dns_resolve == "true" ]];
then 
cs0_host=`dig +short $cs0_host | grep '^[.0-9]*$' | head -n 1`
fi

if [[ $cs1_dns_resolve == "true" ]];
then 
cs1_host=`dig +short $cs1_host | grep '^[.0-9]*$' | head -n 1`
fi

if [[ $reverse_ssh_dns_resolve == "true" ]];
then 
reverse_ssh_host=`dig +short $reverse_ssh_host | grep '^[.0-9]*$' | head -n 1`
fi

# Ports on which weights,image provider  receiver listens/talks
cs1_port_model_receiver=`echo $smpc_config | jq -r .cs1_port_model_receiver`
cs1_port_image_receiver=`echo $smpc_config | jq -r .cs1_port_image_receiver`     

# Port on which final output talks to image provider 
cs0_port_cs1_output_receiver=`echo $smpc_config | jq -r .cs0_port_cs1_output_receiver`


# Ports on which server0 and server1 of the inferencing tasks talk to each other
cs0_port_inference=`echo $smpc_config | jq -r .cs0_port_inference`
cs1_port_inference=`echo $smpc_config | jq -r .cs1_port_inference`
relu0_port_inference=`echo $smpc_config | jq -r .relu0_port_inference`
relu1_port_inference=`echo $smpc_config | jq -r .relu1_port_inference`
fractional_bits=`echo $smpc_config | jq -r .fractional_bits`

image_share="remote_image_shares"

# #number of splits
# splits=`echo "$smpc_config" | jq -r .splits`

if [ ! -d "$debug_1" ];
then
	mkdir -p $debug_1
fi

cd $build_path


if [ -f finaloutput_1 ]; then
   rm finaloutput_1
   # echo "final output 1 is removed"
fi

if [ -f MemoryDetails1 ]; then
   rm MemoryDetails1
   # echo "Memory Details1 are removed"
fi

if [ -f AverageMemoryDetails1 ]; then
   rm AverageMemoryDetails1
   # echo "Average Memory Details1 are removed"
fi
if [ -f AverageMemory1 ]; then
   rm AverageMemory1
   # echo "Average Memory Details1 are removed"
fi

if [ -f AverageTimeDetails1 ]; then
   rm AverageTimeDetails1
   # echo "AverageTimeDetails1 is removed"
fi

if [ -f AverageTime1 ]; then
   rm AverageTime1
   # echo "AverageTime1 is removed"
fi

#########################Image Share Receiver ############################################################################################
echo "Image shares receiver starts"
$build_path/bin/Image_Share_Receiver_CNN --my-id 1 --port $cs1_port_image_receiver --fractional-bits $fractional_bits --file-names $image_config --current-path $build_path > $debug_1/Image_Share_Receiver.txt &
pid2=$!

#########################Weights Share Receiver ############################################################################################
# Here if weights already exist, read from existing files
# if W1 exists , we assume , it has received other weights also.  
# otherwise execute weights_share_receiver to receive shares from weights provider.
if [ -f server1/W1 ]; then 
    echo "Weights exist"

else

echo "Weight shares receiver starts"
$build_path/bin/Weights_Share_Receiver_CNN --my-id 1 --port $cs1_port_model_receiver --current-path $build_path > $debug_1/Weights_Share_Receiver0.txt &
pid1=$!
wait $pid1
check_exit_statuses $? 

fi

#pid2 is shifted here , so that immediately after image share receiver is run, 
#weight share receiever is run after that (if weights dont exist)

wait $pid2
check_exit_statuses $?

echo "Weight shares received"
echo "Image shares received"
# #########################Share receivers end ############################################################################################

########################Inferencing task starts ###############################################################################################

echo "Inferencing task of the image shared starts"
start=$(date +%s)

cp server1/Image_shares/remote_image_shares server1/outputshare_1
cp server1/Image_shares/remote_image_shares server1/cnn_outputshare_1
sed -i "1s/^[^ ]* //" server1/outputshare_1

layer_types=($(cat layer_types1))
number_of_layers=${layer_types[0]}

split_info=$(echo "$smpc_config" | jq -r '.split_layers_genr[]')
split_info_index=0
split_info_layers=($(echo $split_info | jq -r '.layer_id'))
split_info_length=${#split_info_layers[@]}

# echo 'split_info_length' $split_info_length

for ((layer_id=1; layer_id<=$number_of_layers; layer_id++)); do
   num_splits=1
   echo "###########################"
   # echo "split info index" $split_info_index

   # Check for information in split info
   if [[ $split_info_index -lt $split_info_length ]] && [[ $layer_id -eq ${split_info_layers[split_info_index]} ]];
   then
      split=$(jq -r ".split_layers_genr[$split_info_index]" <<< "$smpc_config");
      echo $split
      num_splits=$(jq -r '.splits' <<< "$split");
      # echo $num_splits
      # ((split_info_index++))
   fi


   if [ ${layer_types[layer_id]} -eq 0 ] && [ $num_splits -eq 1 ];
   then
      input_config="outputshare"

      $build_path/bin/tensor_gt_mul_test --my-id 1 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --config-file-input $input_config --config-file-model file_config_model1 --layer-id $layer_id --current-path $build_path > $debug_1/tensor_gt_mul1_layer${layer_id}.txt &
      pid1=$!
      wait $pid1 
      check_exit_statuses $?
      echo "Layer $layer_id: Matrix multiplication and addition is done"
      
      $build_path/bin/tensor_gt_relu --my-id 1 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input1 --current-path $build_path > $debug_1/tensor_gt_relu1_layer${layer_id}.txt &
      pid1=$!
      wait $pid1
      check_exit_statuses $?
      echo "Layer $layer_id: ReLU is done"
   
   elif [ ${layer_types[layer_id]} -eq 0 ] && [ $num_splits -gt 1 ];
   then
      cp $build_path/server1/outputshare_1  $build_path/server1/split_input_1
      input_config="split_input"

      num_rows=$(jq -r '.rows' <<< "$split");
      echo "Number of splits for layer $layer_id matrix multiplication: $num_rows::$num_splits"
      x=$(($num_rows/$num_splits))
      for(( m = 1; m <= $num_splits; m++ )); do 
         let l=$((m-1))
         let a=$((l*x+1))
         let b=$((m*x))
         let r=$((l*x))
         
         $build_path/bin/tensor_gt_mul_split --my-id 1 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --config-file-input $input_config --config-file-model file_config_model1 --layer-id $layer_id --row_start $a --row_end $b --split $num_splits --current-path $build_path > $debug_1/tensor_gt_mul1_layer${layer_id}_split.txt &
         pid1=$!
         wait $pid1
         check_exit_statuses $? 
         echo "Layer $layer_id, split $m: Matrix multiplication and addition is done."
         if [ $m -eq 1 ]; then
            touch finaloutput_1
            printf "$r 1\n" > finaloutput_1
            $build_path/bin/appendfile 1
            pid1=$!
            wait $pid1 
            check_exit_statuses $?
         else 
            $build_path/bin/appendfile 1
            pid1=$!
            wait $pid1 
            check_exit_statuses $?
         fi
         sed -i "1s/${r} 1/${b} 1/" finaloutput_1
      done

      cp finaloutput_1  $build_path/server1/outputshare_1
      if [ -f finaloutput_1 ]; then
         rm finaloutput_1
      fi
      if [ -f server1/split_input_1 ]; then
         rm server1/split_input_1
      fi
      check_exit_statuses $?

      $build_path/bin/tensor_gt_relu --my-id 1 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input1 --current-path $build_path > $debug_1/tensor_gt_relu1_layer${layer_id}.txt &
      pid1=$!
      wait $pid1
      check_exit_statuses $?
      echo "Layer $layer_id: ReLU is done"

   elif [ ${layer_types[layer_id]} -eq 1 ] && [ $num_splits -eq 1 ];
   then
      input_config="cnn_outputshare"
      
      $build_path/bin/conv --my-id 1 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --config-file-input $input_config --config-file-model file_config_model1 --layer-id $layer_id --current-path $build_path > $debug_1/cnn1_layer${layer_id}.txt &
      pid1=$!
      wait $pid1 
      check_exit_statuses $?
      echo "Layer $layer_id: Convolution is done"

      $build_path/bin/tensor_gt_relu --my-id 1 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input1 --current-path $build_path > $debug_1/tensor_gt_relu1_layer${layer_id}.txt &
      pid1=$!
      wait $pid1
      check_exit_statuses $?
      echo "Layer $layer_id: ReLU is done"
      tail -n +2 server1/outputshare_1 >> server1/cnn_outputshare_1
   
   elif [ ${layer_types[layer_id]} -eq 1 ] && [ $num_splits -gt 1 ];
   then
      cp $build_path/server1/cnn_outputshare_1  $build_path/server1/split_input_1
      input_config="split_input"

      echo "Number of splits for layer $layer_id convolution: $num_splits"
      num_kernels=$(jq -r '.kernels' <<< "$split");
      # echo 'There are number of kernels' $num_kernels
      #Added by Sarthak for HS
      #Added by Sarhtak FOR HS
       d_rows=`echo $smpc_config | jq -r .image_rows`
      # echo 'THese are the D_rows' $d_rows
      d_cols=`echo $smpc_config | jq -r .image_cols`

      read d_channels d_rows d_cols <<< $(head -n 1 server1/cnn_outputshare_1)
      # echo 'THese are the D_rows' $d_rows
      # echo 'THese are the D_cols' $d_cols
      # echo 'THese are the D_channels' $d_channels

      #split=$(jq -r ".split_layers_genr[$split_info_index]" <<< "$smpc_config");
      u_splits=$(jq -r ".split_layers_genr[$split_info_index].row_split" <<< "$smpc_config");
      # echo 'USER SPLITS' $u_splits
      kernel_rows=$(jq -r ".split_layers_genr[$split_info_index].kernel_rows" <<< "$smpc_config");
      # echo 'Kernel Rows' $kernel_rows
      kernel_cols=$(jq -r ".split_layers_genr[$split_info_index].kernel_cols" <<< "$smpc_config");
      # padding=($(jq -r ".split_layers_genr[$split_info_index].pads | @csv" <<< "$smpc_config"));
      # echo 'PADDING' $padding
      strides=($(jq -r ".split_layers_genr[$split_info_index].strides[0] " <<< "$smpc_config"))
      # echo 'Strifes' $strides
      strides=${strides[0]}
      # echo 'strides' $strides
      
       pads0=($(jq -r ".split_layers_genr[$split_info_index].pads[0] " <<< "$smpc_config"))
      #  echo 'pads0 ' $pads0
       pads1=($(jq -r ".split_layers_genr[$split_info_index].pads[1] " <<< "$smpc_config"))
      #  echo 'pads1 ' $pads1
       pads2=($(jq -r ".split_layers_genr[$split_info_index].pads[2] " <<< "$smpc_config"))
      #  echo 'pads2 ' $pads2
       pads3=($(jq -r ".split_layers_genr[$split_info_index].pads[3] " <<< "$smpc_config"))
      #  echo 'pads3 ' $pads3

      op_rows=$(( (d_rows - kernel_rows + pads0 + pads2) / strides ))
      op_cols=$(( (d_cols - kernel_cols + pads1 + pads3) / strides ))

      op_rows=$((op_rows+1))
      op_cols=$((op_cols+1))

      # echo 'op rows after adding 1 :' $op_rows
      # echo 'op cols after adding 1 :' $op_cols

      num_op_rows_per_split=$((op_rows / u_splits))
      echo 'Number of op rows/split :' $num_op_rows_per_split

      if [ $((op_rows - (num_op_rows_per_split*u_splits))) -gt 0 ]; then
         actual_splits=$((u_splits+1))
      else 
         actual_splits=$u_splits
      fi

      echo 'User Splits :' $u_splits
      # echo 'Actual Splits :' $actual_splits

      #********************************************************************

       rowstart_arr=()
      rowend_arr=()
      flag=0

      for ((i=0; i<actual_splits; i++))
      do
        # echo '**********'
            t=$((num_op_rows_per_split * strides))

            if [ $strides -eq 1 ]; then
               row_start=$(( (i*t)+ 1))
            else 
               row_start=$(( (i*t)+ (pads0 % 2) + 1))
            fi

            if [ $row_start -le $d_rows ]; then
               t1=$(( row_start + t - strides ))
               if [ $(( t1 + kernel_rows - 1 )) -lt $d_rows ]; then
                     row_end=$(( t1 + kernel_rows - 1 ))
               else
                     row_end=$d_rows
               fi

            fi
            if [ $i -eq 0 ]; then
               row_start=1
            fi

            # echo "row start in for loop : " $row_start
            #  echo "row end in for loop : " $row_end

            if [ $i -eq $((actual_splits-1)) ]; then 

                if [ $((row_end-row_start+1+pads2)) -lt $kernel_rows ]; then 
                    
                  #   echo "flag set"
                    flag=1
                fi
            fi

            if [ $flag -eq 0 ]; then
                rowstart_arr+=($row_start)
                rowend_arr+=($row_end)
            fi

    done

            if [ $flag -eq 1 ]; then 
            actual_splits=$((actual_splits-1))
            row_end=$d_rows
            rowend_arr[$actual_splits-1]=$row_end
            fi


   #  echo "row_start elements: "
   #  for element in "${rowstart_arr[@]}"; do
   #      echo "$element"
   #  done

   #  echo "row_end elements: "
   #  for element in "${rowend_arr[@]}"; do
   #      echo "$element"
   #  done

   echo "actual splits : " $actual_splits
   echo "Horizontal split for each vertical split : " $actual_splits

   #***************************************************************************************************
      x=$(($num_kernels/$num_splits))
      for(( m = 1; m <= $num_splits; m++ )); do 
         let l=$((m-1)) 
         let a=$((l*x+1))
         let b=$((m*x))
         #enter r_split loop

         ###################################################################
            
            
            for(( k=1; k <= actual_splits; k++)); do 
            
               r_start=$((rowstart_arr[k-1]))
               r_end=$((rowend_arr[k-1]))

               echo "*******************"
               echo $r_start
               echo $r_end
               echo "*******************"

               # echo $build_path
               # echo 'cs0_host' $cs0_host 'cs0_port_inference' $cs0_port_inference 'cs1_host' $cs1_host 'cs1_port_inference' $cs1_port_inference 'fractional_bits' $fractional_bits 'input config' $input_config 'input model' file_config_model0 'layer id' $layer_id 'kernel start' $a 'kernel end' $b 'row start' $row_start 'row end' $row_end 'current path' $build_path
               $build_path/bin/conv_h_split --my-id 1 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --config-file-input $input_config --config-file-model file_config_model1 --layer-id $layer_id --kernel_start $a --kernel_end $b --row_start $r_start --row_end $r_end --current-path $build_path --h_actual_split $actual_splits --h_split_index $((k-1)) > $debug_1/cnn1_layer${layer_id}_split.txt &

               pid1=$!
               wait $pid1
               check_exit_statuses $? 
               echo "Layer $layer_id, kernel $m , Horizontal_split $((k)) : Convolution is done."

               $build_path/bin/tensor_gt_relu --my-id 1 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input1 --current-path $build_path > $debug_1/tensor_gt_relu1_layer${layer_id}.txt &
               pid1=$!
               wait $pid1
               check_exit_statuses $?
               echo "Layer $layer_id, kernel $m , Horizontal Split: $((k)) : ReLU is done"
               tail -n +2 server1/outputshare_1 >> server1/cnn_outputshare_1
               tail -n +2 server1/outputshare_1 >> server1/final_outputshare_1
            
            done

         echo "Layer $layer_id, kernel $m: Convolution is done."
         cp server1/final_outputshare_1  server1/outputshare_1 

      done

   fi
   ((split_info_index++))
done

####################################### Argmax  ###########################################################################

#$build_path/bin/argmax --my-id 1 --threads 1 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol beavy --config-filename file_config_input1 --config-input $image_share --current-path $build_path > $debug_1/argmax1_layer${layer_id}.txt &
$build_path/bin/tensor_argmax --my-id 1 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input1 --current-path $build_path > $debug_1/argmax1_layer${layer_id}.txt &

pid1=$!
wait $pid1
check_exit_statuses $?
echo "Layer $layer_id: Argmax is done"

end=$(date +%s)
####################################### Final output provider  ###########################################################################

$build_path/bin/final_output_provider --my-id 1 --connection-ip $reverse_ssh_host --connection-port $cs0_port_cs1_output_receiver --config-input $image_share --current-path $build_path > $debug_1/final_output_provider.txt &
pid4=$!
wait $pid4 
check_exit_statuses $?  
echo "Output shares of server 1 sent to the Image provider"

wait 
#kill $pid5 $pid6

 awk '{ sum += $1 } END { print sum }' AverageTimeDetails1 >> AverageTime1
#  > AverageTimeDetails1 #clearing the contents of the file

  sort -r -g AverageMemoryDetails1 | head  -1 >> AverageMemory1
#  > AverageMemoryDetails1 #clearing the contents of the file

echo -e "\nInferencing Finished"

Mem=`cat AverageMemory1`
Time=`cat AverageTime1`

Mem=$(printf "%.2f" $Mem) 
Convert_KB_to_GB=$(printf "%.14f" 9.5367431640625E-7)
Mem2=$(echo "$Convert_KB_to_GB * $Mem" | bc -l)

Memory=$(printf "%.3f" $Mem2)

echo "Memory requirement:" `printf "%.3f" $Memory` "GB"
echo "Time taken by inferencing task:" $Time "ms"
echo "Elapsed Time: $(($end-$start)) seconds"

cd $scripts_path
done