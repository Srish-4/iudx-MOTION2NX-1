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
# paths required to run cpp files
image_config="remote_image_shares"
build_path=${BASE_DIR}/build_debwithrelinfo_gcc
image_path=${BASE_DIR}/data/ImageProvider
image_provider_path=${BASE_DIR}/data/ImageProvider/Final_Output_Shares
debug_0=${BASE_DIR}/logs/server0
scripts_path=${BASE_DIR}/scripts
smpc_config_path=${BASE_DIR}/config_files/cifar-smpc-cnn-config.json
smpc_config=`cat $smpc_config_path`
#####################Inputs#########################################################################################################
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
cs0_port_model_receiver=`echo $smpc_config | jq -r .cs0_port_model_receiver`
cs0_port_image_receiver=`echo $smpc_config | jq -r .cs0_port_image_receiver`

# Ports on which Image provider listens for final inference output
cs0_port_cs0_output_receiver=`echo $smpc_config | jq -r .cs0_port_cs0_output_receiver`
cs0_port_cs1_output_receiver=`echo $smpc_config | jq -r .cs0_port_cs1_output_receiver`

# Ports on which server0 and server1 of the inferencing tasks talk to each other
cs0_port_inference=`echo $smpc_config | jq -r .cs0_port_inference`
cs1_port_inference=`echo $smpc_config | jq -r .cs1_port_inference`
relu0_port_inference=`echo $smpc_config | jq -r .relu0_port_inference`
relu1_port_inference=`echo $smpc_config | jq -r .relu1_port_inference`
fractional_bits=`echo $smpc_config | jq -r .fractional_bits`

# Index of the image for which inferencing task is run
image_id=`echo $smpc_config | jq -r .image_id`
image_share="remote_image_shares"

# #number of splits
# splits=`echo "$smpc_config" | jq -r .splits`

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

#########################Image Share Receiver ############################################################################################
echo "Image shares receiver starts"

$build_path/bin/Image_Share_Receiver_CNN --my-id 0 --port $cs0_port_image_receiver --fractional-bits $fractional_bits --file-names $image_config --current-path $build_path > $debug_0/Image_Share_Receiver0.txt &
pid2=$!

######################### Weights Share Receiver ############################################################################################
# Here if weights already exist, read from existing files
# if W1 exists , we assume , it has received other weights also.  
# otherwise execute weights_share_receiver to receive shares from weights provider.
if [ -f server0/W1 ]; then 
    echo "Weights exist"

else

echo "Weight shares receiver starts"
$build_path/bin/Weights_Share_Receiver_CNN --my-id 0 --port $cs0_port_model_receiver --current-path $build_path > $debug_0/Weights_Share_Receiver0.txt &
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
######################### Share receivers end ############################################################################################

########################Inferencing task starts ###############################################################################################
echo "Inferencing task of the image shared starts"
start=$(date +%s)

cp server0/Image_shares/remote_image_shares server0/outputshare_0
cp server0/Image_shares/remote_image_shares server0/cnn_outputshare_0
sed -i "1s/^[^ ]* //" server0/outputshare_0

layer_types=($(cat layer_types0))
number_of_layers=${layer_types[0]}

split_info=$(echo "$smpc_config" | jq -r '.split_layers_genr[]')
#Added by Sarthak HS
split_info_index=0
split_info_layers=($(echo $split_info | jq -r '.layer_id'))
split_info_length=${#split_info_layers[@]}

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
      # ((split_info_index++))
   fi

   if [ ${layer_types[layer_id]} -eq 0 ] && [ $num_splits -eq 1 ];
   then
      input_config="outputshare"

      $build_path/bin/tensor_gt_mul_test --my-id 0 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --config-file-input $input_config --config-file-model file_config_model0 --layer-id $layer_id --current-path $build_path > $debug_0/tensor_gt_mul0_layer${layer_id}.txt &
      pid1=$!
      wait $pid1 
      check_exit_statuses $?
      echo "Layer $layer_id: Matrix multiplication and addition is done"
      
      $build_path/bin/tensor_gt_relu --my-id 0 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input0 --current-path $build_path > $debug_0/tensor_gt_relu0_layer${layer_id}.txt &
      pid1=$!
      wait $pid1
      check_exit_statuses $?
      echo "Layer $layer_id: ReLU is done"
   
   elif [ ${layer_types[layer_id]} -eq 0 ] && [ $num_splits -gt 1 ];
   then
      cp $build_path/server0/outputshare_0  $build_path/server0/split_input_0
      input_config="split_input"

      num_rows=$(jq -r '.rows' <<< "$split");
      echo "Number of splits for layer $layer_id matrix multiplication: $num_rows::$num_splits"
      x=$(($num_rows/$num_splits))
      for(( m = 1; m <= $num_splits; m++ )); do 
         let l=$((m-1)) 
         let a=$((l*x+1)) 
         let b=$((m*x)) 
         let r=$((l*x))
         
         $build_path/bin/tensor_gt_mul_split --my-id 0 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --config-file-input $input_config --config-file-model file_config_model0 --layer-id $layer_id --row_start $a --row_end $b --split $num_splits --current-path $build_path > $debug_0/tensor_gt_mul0_layer${layer_id}_split.txt &
         pid1=$!
         wait $pid1
         check_exit_statuses $? 
         echo "Layer $layer_id, split $m: Matrix multiplication and addition is done."
         if [ $m -eq 1 ]; then
            touch finaloutput_0
            printf "$r 1\n" > finaloutput_0
            $build_path/bin/appendfile 0
            pid1=$!
            wait $pid1 
            check_exit_statuses $?
         else 
            $build_path/bin/appendfile 0
            pid1=$!
            wait $pid1 
            check_exit_statuses $?
         fi

         sed -i "1s/${r} 1/${b} 1/" finaloutput_0
      done

      cp finaloutput_0  $build_path/server0/outputshare_0 
      if [ -f finaloutput_0 ]; then
         rm finaloutput_0
      fi
      if [ -f server0/split_input_0 ]; then
         rm server0/split_input_0
      fi
      check_exit_statuses $?
      
      $build_path/bin/tensor_gt_relu --my-id 0 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input0 --current-path $build_path > $debug_0/tensor_gt_relu0_layer${layer_id}.txt &
      pid1=$!
      wait $pid1
      check_exit_statuses $?
      echo "Layer $layer_id: ReLU is done"
   
   elif [ ${layer_types[layer_id]} -eq 1 ] && [ $num_splits -eq 1 ];
   then
      input_config="cnn_outputshare"

      $build_path/bin/conv --my-id 0 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --config-file-input $input_config --config-file-model file_config_model0 --layer-id $layer_id --current-path $build_path 
      #> $debug_0/cnn0_layer${layer_id}.txt &
      pid1=$!
      wait $pid1
      check_exit_statuses $?
      echo "Layer $layer_id: Convolution is done"

      $build_path/bin/tensor_gt_relu --my-id 0 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input0 --current-path $build_path > $debug_0/tensor_gt_relu0_layer${layer_id}.txt &
      pid1=$!
      wait $pid1
      check_exit_statuses $?
      echo "Layer $layer_id: ReLU is done"
      tail -n +2 server0/outputshare_0 >> server0/cnn_outputshare_0
   # PLEase change numspilt condition
   elif [ ${layer_types[layer_id]} -eq 1 ] && [ $num_splits -gt 1 ]; 
   then
      cp $build_path/server0/cnn_outputshare_0  $build_path/server0/split_input_0
      input_config="split_input"

      echo "Number of splits for layer $layer_id convolution: $num_splits"
      num_kernels=$(jq -r '.kernels' <<< "$split");
      #Added by Sarthak for HS
      #Added by Sarhtak FOR HS
      # d_rows=`echo $smpc_config | jq -r .image_rows`
      # echo 'THese are the D_rows' $d_rows
      # d_cols=`echo $smpc_config | jq -r .image_cols`


      read d_channels d_rows d_cols <<< $(head -n 1 server0/cnn_outputshare_0)
      # echo 'THese are the D_rows' $d_rows
      # echo 'THese are the D_cols' $d_cols
      # echo 'THese are the D_channels' $d_channels


      #split=$(jq -r ".split_layers_genr[$split_info_index]" <<< "$smpc_config");
      u_splits=$(jq -r ".split_layers_genr[$split_info_index].row_split" <<< "$smpc_config");
      # echo 'USER SPLITS' $u_splits
      kernel_rows=$(jq -r ".split_layers_genr[$split_info_index].kernel_rows" <<< "$smpc_config");
      # echo 'Kernel Rows' $kernel_rows
      kernel_cols=$(jq -r ".split_layers_genr[$split_info_index].kernel_cols" <<< "$smpc_config");
      strides=($(jq -r ".split_layers_genr[$split_info_index].strides[0] " <<< "$smpc_config"))
      # echo 'Strifes' $strides
      strides=${strides[0]}
      
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

               $build_path/bin/conv_h_split --my-id 0 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --config-file-input $input_config --config-file-model file_config_model0 --layer-id $layer_id --kernel_start $a --kernel_end $b --row_start $r_start --row_end $r_end --current-path $build_path --h_actual_split $actual_splits --h_split_index $((k-1)) > $debug_0/cnn0_layer${layer_id}_split.txt &
               pid1=$!
               wait $pid1
               check_exit_statuses $? 
               echo "Layer $layer_id, kernel $m , Horizontal_split $((k)) : Convolution is done."

               $build_path/bin/tensor_gt_relu --my-id 0 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input0 --current-path $build_path > $debug_0/tensor_gt_relu0_layer${layer_id}.txt &
               pid1=$!
               wait $pid1
               check_exit_statuses $?
               echo "Layer $layer_id, kernel $m , Horizontal Split: $((k)) : ReLU is done"
               tail -n +2 server0/outputshare_0 >> server0/final_outputshare_0
               tail -n +2 server0/outputshare_0 >> server0/cnn_outputshare_0

            done
        
         echo "Layer $layer_id, kernel $m: Convolution is done."
         cp server0/final_outputshare_0  server0/outputshare_0 

      done

      # cp server0/final_outputshare_0  server0/outputshare_0 
      # if [ -f server0/final_outputshare_0 ]; then
      #    rm server0/final_outputshare_0
      # fi
      # if [ -f server0/split_input_0 ]; then
      #    rm server0/split_input_0
      # fi
      # check_exit_statuses $?

      # $build_path/bin/tensor_gt_relu --my-id 0 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input0 --current-path $build_path > $debug_0/tensor_gt_relu0_layer${layer_id}.txt &
      # pid1=$!
      # wait $pid1
      # check_exit_statuses $?
      # echo "Layer $layer_id: ReLU is done"
      # tail -n +2 server0/outputshare_0 >> server0/cnn_outputshare_0
   fi
   ((split_info_index++))
done

####################################### Argmax  ###########################################################################
# $build_path/bin/argmax --my-id 0 --threads 1 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol beavy --config-filename file_config_input0 --config-input $image_share --current-path $build_path > $debug_0/argmax0_layer${layer_id}.txt &
$build_path/bin/tensor_argmax --my-id 0 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input0 --current-path $build_path > $debug_0/argmax0_layer2.txt &

pid1=$!
wait $pid1
check_exit_statuses $?
echo "Layer $layer_id: Argmax is done"

end=$(date +%s)
####################################### Final output provider  ###########################################################################

$build_path/bin/final_output_provider --my-id 0 --connection-ip $reverse_ssh_host --connection-port $cs0_port_cs0_output_receiver --config-input $image_share --current-path $build_path > $debug_0/final_output_provider.txt &
pid3=$!
wait $pid3
check_exit_statuses $?
echo "Output shares of server 0 sent to the image provider"


wait $pid5 $pid6 

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
done