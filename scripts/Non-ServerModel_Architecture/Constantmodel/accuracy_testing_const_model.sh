#!/bin/bash
debug_0=${BASE_DIR}/logs/server0
debug_1=${BASE_DIR}/logs/server1
smpc_config_path=${BASE_DIR}/config_files/smpc-constant-config.json
# echo $smpc_config_path
smpc_config=`cat $smpc_config_path`
image_id=`echo $smpc_config | jq -r .image_id`

echo "This script is about to run another script."
for (( c=1; c<=2; c++ ))
do 

jq --arg new_id "$c" '.image_id = ($new_id | tonumber)' $smpc_config_path | sponge $smpc_config_path

bash ./remote_server0_constmodel.sh > $debug_0/Script0.txt &
bash ./remote_server1_constmodel.sh > $debug_1/Script1.txt &
wait 

done
echo "This script has just run another script."
