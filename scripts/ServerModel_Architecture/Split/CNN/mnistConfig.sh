#!/bin/bash

build_path=${BASE_DIR}/build_debwithrelinfo_gcc

# Remove files starting with 'W' in server0
if ls ${build_path}/server0/W* 1> /dev/null 2>&1; then
    rm ${build_path}/server0/W*
fi

# Remove files starting with 'W' in server1
if ls ${build_path}/server1/W* 1> /dev/null 2>&1; then
    rm ${build_path}/server1/W*
fi

# Remove files starting with 'B' in server0
if ls ${build_path}/server0/B[1-6]* 1> /dev/null 2>&1; then
    rm ${build_path}/server0/B[1-6]*
fi

# Remove files starting with 'B' in server1
if ls ${build_path}/server1/B[1-6]* 1> /dev/null 2>&1; then
    rm ${build_path}/server1/B[1-6]*
fi

# Update mnist-smpc-cnn-config.json
# echo '{
#     "cs0_host": "20.197.27.117",
#     "cs1_host": "13.200.145.195",
#     "helpernode_host": "15.206.80.191",
#     "reverse_ssh_host": "52.172.238.250",
#      "cs0_dns_resolve": false,
#      "cs1_dns_resolve": false,
#      "reverse_ssh_dns_resolve": false,
#      "cs0_port_model_receiver": 4005,
#      "cs1_port_model_receiver": 4006,
#      "cs0_port_cs0_output_receiver": 4007,
#      "cs0_port_cs1_output_receiver": 4008,
#      "cs0_port_inference": 4009,
#      "cs1_port_inference":4010,
#      "relu0_port_inference": 4012,
#      "relu1_port_inference": 4013,
#      "cs0_port_image_receiver": 4014,
#      "cs1_port_image_receiver": 4015,
#      "fractional_bits": 13,
#      "image_id": 1, 
#      "image_rows": 28,
#      "image_cols": 28,
#      "channels": 1, 
#      "split_layers_genr": [{"layer_id":1, "kernels":5, "splits":1, "kernel_rows":5, "kernel_cols":5, "pads":[0,0,0,0], "strides": [1,1], "row_split": 1},
#                            {"layer_id":2, "kernels":3, "splits":3, "kernel_rows":4, "kernel_cols":4, "pads":[0,0,0,0], "strides": [1,1], "row_split": 1},
#                            {"layer_id":3, "rows":100, "splits":20},
#                            {"layer_id":4, "rows":10, "splits":2}]
# }' > ${BASE_DIR}/config_files/mnist-smpc-cnn-config.json

echo '{
      "cs0_host": "127.0.0.1",
      "cs1_host": "127.0.0.1",
      "reverse_ssh_host": "127.0.0.1",
      "cs0_dns_resolve": false,
      "cs1_dns_resolve": false,
      "reverse_ssh_dns_resolve": false,
      "cs0_port_model_receiver": 4005,
      "cs1_port_model_receiver": 4006,
      "cs0_port_cs0_output_receiver": 4007,
      "cs0_port_cs1_output_receiver": 4008,
      "cs0_port_inference": 4009,
      "cs1_port_inference":4010,
      "relu0_port_inference": 4012,
      "relu1_port_inference": 4013,
      "cs0_port_image_receiver": 4014,
      "cs1_port_image_receiver": 4015,
      "fractional_bits": 13,
      "image_id": 1, 
      "image_rows": 28,
      "image_cols": 28,
      "channels": 1, 
      "split_layers_genr": [{"layer_id":1, "kernels":5, "splits":1, "kernel_rows":5, "kernel_cols":5, "pads":[0,0,0,0], "strides": [1,1], "row_split": 1},
                            {"layer_id":2, "kernels":3, "splits":3, "kernel_rows":4, "kernel_cols":4, "pads":[0,0,0,0], "strides": [1,1], "row_split": 1},
                            {"layer_id":3, "rows":100, "splits":20},
                            {"layer_id":4, "rows":10, "splits":2}]
}' > ${BASE_DIR}/config_files/mnist-smpc-cnn-config.json

# Update image_config.json
# echo '{
#     "cs0_host": "20.197.27.117",
#     "cs1_host": "13.200.145.195",
#     "cs0_dns_resolve": false,
#     "cs1_dns_resolve": false,
#     "cs0_port_cs0_output_receiver": 4007,
#     "cs0_port_cs1_output_receiver": 4008,
#     "cs0_port_image_receiver": 4014,
#     "cs1_port_image_receiver": 4015,
#     "fractional_bits": 13,
#     "image_id": 1,
#     "image_rows": 28,
#     "image_cols": 28,
#     "channels": 1
# }' > ${BASE_DIR}/config_files/image_config.json

echo '{
    "cs0_host": "127.0.0.1",
    "cs1_host": "127.0.0.1",
    "cs0_dns_resolve": false,
    "cs1_dns_resolve": false,
    "cs0_port_cs0_output_receiver": 4007,
    "cs0_port_cs1_output_receiver": 4008,
    "cs0_port_image_receiver": 4014,
    "cs1_port_image_receiver": 4015,
    "fractional_bits": 13,
    "image_id": 1,
    "image_rows": 28,
    "image_cols": 28,
    "channels": 1
}' > ${BASE_DIR}/config_files/image_config.json

# Update model_cnn_config.json
echo '{
    "no_of_layers" : 4,
    "Layers" : {
        "1" : {
            "Weights" : {"type": "CN", "kernels": 5, "channels": 1, "rows" : 5, "columns" : 5, "pads": [1,1,0,0], "strides": [2,2], "file_name" : "2CN/W1.csv"}, 
            "Bias" : {"rows" : 5, "columns" : 1, "file_name" : "2CN/B1.csv"}},
        "2" : {
            "Weights" : {"type": "CN", "kernels": 3, "channels": 5, "rows" : 4, "columns" : 4, "pads":[1,1,0,0], "strides": [2,2], "file_name" :"2CN/W2.csv"},
            "Bias" : {"rows": 3, "columns" : 1, "file_name": "2CN/B2.csv"}},
        "3" : {
            "Weights" : {"rows" : 100, "columns" : 108, "file_name" : "2CN/W3.csv"},
            "Bias" : {"rows" : 100, "columns" : 1, "file_name" : "2CN/B3.csv"}},
        "4" : {
            "Weights" : {"rows" : 10, "columns" : 100, "file_name" : "2CN/W4.csv"},
            "Bias" : {"rows" : 10, "columns" : 1, "file_name" : "2CN/B4.csv"}}
    }
}' > ${BASE_DIR}/config_files/model_cnn_config.json
