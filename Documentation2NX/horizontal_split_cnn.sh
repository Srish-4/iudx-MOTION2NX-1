num_kernels=1
kernel_rows=5
kernel_cols=5
d_channels=1
d_rows=28
d_cols=28
strides=2

u_splits=5 #user splits


padding=(1 1 0 0)
op_rows=$(( (d_rows - kernel_rows + padding[0] + padding[2]) / strides ))
op_cols=$(( (d_cols - kernel_cols + padding[1] + padding[3]) / strides ))

op_rows=$((op_rows+1))
op_cols=$((op_cols+1))

echo 'op rows :' $op_rows
echo 'op cols :' $op_cols


num_op_rows_per_split=$((op_rows / u_splits))
echo 'Number of op rows/split :' $num_op_rows_per_split

if [ $((op_rows - (num_op_rows_per_split*u_splits))) -gt 0 ]; then
    actual_splits=$((u_splits+1))
else 
    actual_splits=$u_splits
fi

echo 'User Splits :' $u_splits
echo 'Actual Splits :' $actual_splits

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
               row_start=$(( (i*t)+ (padding[0] % 2) + 1))
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

            echo "row start in for loop : " $row_start
             echo "row end in for loop : " $row_end

            if [ $i -eq $((actual_splits-1)) ]; then 

                if [ $((row_end-row_start+1+padding[2])) -lt $kernel_rows ]; then 
                    
                    echo "flag set"
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


    echo "row_start elements: "
    for element in "${rowstart_arr[@]}"; do
        echo "$element"
    done

    echo "row_end elements: "
    for element in "${rowend_arr[@]}"; do
        echo "$element"
    done

    echo "actual splits : " $actual_splits


     for(( k=1; k <= $actual_splits; k++)); do 
            
               r_start=${row_start[$k-1]}
               #${mahabharata[$j]}
               r_end=${row_end[$k-1]}

               echo "r start: " $r_start
               echo "r_end: " $r_end

      done


                    
   



