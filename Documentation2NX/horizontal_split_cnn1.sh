num_kernels=1
kernel_rows=3
kernel_cols=3
d_channels=1
d_rows=7
d_cols=3
strides=1


u_splits=2 #user splits


padding=(1 1 1 1)
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

for ((i=0; i<actual_splits; i++))
do
    echo '**********'
    t=$((num_op_rows_per_split * strides))

    #Calculating Start row
    if [ $strides -eq 1 ]; then
        start_row=$(( (i*t)+ 1))
    else 
        start_row=$(( (i*t)+ (padding[0] % 2) + 1))
    fi

    #Calculating End row
    if [ $start_row -le $d_rows ]; then
        t1=$(( start_row + t - strides ))
        if [ $(( t1 + kernel_rows - 1 )) -lt $d_rows ]; then 
            end_row=$(( t1 + kernel_rows - 1 )) 
        else
            end_row=$d_rows
        fi

    fi

    #Startrow will always be 1 for the first split
    if [ $i -eq 0 ]; then
        start_row=1
    fi
    
    echo $start_row
    echo $end_row
done


