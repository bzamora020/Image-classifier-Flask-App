# !/bin/bash
#0.008 0.010 0.012 0.014 0.016 0.018 0.020 0.022 0.024 0.026 0.028 0.030 0.032 0.03 0.034


for i in 0.008 0.010 0.012 0.014 0.016 0.018 0.020 0.022 0.024 0.026 0.028 0.030 0.032 0.03 0.034;
do 
python classifier_multiple.py --testing_file /data/ersp21/classifier/test_data.txt --threshold $i;
done
 
