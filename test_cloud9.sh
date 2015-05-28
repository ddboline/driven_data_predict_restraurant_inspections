#!/bin/bash

scp ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/driven_data_predict_restraurant_inspections/predict_restraurant_inspections.tar.gz .
tar zxvf predict_restraurant_inspections.tar.gz
rm predict_restraurant_inspections.tar.gz

./text_features.py > output.out 2> output.err ### do this beforehand...
# ./split_csv.py > output.out 2> output.err ### do this beforehand...
# ./load_data.py > output.out 2> output.err 
# ./my_model.py > output.out 2> output.err

# D=`date +%Y%m%d%H%M%S`
# tar zcvf output_${D}.tar.gz model.pkl.gz output.out output.err
# scp model_*.pkl.gz ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/driven_data_predict_restraurant_inspections/
ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk done_driven_data_predict_restraurant_inspections"
echo "JOB DONE predict_restraurant_inspections"
