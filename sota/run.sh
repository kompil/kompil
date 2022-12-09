#!/bin/bash
RUN_ONE="sota/run_one.py"
MOCK=""

python3 $RUN_ONE tunnel_4000kp.json $MOCK
python3 $RUN_ONE tunnel_3000kp.json $MOCK
python3 $RUN_ONE tunnel_2000kp.json $MOCK
python3 $RUN_ONE tunnel_1000kp.json $MOCK
python3 $RUN_ONE tunnel_500kp.json $MOCK

python3 $RUN_ONE witcher_4000kp.json $MOCK
python3 $RUN_ONE witcher_3000kp.json $MOCK
python3 $RUN_ONE witcher_2000kp.json $MOCK
python3 $RUN_ONE witcher_1000kp.json $MOCK
python3 $RUN_ONE witcher_500kp.json $MOCK

python3 $RUN_ONE skate_2500kp.json $MOCK
python3 $RUN_ONE skate_2000kp.json $MOCK
python3 $RUN_ONE skate_1500kp.json $MOCK
python3 $RUN_ONE skate_1000kp.json $MOCK
python3 $RUN_ONE skate_500kp.json $MOCK

python3 $RUN_ONE beauty_2000kp.json $MOCK
python3 $RUN_ONE beauty_1500kp.json $MOCK
python3 $RUN_ONE beauty_1000kp.json $MOCK
python3 $RUN_ONE beauty_500kp.json $MOCK
