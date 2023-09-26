: <<'END'
python3 train_keras.py --exp_num 7 --rnn_state_size 32 --rnn_state_size_lstm_ped 32 --rnn_state_size_bilstm_ped 16 --rnn_state_size_lstm_concat 64 --fc_hidden_unit_size 64;
python3 test_keras.py --exp_num 7;
python3 test_keras.py --exp_num 7 --freeze_other_agents true;

python3 train_keras.py --exp_num 8 --rnn_state_size 32 --rnn_state_size_lstm_ped 64 --rnn_state_size_bilstm_ped 32 --rnn_state_size_lstm_concat 128 --fc_hidden_unit_size 128;
python3 test_keras.py --exp_num 8;
python3 test_keras.py --exp_num 8 --freeze_other_agents true;

python3 train_keras.py --exp_num 9 --rnn_state_size 32 --rnn_state_size_lstm_ped 64 --rnn_state_size_bilstm_ped 32 --rnn_state_size_lstm_concat 256 --fc_hidden_unit_size 128;
python3 test_keras.py --exp_num 9;
python3 test_keras.py --exp_num 9 --freeze_other_agents true;

python3 test_keras.py --exp_num 9 --data_path ../data/2_agents_random/trajs/ --scenario GA3C-CADRL-10;

python3 test_keras.py --exp_num 9 --data_path ../data/ --scenario 2_agents_random/trajs/GA3C-CADRL-10 --freeze_other_agents true;

python3 train_keras.py --exp_num 10 --scenario 2_agents_random/trajs/GA3C-CADRL-10 --scenario_val 2_agents/trajs/GA3C-CADRL-10 --rnn_state_size 32 --rnn_state_size_lstm_ped 64 --rnn_state_size_bilstm_ped 32 --rnn_state_size_lstm_concat 256 --fc_hidden_unit_size 128;

python3 test_keras.py --exp_num 10 --scenario 2_agents/trajs/GA3C-CADRL-10;
python3 test_keras.py --exp_num 10 --scenario 2_agents_swap/trajs/GA3C-CADRL-10-py27;
python3 test_keras.py --exp_num 10 --scenario 2_agents_random/trajs/GA3C-CADRL-10;

END

# # python3 train_keras_v2.py --exp_num 200 --input1_type vel --input2_type posrel_velabs \
# #     --scenario 2_agents/trajs/GA3C-CADRL-10 --scenario_val 2_agents_random/trajs/GA3C-CADRL-10\
# #     --rnn_state_size 32 --rnn_state_size_lstm_ped 64 --rnn_state_size_bilstm_ped 32 --rnn_state_size_lstm_concat 256 --fc_hidden_unit_size 128;

# python3 test_keras_v2.py --exp_num 200 --scenario 2_agents/trajs/GA3C-CADRL-10;
# python3 test_keras_v2.py --exp_num 200 --scenario 2_agents_random/trajs/GA3C-CADRL-10;

# # python3 train_keras_v2.py --exp_num 201 --input1_type polarvel --input2_type posrel_velabs \
# #     --scenario 2_agents/trajs/GA3C-CADRL-10 --scenario_val 2_agents_random/trajs/GA3C-CADRL-10\
# #     --rnn_state_size 32 --rnn_state_size_lstm_ped 64 --rnn_state_size_bilstm_ped 32 --rnn_state_size_lstm_concat 256 --fc_hidden_unit_size 128;

# python3 test_keras_v2.py --exp_num 201 --scenario 2_agents/trajs/GA3C-CADRL-10;
# python3 test_keras_v2.py --exp_num 201 --scenario 2_agents_random/trajs/GA3C-CADRL-10;

# # python3 train_keras_v2.py --exp_num 202 --input1_type polarvel --input2_type polarposinvdist_polarvelreldiff \
# #     --scenario 2_agents/trajs/GA3C-CADRL-10 --scenario_val 2_agents_random/trajs/GA3C-CADRL-10\
# #     --rnn_state_size 32 --rnn_state_size_lstm_ped 64 --rnn_state_size_bilstm_ped 32 --rnn_state_size_lstm_concat 256 --fc_hidden_unit_size 128;
# python3 test_keras_v2.py --exp_num 202 --scenario 2_agents/trajs/GA3C-CADRL-10;
# python3 test_keras_v2.py --exp_num 202 --scenario 2_agents_random/trajs/GA3C-CADRL-10;

# # python3 train_keras_v2.py --exp_num 203 --input1_type polarvel --input2_type polarposinvdist_polarvelreldiff \
# #     --scenario 2_agents/trajs/GA3C-CADRL-10 --scenario_val 2_agents_random/trajs/GA3C-CADRL-10\
# #     --rnn_state_size 32 --rnn_state_size_lstm_ped 64 --rnn_state_size_bilstm_ped 32 --rnn_state_size_lstm_concat 256 --fc_hidden_unit_size 128;
# python3 test_keras_v2.py --exp_num 203 --scenario 2_agents/trajs/GA3C-CADRL-10;
# python3 test_keras_v2.py --exp_num 203 --scenario 2_agents_random/trajs/GA3C-CADRL-10;

# # python3 train_keras_v2.py --exp_num 204 --input1_type vel --input2_type posinvdist_velrel \
# #     --scenario 2_agents/trajs/GA3C-CADRL-10 --scenario_val 2_agents_random/trajs/GA3C-CADRL-10\
# #     --rnn_state_size 32 --rnn_state_size_lstm_ped 64 --rnn_state_size_bilstm_ped 32 --rnn_state_size_lstm_concat 256 --fc_hidden_unit_size 128;
# python3 test_keras_v2.py --exp_num 204 --scenario 2_agents/trajs/GA3C-CADRL-10;
# python3 test_keras_v2.py --exp_num 204 --scenario 2_agents_random/trajs/GA3C-CADRL-10;

# python3 train_keras_v2.py --exp_num 205 --input1_type polarvel --input2_type polarposinvdist_polarvelrelcos \
#     --scenario 2_agents/trajs/GA3C-CADRL-10 --scenario_val none\
#     --rnn_state_size 32 --rnn_state_size_lstm_ped 64 --rnn_state_size_bilstm_ped 32 --rnn_state_size_lstm_concat 256 --fc_hidden_unit_size 128;
# python3 test_keras_v2.py --exp_num 205 --scenario 2_agents/trajs/GA3C-CADRL-10;
python3 test_keras_v2.py --exp_num 205 --scenario 2_agents_random/trajs/GA3C-CADRL-10;


# LIST OF SCENARIOS
# 2_agents/trajs/GA3C-CADRL-10
# 2_agents_swap/trajs/GA3C-CADRL-10-py27 <-- Running into issues because of lack of 'robot_velocity' key in the dictionary containing the data at each time step
# 2_agents_random/trajs/GA3C-CADRL-10


notify-send "Keras train/test script finished"
