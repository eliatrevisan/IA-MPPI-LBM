python3 train_roboat.py --exp_num 1000 --model_name VGDNN --total_training_steps 1000 --n_mixtures 1 --rnn_state_size 32 --rnn_state_size_lstm_grid 128 --rnn_state_size_lstm_ped 128 --rnn_state_size_lstm_concat 256 --latent_space_size 64 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 14 --prediction_horizon 24 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;

# python3 test_VGDNN.py --exp_num 1000 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 1000 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 1000 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 1000 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 1000 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_amstel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

