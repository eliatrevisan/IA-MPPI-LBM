: <<'END'

python3 train_VGDNN.py --exp_num 110 --model_name STORN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 110 --model_name STORN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 110 --model_name STORN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 111 --model_name STORN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 111 --model_name STORN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 111 --model_name STORN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 112 --model_name STORN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 112 --model_name STORN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 112 --model_name STORN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 113 --model_name STORN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 113 --model_name STORN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 113 --model_name STORN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 115 --model_name STORN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 115 --model_name STORN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 115 --model_name STORN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 114 --model_name STORN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 114 --model_name STORN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 114 --model_name STORN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 116 --model_name STORN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 116 --model_name STORN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 116 --model_name STORN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 117 --model_name STORN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 117 --model_name STORN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 117 --model_name STORN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 100 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 100 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 100 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 101 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 101 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 101 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 102 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 102 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 102 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 103 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 103 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 103 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 105 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 105 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 105 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 104 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 104 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 104 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 106 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 106 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 106 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 107 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 107 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 107 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 103 --model_name SocialVDGNNFull --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6;
python3 test_VGDNN.py --exp_num 103 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 103 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 107 --model_name SocialVDGNNFull --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6;
python3 test_VGDNN.py --exp_num 107 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 107 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 100 --model_name SocialVDGNNFull --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6;
python3 test_VGDNN.py --exp_num 100 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 100 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 101 --model_name SocialVDGNNFull --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6;
python3 test_VGDNN.py --exp_num 101 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 101 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 102 --model_name SocialVDGNNFull --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6;
python3 test_VGDNN.py --exp_num 102 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 102 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 104 --model_name SocialVDGNNFull --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6;
python3 test_VGDNN.py --exp_num 104 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 104 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 105 --model_name SocialVDGNNFull --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6;
python3 test_VGDNN.py --exp_num 105 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 105 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 106 --model_name SocialVDGNNFull --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6;
python3 test_VGDNN.py --exp_num 106 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 106 --model_name SocialVDGNNFull --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 111 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 111 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 111 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 112 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 112 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 112 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 113 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 113 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 113 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 115 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 115 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 115 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 114 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 114 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 114 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 116 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 116 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 116 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 117 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 117 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 117 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 110 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info angular_grid --pedestrian_vector_dim 36 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 110 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 110 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 103 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 103 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 103 --model_name VRNN --num_test_sequences 100 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 107 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 107 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 107 --model_name VRNN --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 101 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 101 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 101 --model_name VRNN --num_test_sequences 100 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 102 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 102 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 102 --model_name VRNN --num_test_sequences 100 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


python3 train_VGDNN.py --exp_num 105 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 105 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 105 --model_name VRNN --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 104 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 104 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 104 --model_name VRNN --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 106 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 106 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 106 --model_name VRNN --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 107 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 107 --model_name VRNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 107 --model_name VRNN --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 100 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 100 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 100 --model_name VRNN --num_test_sequences 100 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


python3 train_VGDNN.py --exp_num 111 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_02 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 111 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_02 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 111 --model_name VRNN --num_test_sequences 100 --scenario real_world/zara_02 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 112 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_02 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 112 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_02 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 112 --model_name VRNN --num_test_sequences 100 --scenario real_world/zara_02 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 113 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_02 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 113 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_02 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 113 --model_name VRNN --num_test_sequences 100 --scenario real_world/zara_02 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 115 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_eth --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 115 --model_name VRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_eth --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 115 --model_name VRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_eth --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 114 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_eth --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 114 --model_name VRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_eth --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 114 --model_name VRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_eth --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 116 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_eth --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 116 --model_name VRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_eth --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 116 --model_name VRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_eth --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 117 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_eth --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 117 --model_name VRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_eth --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 117 --model_name VRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_eth --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 110 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_02 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 110 --model_name VRNN --num_test_sequences 10 --scenario real_world/zara_02 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 110 --model_name VRNN --num_test_sequences 100 --scenario real_world/zara_02 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;



python3 train_VGDNN.py --exp_num 122 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_hotel --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 122 --model_name VRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_hotel --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 122 --model_name VRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_hotel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 123 --model_name VRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_hotel --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 123 --model_name VRNN --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_hotel --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 123 --model_name VRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_hotel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 200 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 200 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 200 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 201 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 201 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 201 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 202 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 202 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 202 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 203 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 203 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 203 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 205 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 205 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 205 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 204 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 204 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 204 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 206 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 206 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 206 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 207 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 207 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 207 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


python3 train_VGDNN.py --exp_num 300 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 300 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 300 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 301 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 301 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 301 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 302 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 302 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 302 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 303 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 303 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 303 --model_name VGDNN --num_test_sequences 10 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 305 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 305 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 305 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 304 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 304 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 304 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 306 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 306 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 306 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 307 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 307 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 307 --model_name VGDNN --num_test_sequences 10 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;



python3 train_VGDNN.py --exp_num 103 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 103 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 103 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 107 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 107 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 107 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 101 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 101 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 101 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 102 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 102 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 102 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


python3 train_VGDNN.py --exp_num 105 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 105 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 105 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 104 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 104 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 104 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 106 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 106 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 106 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 107 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 107 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/st --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 107 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 100 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 100 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/zara_01 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 100 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


python3 train_VGDNN.py --exp_num 111 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_02 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 111 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/zara_02 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 111 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/zara_02 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 112 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_02 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 112 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/zara_02 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 112 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/zara_02 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 113 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_02 --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 113 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/zara_02 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 113 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/zara_02 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 115 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_eth --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 115 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_eth --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 115 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_eth --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 114 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_eth --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 114 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_eth --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 114 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_eth --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 116 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_eth --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 116 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_eth --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 116 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_eth --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 117 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_eth --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 117 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/ewap_dataset/seq_eth --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 117 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_eth --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 110 --model_name VRNNwLikelihood --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_02 --gpu false --prev_horizon 0 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
python3 test_VGDNN.py --exp_num 110 --model_name VRNNwLikelihood --num_test_sequences 10 --scenario real_world/zara_02 --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 110 --model_name VRNNwLikelihood --num_test_sequences 100 --scenario real_world/zara_02 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

END

python3 train_VGDNN.py --exp_num 1 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
