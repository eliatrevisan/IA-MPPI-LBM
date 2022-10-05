

# 3 MIXTURES, NO DIVERSITY

# DIFFERENT SUBMAP SIZES
: '
python3 train_VGDNN.py --exp_num 134 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 60 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 134 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 135 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 12 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 61 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 135 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 136 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 62 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 136 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 137 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 16 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 63 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 137 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 138 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 18 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 64 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 138 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 139 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 65 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 139 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT COVARIANCE

python3 train_VGDNN.py --exp_num 140 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 8.0;
#python3 compare_VGDNN.py --exp_num 66 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 140 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 141 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 67 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 141 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 142 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 12.0;
#python3 compare_VGDNN.py --exp_num 68 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 142 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 143 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 14.0;
#python3 compare_VGDNN.py --exp_num 69 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 143 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 145 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 16.0;
#python3 compare_VGDNN.py --exp_num 70 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 145 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT HORIZONS

python3 train_VGDNN.py --exp_num 146 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 71 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 146 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 147 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 10 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 72 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 147 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 148 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 12 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 73 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 148 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 149 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 14 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 74 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 149 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 150 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 16 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 75 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 150 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;



# 3 MIXTURES, NO DIVERSITY

# DIFFERENT SUBMAP SIZES

python3 train_VGDNN.py --exp_num 151 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 60 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 151 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 152 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 12 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 61 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 152 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 153 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 62 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 153 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 154 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 16 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 63 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 154 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 155 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 18 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 64 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 155 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 156 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 65 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 156 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT COVARIANCE

python3 train_VGDNN.py --exp_num 157 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 8.0;
#python3 compare_VGDNN.py --exp_num 66 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 157 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
'
python3 train_VGDNN.py --exp_num 158 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 67 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 158 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 159 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 12.0;
#python3 compare_VGDNN.py --exp_num 68 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 159 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 160 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 14.0;
#python3 compare_VGDNN.py --exp_num 69 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 160 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 161 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 16.0;
#python3 compare_VGDNN.py --exp_num 70 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 161 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT HORIZONS

python3 train_VGDNN.py --exp_num 162 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 71 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 162 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 163 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 10 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 72 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 163 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 164 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 12 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 73 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 164 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 165 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 14 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 74 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 165 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 166 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 16 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 75 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 166 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
