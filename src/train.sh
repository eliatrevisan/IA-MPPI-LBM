: <<'END'

#python3 train_VGDNN.py --exp_num 1 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 train_VGDNN.py --exp_num 2 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;

#python3 train_VGDNN.py --exp_num 3 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;

python3 train_VGDNN.py --exp_num 1 --model_name VGDNN --n_mixtures 1 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 1 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 1 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 2 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 2 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 2 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#####

python3 train_VGDNN.py --exp_num 3 --model_name VGDNN --n_mixtures 1 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 3 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 3 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 4 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 4 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 4 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#####

python3 train_VGDNN.py --exp_num 5 --model_name VGDNN --n_mixtures 1 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 12 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 5 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 5 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 6 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 12 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 6 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 6 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#####

python3 train_VGDNN.py --exp_num 7 --model_name VGDNN --n_mixtures 1 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 12 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 7 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 7 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 8 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 12 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 8 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 8 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#####

python3 train_VGDNN.py --exp_num 9 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 9 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 9 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


python3 train_VGDNN.py --exp_num 10 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 12 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 10 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 10 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 11 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 11 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 11 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 12 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 12 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 12 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 13 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 13 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 13 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 14 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 12 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 14 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 14 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 15 --model_name VGDNN --n_mixtures 1 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 15 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 15 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 17 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true --total_training_steps 500000;
python3 test_VGDNN.py --exp_num 17 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 17 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 test_VGDNN.py --exp_num 17 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;


#python3 train_VGDNN.py --exp_num 18 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 18 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 18 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 19 --model_name VGDNN --n_mixtures 1 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 19 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 19 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 20 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 12 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 20 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 20 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 21 --model_name VGDNN --n_mixtures 1 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 12 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 21 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 21 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 22 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 22 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 22 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;



#python3 train_VGDNN.py --exp_num 24 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 24 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 24 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 25 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 25 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 25 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 26 --model_name VGDNN --n_mixtures 3 --kl_weight 0.05 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 26 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 26 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 27 --model_name VGDNN --n_mixtures 3 --kl_weight 0.01 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 27 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 27 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 28 --model_name VGDNN --n_mixtures 3 --kl_weight 0.001 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 28 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 28 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 29 --model_name VGDNN --n_mixtures 3 --kl_weight 0.005 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 29 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 29 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


python3 train_VGDNN.py --exp_num 30 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 30 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 30 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 31 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 31 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 31 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


#python3 train_VGDNN.py --exp_num 32 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 32 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 32 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


#python3 train_VGDNN.py --exp_num 33 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 33 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 33 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


#python3 train_VGDNN.py --exp_num 34 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 34 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 34 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


#python3 train_VGDNN.py --exp_num 35 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 35 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 35 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


#python3 train_VGDNN.py --exp_num 37 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 4 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 37 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 37 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 38 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 38 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 38 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 39 --model_name VGDNN --n_mixtures 1 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 39 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 39 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 40 --model_name VGDNN --n_mixtures 1 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 40 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 40 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 41 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 41 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 41 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 42 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 42 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 42 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 43 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 43 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 43 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 44 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu false --prev_horizon 8 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 44 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 44 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 45 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 10 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 45 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 45 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 46 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 4 --prediction_horizon 10 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 46 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 46 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 47 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 10 --dt 1.0 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 47 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 47 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 48 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 4 --prediction_horizon 10 --dt 1.0 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 48 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 48 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


#python3 train_VGDNN.py --exp_num 49 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 49 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 49 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 50 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 10 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 50 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 50 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 51 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 51 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 51 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 52 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 10 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 52 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 52 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 53 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 1.0 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 53 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 53 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 54 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 54 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 54 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 55 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 55 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 55 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 56 --model_name VGDNN_diversity --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 56 --model_name VGDNN_diversity --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 56 --model_name VGDNN_diversity --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 60 --model_name VGDNN_diversity --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 compare_VGDNN.py --exp_num 1 --model_name VGDNN_diversity --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 1 --model_name VGDNN_diversity --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#END


# 3 MIXTURES, NO DIVERSITY

# DIFFERENT SUBMAP SIZES

python3 train_VGDNN.py --exp_num 60 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 60 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 60 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 61 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 12 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 61 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 61 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 62 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 62 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 62 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 63 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 16 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 63 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 63 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 64 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 18 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 64 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 64 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 65 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 65 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 65 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT COVARIANCE

python3 train_VGDNN.py --exp_num 66 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 8.0;
python3 compare_VGDNN.py --exp_num 66 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 66 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 67 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 67 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 67 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 68 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 12.0;
python3 compare_VGDNN.py --exp_num 68 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 68 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 69 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 14.0;
python3 compare_VGDNN.py --exp_num 69 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 69 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 70 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 16.0;
python3 compare_VGDNN.py --exp_num 70 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 70 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT HORIZONS

python3 train_VGDNN.py --exp_num 71 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 71 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 71 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 72 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 10 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 72 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 72 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 73 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 12 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 73 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 73 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 74 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 14 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 74 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 74 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 75 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 16 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 75 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 75 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT TIME STEPS FOR FURTHER PREDICTIONS

python3 train_VGDNN.py --exp_num 76 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 10 --prediction_horizon 20 --dt 1.0 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 76 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 76 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 77 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 10 --prediction_horizon 10 --dt 0.5 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 77 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 77 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# 1 MIXTURE, NO DIVERSITY

# DIFFERENT SUBMAP SIZES

python3 train_VGDNN.py --exp_num 80 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 80 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 80 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 81 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 12 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 81 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 81 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 82 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 82 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 82 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 83 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 16 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 83 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 83 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 84 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 18 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 84 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 84 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 85 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 85 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 85 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT COVARIANCE

python3 train_VGDNN.py --exp_num 86 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 8.0;
python3 compare_VGDNN.py --exp_num 86 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 86 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 87 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 87 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 87 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 88 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 12.0;
python3 compare_VGDNN.py --exp_num 88 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 88 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 89 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 14.0;
python3 compare_VGDNN.py --exp_num 89 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 89 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 90 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 16.0;
python3 compare_VGDNN.py --exp_num 90 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 90 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT HORIZONS

python3 train_VGDNN.py --exp_num 91 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 91 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 91 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 92 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 10 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 92 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 92 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 93 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 12 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 93 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 93 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 94 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 14 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 94 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 94 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 95 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 16 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 95 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 95 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT TIME STEPS FOR FURTHER PREDICTIONS

python3 train_VGDNN.py --exp_num 96 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 10 --prediction_horizon 20 --dt 1.0 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 96 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 96 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 97 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 10 --prediction_horizon 10 --dt 0.5 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 97 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 97 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# 3 MIXTURES, WITH DIVERSITY

python3 train_VGDNN.py --exp_num 100 --model_name VGDNN_diversity --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true --submap_span_real 10 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 100 --model_name VGDNN_diversity --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 100 --model_name VGDNN_diversity --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

END

python3 train_VGDNN.py --exp_num 101 --model_name VGDNN --total_training_steps 50000 --n_mixtures 1 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 12.0;
python3 compare_VGDNN.py --exp_num 101 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 101 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 102 --model_name RNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 12.0;

#python3 train_VGDNN.py --exp_num 1000 --model_name VGDNN_diversity --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 1000 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 1000 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 test_VGDNN.py --exp_num 93 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 94 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 95 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 97 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
