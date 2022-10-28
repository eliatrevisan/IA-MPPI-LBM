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
python3 test_VGDNN.py --exp_num 86 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;64

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

# 3 MIXTURES, NO DIVERSITY

# DIFFERENT SUBMAP SIZES
: '
python3 train_VGDNN.py --exp_num 102 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 60 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 102 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 103 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 12 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 61 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 103 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 104 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 62 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 104 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 105 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 16 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 63 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 105 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 106 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 18 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 64 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 106 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 107 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 65 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 107 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT COVARIANCE

python3 train_VGDNN.py --exp_num 108 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 8.0;
#python3 compare_VGDNN.py --exp_num 66 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 108 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 109 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 67 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 109 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 110 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 12.0;
#python3 compare_VGDNN.py --exp_num 68 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 110 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 111 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 14.0;
#python3 compare_VGDNN.py --exp_num 69 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 111 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
   
#python3 train_VGDNN.py --exp_num 112 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 16.0;
#python3 compare_VGDNN.py --exp_num 70 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 112 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT HORIZONS

python3 train_VGDNN.py --exp_num 113 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 71 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 113 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 114 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 10 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 72 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 114 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 115 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 12 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 73 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 115 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 116 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 14 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 74 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 116 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 117 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 16 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 75 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 117 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;



# 3 MIXTURES, NO DIVERSITY

# DIFFERENT SUBMAP SIZES

python3 train_VGDNN.py --exp_num 118 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 60 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 118 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 119 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 12 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 61 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 119 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 120 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 62 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 120 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 121 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 16 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 63 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 121 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 122 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 18 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 64 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 122 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 123 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 65 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 123 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT COVARIANCE

python3 train_VGDNN.py --exp_num 124 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 8.0;
#python3 compare_VGDNN.py --exp_num 66 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 124 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 125 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 67 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 125 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 126 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 12.0;
#python3 compare_VGDNN.py --exp_num 68 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 126 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 127 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 14.0;
#python3 compare_VGDNN.py --exp_num 69 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 127 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 128 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 16.0;
#python3 compare_VGDNN.py --exp_num 70 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 128 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# DIFFERENT HORIZONS

python3 train_VGDNN.py --exp_num 129 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 71 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 129 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 130 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 10 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 72 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 130 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 131 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 12 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 73 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 131 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 132 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 14 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 74 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 132 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 133 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 16 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 10 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 75 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 133 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


python3 train_VGDNN.py --exp_num 200 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 200 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 200 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 201 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 201 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 201 --model_name VGDNN --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 202 --model_name VGDNN_diversity --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 202 --model_name VGDNN_diversity --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 202 --model_name VGDNN_diversity --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 203 --model_name VGDNN_diversity --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 203 --model_name VGDNN_diversity --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 203 --model_name VGDNN_diversity --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


#python3 train_VGDNN.py --exp_num 300 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true --submap_span_real 20 --relative_covariance 14.0;
#python3 compare_VGDNN.py --exp_num 300 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 300 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


python3 train_VGDNN.py --exp_num 170 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 170 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 170 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 171 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 171 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 171 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 172 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 20 --dt 0.5 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 172 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 172 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 181 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 16 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 181 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 181 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 182 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 25 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 16 --relative_covariance 14.0;
python3 compare_VGDNN.py --exp_num 182 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 182 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 173 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 173 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 173 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 174 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 11 --dt 0.6 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 174 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 174 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


python3 train_VGDNN.py --exp_num 183 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 14.0;
python3 compare_VGDNN.py --exp_num 183 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 183 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 184 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 14 --prediction_horizon 24 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 14.0;
python3 compare_VGDNN.py --exp_num 184 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 184 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 185 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 14.0;
python3 compare_VGDNN.py --exp_num 185 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 185 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_VGDNN.py --exp_num 186 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario comparison --gpu true --prev_horizon 10 --prediction_horizon 16 --dt 0.6 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 14.0;
python3 compare_VGDNN.py --exp_num 186 --model_name VGDNN --num_test_sequences 10 --scenario comparison --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 186 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

'

#python3 train_crossdata.py --exp_num 200 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 200 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 200 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 compare_VGDNN.py --exp_num 200 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 200 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_crossdata.py --exp_num 201 --model_name VGDNN --total_training_steps 20000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 201 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 201 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 compare_VGDNN.py --exp_num 201 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 201 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_crossdata.py --exp_num 202 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 202 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 202 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 compare_VGDNN.py --exp_num 202 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 202 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_crossdata.py --exp_num 204 --model_name VGDNN --total_training_steps 20000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 204 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 204 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 compare_VGDNN.py --exp_num 204 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 204 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


#python3 train_crossdata.py --exp_num 203 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
#python3 compare_VGDNN.py --exp_num 203 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 203 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 compare_VGDNN.py --exp_num 203 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 203 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 compare_VGDNN.py --exp_num 203 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 203 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
"""
python3 train_crossdata.py --exp_num 207 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 207 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 207 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 207 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 207 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 207 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 207 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 206 --model_name VGDNN --total_training_steps 25000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 206 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 206 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 206 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 206 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 206 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 206 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 205 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 15.0;
python3 compare_VGDNN.py --exp_num 205 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 205 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 205 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 205 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 205 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 205 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 208 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 15.0;
python3 compare_VGDNN.py --exp_num 208 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 208 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 208 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 208 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 208 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 208 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 209 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 209 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 209 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 209 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 209 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 209 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 209 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 210 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 15.0;
python3 compare_VGDNN.py --exp_num 210 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 210 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 210 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 210 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 210 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 210 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 211 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 15.0;
python3 compare_VGDNN.py --exp_num 211 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 211 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 211 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 211 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 211 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 211 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


python3 train_crossdata.py --exp_num 212 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 212 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 212 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 212 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 212 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 212 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 212 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 213 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 15.0;
python3 compare_VGDNN.py --exp_num 213 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 213 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 213 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 213 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 213 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 213 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 214 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 214 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 214 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 214 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 214 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 214 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 214 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 214 --model_name VGDNN --total_training_steps 200000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 214 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 214 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 214 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 214 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 214 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 214 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 215 --model_name VGDNN --total_training_steps 15000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 215 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 215 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 215 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 215 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 215 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 215 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 216 --model_name VGDNN --total_training_steps 20000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 216 --model_name VGDNN --num_test_sequences 10 --scenario herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 216 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 216 --model_name VGDNN --num_test_sequences 10 --scenario prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 216 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 216 --model_name VGDNN --num_test_sequences 10 --scenario bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 216 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 217 --model_name VGDNN --total_training_steps 20000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 217 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 217 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 217 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 217 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 217 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 217 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 218 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 218 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 218 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 218 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 218 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 218 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 218 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 219 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 219 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 219 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 219 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 219 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 219 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 219 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 220 --model_name VGDNN --total_training_steps 200000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 220 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 220 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 220 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 220 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 220 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 220 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


python3 train_crossdata.py --exp_num 221 --model_name VGDNN --total_training_steps 20000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 221 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 221 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 221 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 221 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 221 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 221 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 223 --model_name VGDNN --total_training_steps 20000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 compare_VGDNN.py --exp_num 223 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 223 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 223 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 223 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 223 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 223 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 223 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 223 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 224 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 224 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 224 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 224 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 224 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 224 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 224 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 224 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 224 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# Stopped due to overfitting
python3 train_crossdata.py --exp_num 225 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 225 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 225 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 225 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 225 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 225 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 225 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 225 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 225 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_crossdata.py --exp_num 226 --model_name VGDNN --total_training_steps 20000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 226 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 226 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 226 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 226 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 226 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 226 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 226 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 226 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# With cyclical KL annealing, all data
python3 train_crossdata.py --exp_num 229 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 229 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 229 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 229 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 229 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 229 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 229 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 229 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 229 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# With linear KL annealing (not sure if i actually changed), all data
python3 train_crossdata.py --exp_num 230 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 230 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 230 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 230 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 230 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 230 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 230 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 230 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 230 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# Changed the loss returned in validation_step to total_loss instead of reconstruction_loss, all data
python3 train_crossdata.py --exp_num 231 --model_name VGDNN --total_training_steps 20000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 231 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 231 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 231 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 231 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 231 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 231 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 231 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 231 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# Cyclical KL annealing, all data expect open_crossing
python3 train_crossdata.py --exp_num 232 --model_name VGDNN --total_training_steps 20000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 232 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 232 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 232 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 232 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 232 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 232 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 232 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 232 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# Cyclical KL annealing, all data exept the herengracht
python3 train_crossdata.py --exp_num 233 --model_name VGDNN --total_training_steps 20000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 233 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 233 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 233 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 233 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 233 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 233 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_roboat.py --exp_num 235 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 235 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 235 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 235 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 235 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 235 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 235 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_roboat.py --exp_num 236 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 236 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 236 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 236 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 236 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 236 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 236 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_roboat.py --exp_num 237 --model_name VGDNN --total_training_steps 200000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 237 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 237 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 237 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 237 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 237 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 237 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# Cycl.KL, epoch diff
python3 train_roboat.py --exp_num 238 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 238 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 238 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 238 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 238 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 238 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 238 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 238 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 238 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# Cycl.KL, epoch diff
python3 train_roboat.py --exp_num 239 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 239 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 239 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 239 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 239 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 239 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 239 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 239 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 239 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


# Same as 238 but differetn log output (idx instead of agent id)
python3 train_roboat.py --exp_num 240 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 240 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 240 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 240 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 240 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 240 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 240 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 240 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 240 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# Same as 240 but again, differetn log output (idx instead of agent id)
python3 train_roboat.py --exp_num 241 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 241 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 241 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 241 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 241 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 241 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 241 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 241 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 241 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# 241 was useless for log output, with this one should work
python3 train_roboat.py --exp_num 242 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 242 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 242 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 242 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 242 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 242 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 242 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 242 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 242 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# Testing with combined trajectory_set 
python3 train_roboat.py --exp_num 243 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 243 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 243 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 243 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 243 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 243 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 243 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 243 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 243 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


# Testing with combined trajectory_set and log
python3 train_roboat.py --exp_num 244 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 244 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 244 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 244 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 244 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 244 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 244 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 244 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 244 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


# Testing with combined trajectory_set and log
python3 train_roboat.py --exp_num 245 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 8.0;
python3 test_VGDNN.py --exp_num 245 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 245 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 245 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 245 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 245 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 245 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 245 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 245 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# Testing with combined trajectory_set and log
python3 train_roboat.py --exp_num 246 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
python3 test_VGDNN.py --exp_num 246 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 246 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 246 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 246 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 246 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 246 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 246 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 246 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
"""

# # Testing with combined trajectory_set and log
# python3 train_roboat.py --exp_num 247 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --regularization_weight 0.01 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
# python3 test_VGDNN.py --exp_num 247 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 247 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 247 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 247 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 247 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 247 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 247 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 247 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# # Testing with combined trajectory_set and log
# python3 train_roboat.py --exp_num 248 --model_name VGDNN --total_training_steps 100000 --n_mixtures 3 --regularization_weight 0.001 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
# python3 test_VGDNN.py --exp_num 248 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 248 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 248 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 248 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 248 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 248 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 248 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 248 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 253 --model_name VGDNN --total_training_steps 30000 --n_mixtures 3 --output_pred_state_dim 4 --scenario bloemgracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
#python3 train_VGDNN.py --exp_num 254 --model_name VGDNN --total_training_steps 30000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
#python3 train_VGDNN.py --exp_num 255 --model_name VGDNN --total_training_steps 30000 --n_mixtures 3 --output_pred_state_dim 4 --scenario prinsengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
#python3 train_VGDNN.py --exp_num 256 --model_name VGDNN --total_training_steps 30000 --n_mixtures 3 --output_pred_state_dim 4 --scenario open_crossing --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;

#python3 train_VGDNN.py --exp_num 257 --model_name VGDNN --total_training_steps 30000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;

#python3 train_roboat.py --exp_num 260 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
#python3 test_VGDNN.py --exp_num 260 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 260 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 260 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 260 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 260 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 260 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 260 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 260 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 260 --model_name VGDNN --num_test_sequences 10 --scenario test_amstel --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 260 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_amstel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


#python3 test_VGDNN.py --exp_num 270 --model_name VGDNN --num_test_sequences 10 --scenario test_amstel --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 270 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_amstel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 test_VGDNN.py --exp_num 271 --model_name VGDNN --num_test_sequences 10 --scenario test_amstel --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 271 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_amstel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 test_VGDNN.py --exp_num 272 --model_name VGDNN --num_test_sequences 10 --scenario test_amstel --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 272 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_amstel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
##

# feeddicttrain
# python3 train_roboat.py --exp_num 274 --model_name VGDNN --total_training_steps 50000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
# python3 test_VGDNN.py --exp_num 274 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 274 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 274 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 274 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 274 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 274 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 274 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 274 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 274 --model_name VGDNN --num_test_sequences 10 --scenario test_amstel --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 274 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_amstel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


#python3 train_roboat.py --exp_num 276 --model_name VGDNN --total_training_steps 25000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 10.0;
#python3 test_VGDNN.py --exp_num 276 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 276 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 276 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 276 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 276 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 276 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 276 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 276 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 276 --model_name VGDNN --num_test_sequences 10 --scenario test_amstel --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 276 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_amstel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

# python3 train_roboat.py --exp_num 278 --model_name VGDNN --total_training_steps 40000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 10.0;
# python3 test_VGDNN.py --exp_num 278 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 278 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 278 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 278 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 278 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 278 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 278 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 278 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 278 --model_name VGDNN --num_test_sequences 10 --scenario test_amstel --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
# python3 test_VGDNN.py --exp_num 278 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_amstel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_roboat.py --exp_num 282 --model_name VGDNN --total_training_steps 40000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 14 --relative_covariance 15.0;
python3 test_VGDNN.py --exp_num 282 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 282 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 282 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 282 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 282 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 282 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 282 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 282 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 282 --model_name VGDNN --num_test_sequences 10 --scenario test_amstel --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 282 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_amstel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 train_roboat.py --exp_num 283 --model_name VGDNN --total_training_steps 40000 --n_mixtures 3 --output_pred_state_dim 4 --scenario herengracht --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.8 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update false --submap_span_real 20 --relative_covariance 15.0;
python3 test_VGDNN.py --exp_num 283 --model_name VGDNN --num_test_sequences 10 --scenario test_herengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 283 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_herengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 283 --model_name VGDNN --num_test_sequences 10 --scenario test_prinsengracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 283 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_prinsengracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 283 --model_name VGDNN --num_test_sequences 10 --scenario test_bloemgracht --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 283 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_bloemgracht --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 283 --model_name VGDNN --num_test_sequences 10 --scenario test_open_crossing --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 283 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_open_crossing --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 283 --model_name VGDNN --num_test_sequences 10 --scenario test_amstel --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 283 --model_name VGDNN --constant_velocity true --num_test_sequences 100 --scenario test_amstel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
