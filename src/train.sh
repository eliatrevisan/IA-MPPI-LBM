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

END

python3 train_VGDNN.py --exp_num 24 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
python3 test_VGDNN.py --exp_num 24 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 test_VGDNN.py --exp_num 24 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
