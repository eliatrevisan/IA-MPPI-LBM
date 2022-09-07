#python3 test_VGDNN.py --exp_num 100 --model_name VRNN --num_test_sequences 100 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 101 --model_name VRNN --num_test_sequences 100 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 102 --model_name VRNN --num_test_sequences 100 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 103 --model_name VRNN --num_test_sequences 100 --scenario real_world/zara_01 --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 test_VGDNN.py --exp_num 104 --model_name VRNN --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 105 --model_name VRNN --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 106 --model_name VRNN --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 107 --model_name VRNN --num_test_sequences 100 --scenario real_world/st --record false --n_samples 1 --unit_testing false --freeze_other_agents false;


#python3 test_VGDNN.py --exp_num 122 --model_name VRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_hotel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 123 --model_name VRNN --num_test_sequences 100 --scenario real_world/ewap_dataset/seq_hotel --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 test_VGDNN.py --exp_num 3 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 3 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 test_VGDNN.py --exp_num 3 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 1 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 test_VGDNN.py --exp_num 8 --model_name VGDNN --num_test_sequences 10 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 8 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 test_VGDNN.py --exp_num 13 --model_name VGDNN --num_test_sequences 4 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;
#python3 test_VGDNN.py --exp_num 8 --model_name VGDNN --num_test_sequences 100 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 train_VGDNN.py --exp_num 100 --model_name VGDNN --n_mixtures 3 --output_pred_state_dim 4 --scenario simulation/roboat --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 8 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --diversity_update true;
#python3 test_VGDNN.py --exp_num 30 --model_name VGDNN --constant_velocity true --num_test_sequences 12 --scenario simulation/roboat --record false --n_samples 1 --unit_testing false --freeze_other_agents false;

#python3 compare_VGDNN.py --exp_num 33 --model_name VGDNN --constant_velocity false --num_test_sequences 1 --scenario simulation/roboat --record true --n_samples 1 --unit_testing false --freeze_other_agents false;

python3 compare_VGDNN.py --exp_num 43 --model_name VGDNN --constant_velocity false --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
python3 compare_VGDNN.py --exp_num 44 --model_name VGDNN --constant_velocity false --num_test_sequences 100 --scenario comparison --record false --n_samples 1 --unit_testing false --freeze_other_agents false;
