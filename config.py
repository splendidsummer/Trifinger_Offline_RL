# PROJECT_NAME = "Trifinger_RL_Offline_SE"
FEAT_PROJECT_NAME = "Trifinger_Features"
AUG_PROJECT_NAME = 'Trifinger_Data_Augmentation2'
PROJECT_NAME = 'Trifinger_Training'
SWEEP_PROJECT_NAME = 'Trifinger_Lift_Sweep'

# SEEDS = [3047, 168, 186, 200, 250, 4, 9, 25, 11, 16]
NUM_EVAL = 10
# RESULT_DF_COLS = ['n_episodes',  'success_rate', 'mean_momentary_success',
#              'transient_success_rate', 'return', 'max_reward', 'dataset']

RESULT_DF_COLS = ['n_episodes', 'return', 'max_reward', 'dataset']

push_features = {'object_observations': [
                    'curr_position', 'previous_actions', 'desired_position', 'confidence', 'delay',
                    'keypoints', 'curr_orientation', 'dupli_curr_position'],

                 'robot_observations': [
                    'finger_tip_force', 'finger_tip_position', 'finger_tip_velocity',
                    'robot_position', 'robot_id',  'robot_torque', 'robot_velocity']
                }

push_fea1 = {'object_observations': [
                'curr_position', 'previous_actions', 'desired_position',
                'keypoints', 'curr_orientation', 'dupli_curr_position'],

            'robot_observations': [
                'finger_tip_force', 'finger_tip_position', 'finger_tip_velocity',
                'robot_position',  'robot_torque', 'robot_velocity']
            }

push_fea2 = {'object_observations': [
                'curr_position', 'previous_actions', 'desired_position',
                'keypoints', 'curr_orientation'],

            'robot_observations': [
                'finger_tip_force', 'finger_tip_position', 'finger_tip_velocity',
                'robot_position',  'robot_torque', 'robot_velocity']
            }

push_fea3 = {'object_observations': [
                'curr_position', 'previous_actions', 'desired_position', 'curr_orientation'],

            'robot_observations': [
                'finger_tip_force', 'finger_tip_position', 'finger_tip_velocity',
                'robot_position',  'robot_torque', 'robot_velocity']
            }

push_features_indices = {'object_observations': [
                    [0, 3], [3, 12], [12, 15], [15, 16], [16, 17],
                    [17, 41], [41, 45], [45, 48]],
                 'robot_observations': [
                     [48, 51], [51, 60], [60, 69],
                     [69, 78], [78, 79], [79, 88], [88, 97]]
                }

lift_features = {'object_observations': [
                    'curr_keypoints', 'previous_actions', 'desired_keypoints',
                    'confidence', 'delay', 'dupli_curr_keypoints', 'curr_orientation', 'curr_position'],
                 'robot_observations': [
                     'finger_tip_force', 'finger_tip_position', 'finger_tip_velocity',
                     'robot_position', 'robot_id', 'robot_torque', 'robot_velocity', ]
}

lift_fea1 = {'object_observations': [
                    'curr_keypoints', 'previous_actions', 'desired_keypoints',
                     'dupli_curr_keypoints', 'curr_orientation', 'curr_position'],
                 'robot_observations': [
                     'finger_tip_force', 'finger_tip_position', 'finger_tip_velocity',
                     'robot_position', 'robot_torque', 'robot_velocity', ]
}

lift_fea2 = {'object_observations': [
                    'curr_keypoints', 'previous_actions', 'desired_keypoints',
                    'curr_orientation', 'curr_position'],
                 'robot_observations': [
                     'finger_tip_force', 'finger_tip_position', 'finger_tip_velocity',
                     'robot_position', 'robot_torque', 'robot_velocity', ]
}

lift_fea3 = {'object_observations': [
                    'curr_keypoints', 'previous_actions', 'desired_keypoints'],
                 'robot_observations': [
                     'finger_tip_force', 'finger_tip_position', 'finger_tip_velocity',
                     'robot_position', 'robot_torque', 'robot_velocity', ]
}

lift_features_indices = {'object_observations': [
                            [0, 24], [24, 33], [33, 57],
                            [57, 58], [58, 59], [59, 83], [83, 87], [87, 90]],
                         'robot_observations': [
                             [90, 93], [93, 102], [102, 111],
                             [111, 120], [120, 121], [121, 130], [130, 139]]
}

push_feat_dict = {'dataset/push_100_raw.npy': 'raw', 'dataset/push_feat1.pkl': 'feat1',
                   'dataset/push_feat2.pkl': 'feat2', 'dataset/push_feat3.pkl': 'feat3'}
lift_feat_dict = {'dataset/lift_100_raw.npy': 'raw', 'dataset/lift_feat1.pkl': 'feat1',
                   'dataset/lift_feat2.pkl': 'feat2', 'dataset/lift_feat3.pkl': 'feat3'}


## configuration for sweep
sweep_config = {'method': 'random'}
metric = {'name': 'return', 'goal': 'maximize'}
sweep_config['metric'] = metric
lr = {'value': 0.001}
# batch_size = {"distribution": 'q_uniform', 'q': 8, 'min': 64, 'max': 256}
scaler = {'values': ['standard', 'min_max', None]}
action_scaler = {'values': ['min_max', None]}
sweep_config['parameters'] = {}
sweep_config['parameters'].update({
    'lr': lr,
    # 'batch_size': batch_size,
    'scaler': scaler, 'action_scaler': action_scaler})

