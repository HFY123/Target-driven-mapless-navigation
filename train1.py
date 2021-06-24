import os
import random
import shutil
import time
import math
import scipy

import numpy as np
import tensorflow.compat.v1 as tf
import csv

from vae_common import create_encode_state_fn, load_vae
from ppo import PPO
from reward_functions import reward_functions
from run_eval import run_eval
from utils import compute_gae
from vae.models import ConvVAE, MlpVAE
import carla
import pygame



USE_ROUTE_ENVIRONMENT = True

if USE_ROUTE_ENVIRONMENT:
    from CarlaEnv.carla_route_env import CarlaRouteEnv as CarlaEnv
else:
    from CarlaEnv.carla_lap_env import CarlaLapEnv as CarlaEnv

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)


def train(params, start_carla=True, restart=False):
    # Read parameters
    learning_rate    = params["learning_rate"]
    lr_decay         = params["lr_decay"]
    discount_factor  = params["discount_factor"]
    gae_lambda       = params["gae_lambda"]
    ppo_epsilon      = params["ppo_epsilon"]
    initial_std      = params["initial_std"]
    value_scale      = params["value_scale"]
    entropy_scale    = params["entropy_scale"]
    horizon          = params["horizon"]
    num_epochs       = params["num_epochs"]
    num_episodes     = params["num_episodes"]
    batch_size       = params["batch_size"]
    vae_model        = params["vae_model"]
    vae_model_type   = params["vae_model_type"]
    vae_z_dim        = params["vae_z_dim"]
    synchronous      = params["synchronous"]
    fps              = params["fps"]
    action_smoothing = params["action_smoothing"]
    model_name       = params["model_name"]
    reward_fn        = params["reward_fn"]
    seed             = params["seed"]
    eval_interval    = params["eval_interval"]
    record_eval      = params["record_eval"]
    percentage       = params["percentage"]

    with open(f'oneturn1_record\data_{percentage}_' + model_name[-1] + '.csv', "a+", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['step', 'reward', 'distance'])

    # Set seeds
    if isinstance(seed, int):
        tf.random.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(0)

    # Load VAE
    vae = load_vae(vae_model, vae_z_dim, vae_model_type)
    
    # Override params for logging
    params["vae_z_dim"] = vae.z_dim
    params["vae_model_type"] = "mlp" if isinstance(vae, MlpVAE) else "cnn"

    print("")
    print("Training parameters:")
    for k, v, in params.items(): print(f"  {k}: {v}")
    print("")

    # Create state encoding fn
    #measurements_to_include = set(["steer", "throttle", "speed"])
    measurements_to_include = set(["steer", "throttle", "speed", "distance", "angle"])  #--hfy
    encode_state_fn = create_encode_state_fn(vae, measurements_to_include)

    # Create env
    cur_len = 50
    print("Creating environment")
    env = CarlaEnv(obs_res=(160, 80),
                   action_smoothing=action_smoothing,
                   encode_state_fn=encode_state_fn,
                   reward_fn1=reward_functions["reward_speed_centering_angle_add"],
                   reward_fn2=reward_functions["reward_speed_centering_angle_add_eval"],
                   synchronous=synchronous,
                   fps=fps,
                   start_carla=start_carla)
    if isinstance(seed, int):
        env.seed(seed)
    best_eval_reward = -float("inf")

    # Environment constants
    input_shape = np.array([vae.z_dim + len(measurements_to_include)])  #vae.z_dim = 64  --hfy
    num_actions = env.action_space.shape[0]

    # Create model
    print("Creating model")
    model = PPO(input_shape, env.action_space,
                learning_rate=learning_rate, lr_decay=lr_decay,
                epsilon=ppo_epsilon, initial_std=initial_std,
                value_scale=value_scale, entropy_scale=entropy_scale,
                model_dir=os.path.join("models", model_name))

    # Prompt to load existing model if any
    if not restart:
        if os.path.isdir(model.log_dir) and len(os.listdir(model.log_dir)) > 0:
            answer = input("Model \"{}\" already exists. Do you wish to continue (C) or restart training (R)? ".format(model_name))
            if answer.upper() == "C":
                pass
            elif answer.upper() == "R":
                restart = True
            else:
                raise Exception("There are already log files for model \"{}\". Please delete it or change model_name and try again".format(model_name))
    
    if restart:
        shutil.rmtree(model.model_dir)
        for d in model.dirs:
            os.makedirs(d)
    model.init_session()
    if not restart:
        model.load_latest_checkpoint()
    model.write_dict_to_summary("hyperparameters", params, 0)

    
    
    if os.access("D:/Edge Download/QQ/283967637/FileRecv/Carla_ppo_master/human/a.txt", os.F_OK):
        a = np.loadtxt("D:/Edge Download/QQ/283967637/FileRecv/Carla_ppo_master/human/a.txt")
        s = np.loadtxt("D:/Edge Download/QQ/283967637/FileRecv/Carla_ppo_master/human/s.txt")
        d = np.loadtxt("D:/Edge Download/QQ/283967637/FileRecv/Carla_ppo_master/human/d.txt")
        r = np.loadtxt("D:/Edge Download/QQ/283967637/FileRecv/Carla_ppo_master/human/r.txt")
        v_ = np.loadtxt("D:/Edge Download/QQ/283967637/FileRecv/Carla_ppo_master/human/v.txt")
        human_len = len(a)
        print('human experience loaded')

        _, last_v = model.predict(s[-1])
        advantages1 = compute_gae(r, v_, last_v, d, discount_factor, gae_lambda)
        returns1 = advantages1 + v_
        advantages1 = (advantages1 - advantages1.mean()) / (advantages1.std() + 1e-8)

        s = s[10:]
        a = a[10:]
        d = d[10:]
        r = r[10:]
        len_a = len(a)
    else:
        human_control = False
        control = input("Would you like to play the route(y/n)?")
        s, a, r, d, v_ = [], [], [], [], []
        if control.upper() == "Y":
            human_control = True
            action = np.zeros(env.action_space.shape[0])
            
            obs = env.reset()
            while True:
                temp = [0.0]*2
                # Process key inputs
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[K_LEFT] or keys[K_a]:
                    action[0] = -0.4
                elif keys[K_RIGHT] or keys[K_d]:
                    action[0] = 0.4
                else:
                    action[0] = 0.0
                action[0] = np.clip(action[0], -1, 1)
                action[1] = 0.35 if keys[K_UP] or keys[K_w] else 0.0
                temp[0], temp[1] = action[0], action[1]
                a.append(temp) 
                
                # Take action
                _, v = model.predict(obs, greedy=False, write_to_summary=False)
                new_obs, reward, done, info = env.step(action, is_training=True)
                obs = new_obs

                s.append(obs)
                
                r.append(reward)
                d.append(done) 
                v_.append(v)

                if info["closed"]: # Check if closed
                    exit(0)
                env.render() # Render
                
                if done: break
            _, last_v = model.predict(obs)
            advantages1 = compute_gae(r, v_, last_v, d, discount_factor, gae_lambda)
            returns1 = advantages1 + v_
            advantages1 = (advantages1 - advantages1.mean()) / (advantages1.std() + 1e-8)
            
        elif control.upper() == "N":
            pass
        else:
            raise Exception("Please enter Y/y or N/n")

        if human_control:
            human_len = len(a)
            print("Play finished")
            time.sleep(1)

        s = s[10:]
        a = a[10:]
        d = d[10:]
        r = r[10:]
        v_ = v_[10:]

        len_a = len(a)
        
        np.savetxt('human/s.txt',s)
        np.savetxt('human/a.txt',a)
        np.savetxt('human/r.txt',r)
        np.savetxt('human/d.txt',d)
        np.savetxt('human/v.txt',v_)

    
    '''
    with open("file.txt", "w") as output:
        output.write(a)
    '''
    '''
    human_s = np.var(s)
    human_a = np.var(a)
    human_r = np.var(r)
    human_d = np.var(d)
    '''
    pre_len = 0
    count = 1
    # For every episode
    while num_episodes <= 0 or model.get_episode_idx() < num_episodes:
        episode_idx = model.get_episode_idx()
        
        # Run evaluation periodically
        
        if episode_idx % eval_interval == 0:
            video_filename = os.path.join(model.video_dir, "episode{}.avi".format(episode_idx))
            eval_reward = run_eval(env, model, video_filename=None)
            model.write_value_to_summary("eval/reward", eval_reward, episode_idx)
            model.write_value_to_summary("eval/distance_traveled", env.distance_traveled, episode_idx)
            model.write_value_to_summary("eval/average_speed", 3.6 * env.speed_accum / env.step_count, episode_idx)
            model.write_value_to_summary("eval/center_lane_deviation", env.center_lane_deviation, episode_idx)
            model.write_value_to_summary("eval/average_center_lane_deviation", env.center_lane_deviation / env.step_count, episode_idx)
            model.write_value_to_summary("eval/distance_over_deviation", env.distance_traveled / env.center_lane_deviation, episode_idx)
            if eval_reward > best_eval_reward:
                model.save()
                best_eval_reward = eval_reward
        

        # Reset environment
        state, terminal_state, total_reward = env.reset(), False, 0
        #time.sleep(0.5)
        
        # While episode not done
        #print("Episode {episode_idx} (Step {model.get_train_step_idx()})")
        while not terminal_state:
            states, taken_actions, values, rewards, dones = [], [], [], [], []
            for _ in range(horizon):
                action, value = model.predict(state, greedy=False, write_to_summary=True)

                # Perform action
                new_state, reward, terminal_state, info = env.step(action, is_training=True)
                #print(new_state)

                if info["closed"] == True:
                    exit(0)
                    
                env.extra_info.extend([
                    "Episode {}".format(episode_idx),
                    "Training...",
                    "",
                    "Value:  % 20.2f" % value
                ])

                env.render()
                total_reward += reward
                

                # Store state, action and reward
                states.append(state)         # [T, *input_shape]
                taken_actions.append(action) # [T,  num_actions]
                values.append(value)         # [T]
                rewards.append(reward)       # [T]
                dones.append(terminal_state) # [T]
                state = new_state

                if terminal_state:
                    break
                
            np.savetxt('rate/success_rate.txt', env.success_rate)
            np.savetxt('rate/collision_rate.txt', env.collision_rate)
            np.savetxt('rate/stop_rate.txt', env.stop_rate)
            np.savetxt('rate/off_track_rate.txt', env.off_track_rate)
            np.savetxt('rate/speeding_rate.txt', env.speeding_rate)

            # np.savetxt(f1, env.success_rate)
            # np.savetxt(f2, env.collision_rate)
            # np.savetxt(f3, env.stop_rate)
            # np.savetxt(f4, env.off_track_rate)
            # np.savetxt(f5, env.speeding_rate)

            # Calculate last value (bootstrap value)       
            # Compute GAE
            
            taken_actions_ = np.array(taken_actions)
            states2 = np.array(states)


            '''
            if cur_len <= len(s):
                s = s[:cur_len]
                a = a[:cur_len]
                returns1 = returns1[:cur_len]
                advantages1 = advantages1[:cur_len]
                print(episode_idx)
                if episode_idx != 0 and episode_idx % 10 == 0:
                    cur_len += 50
            '''
            
            length = pre_len + int(human_len*env.routes_completed) + 1
            # if length > len(a):
            #     length = len(a)
            #length = 30
            # print("[%d,%d]" %(pre_len,length))
            temp_s = random.sample(list(s),50)
            # temp_s = s[:length]
            
            
            
            temp_a = random.sample(list(a),50)
            # temp_a = a[:length]
            
            
            temp_returns1 = random.sample(list(returns1),50)
            # temp_returns1 = returns1[:length]
            

            
            temp_advantages1 = random.sample(list(advantages1),50)
            # temp_advantages1 = advantages1[:length]
            
            len_mat = []
            len_mat.append(len(temp_s))
            len_mat.append(len(temp_a))
            len_mat.append(len(temp_returns1))
            len_mat.append(len(temp_advantages1))
            print(len_mat)
            # pre_len = length
            # if pre_len >= horizon:
            #     pre_len = horizon*count
            #     count += 1
            # else:
            #     pre_len = 0
            #     count = 1
            
            
            _1, last_values = model.predict(state) # []  

            # Compute GAE
            advantages2 = compute_gae(rewards, values, last_values, dones, discount_factor, gae_lambda)
            returns2 = advantages2 + values
            advantages_2 = (advantages2 - advantages2.mean()) / (advantages2.std() + 1e-8)

            returns2_ = np.array(returns2)
            advantages2_ = np.array(advantages_2)

            max_exp = int(float(percentage) * len_a)
            
            if min(len_mat) >= max_exp:
                print(min(len_mat), max_exp)         
                states_1 = random.sample(list(temp_s), max_exp)
                action_1 = random.sample(list(temp_a), max_exp)
                returns_1 = random.sample(list(temp_returns1), max_exp)
                advantages_1 = random.sample(list(temp_advantages1), max_exp)

                _states_1 = np.array(states_1)
                _action_1 = np.array(action_1)
                _returns_1 = np.array(returns_1)
                _advantages_1 = np.array(advantages_1)
            else:
                _states_1 = np.array(temp_s)
                _action_1 = np.array(temp_a)
                _returns_1 = np.array(temp_returns1)
                _advantages_1 = np.array(temp_advantages1)
            # if env.routes_completed < 0.9:
            #     s = s + 
            # states1 = np.array(s)
            # a = np.array(a)
            # returns1 = np.array(returns1)
            # advantages1 = np.array(advantages1)

            returns_hfy = np.hstack((_returns_1, returns2_))
            advantages_hfy = np.hstack((_advantages_1, advantages2_))
            states_hfy = np.vstack((_states_1, states2))
            actions_hfy = np.vstack((_action_1, taken_actions_))

            
            # Flatten arrays
            states_final        = np.array(states_hfy)
            taken_actions_final = np.array(actions_hfy)
            returns_final       = np.array(returns_hfy)
            advantages_final    = np.array(advantages_hfy)

            if env.distance_traveled > 0.0:
            # Train for some number of epochs
                model.update_old_policy() # θ_old <- θ
                for _ in range(num_epochs):
                    num_samples = len(taken_actions_final)
                    indices = np.arange(num_samples)
                    np.random.shuffle(indices)
                    for i in range(int(np.ceil(num_samples / batch_size))):
                        # Sample mini-batch randomly
                        begin = i * batch_size
                        end   = begin + batch_size
                        if end > num_samples:
                            end = None
                        mb_idx = indices[begin:end]

                        # Optimize network
                        #print(mb_idx)
                        model.train(states_final[mb_idx], taken_actions_final[mb_idx],
                                    returns_final[mb_idx], advantages_final[mb_idx])
                


        #print(env.collision_num)
        env.success_rate.append(env.success_num / (episode_idx + 1))
        env.collision_rate.append(env.collision_num / (episode_idx + 1))
        env.stop_rate.append(env.stop_num / (episode_idx + 1))
        env.off_track_rate.append(env.off_track_num / (episode_idx + 1))
        env.speeding_rate.append(env.speeding_num / (episode_idx + 1))
        

        
        if total_reward > best_eval_reward:
                model.save()
                best_eval_reward = total_reward
        

        # Write episodic values
        model.write_value_to_summary("train/reward", total_reward, episode_idx)
        model.write_value_to_summary("train/distance_traveled", env.distance_traveled, episode_idx)
        model.write_value_to_summary("train/average_speed", 3.6 * env.speed_accum / env.step_count, episode_idx)
        model.write_value_to_summary("train/center_lane_deviation", env.center_lane_deviation, episode_idx)
        model.write_value_to_summary("train/average_center_lane_deviation", env.center_lane_deviation / env.step_count, episode_idx)
        model.write_value_to_summary("train/distance_over_deviation", env.distance_traveled / env.center_lane_deviation, episode_idx)
        model.write_episodic_summaries()

        with open(f'oneturn1_record\data_{percentage}_' + model_name[-1] + '.csv', "a+", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([episode_idx, total_reward, env.distance_traveled])

        



if __name__ == "__main__":
    import argparse

    default = 0
    while(default==0):
        try:
            client = carla.Client("localhost", 2000) #connect to the localhost
            client.set_timeout(5.0) 
            if client:
                world = client.load_world('Town07')
                if world:
                    default = 1
                    print('default value changed,build success')
                else:
                    print('failed to build world')
            else:
                print('failed to connect')
        except:
            continue

    weather = carla.WeatherParameters(
                    cloudiness=0.0,
                    precipitation=0.0,
                    sun_altitude_angle=50.0)
    world.set_weather(weather) #set weather as a sunny day
    
    settings = world.get_settings()
    settings.synchronous_mode = True # Enables synchronous mode
    world.apply_settings(settings)
    
    parser = argparse.ArgumentParser(description="Trains a CARLA agent with PPO")

    # PPO hyper parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="Per-episode exponential learning rate decay")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="GAE discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ppo_epsilon", type=float, default=0.2, help="PPO epsilon")
    parser.add_argument("--initial_std", type=float, default=1.0, help="Initial value of the std used in the gaussian policy")
    parser.add_argument("--value_scale", type=float, default=1.0, help="Value loss scale factor")
    parser.add_argument("--entropy_scale", type=float, default=0.01, help="Entropy loss scale factor")
    parser.add_argument("--horizon", type=int, default=128, help="Number of steps to simulate per training step")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of PPO training epochs per traning step")
    parser.add_argument("--batch_size", type=int, default=32, help="Epoch batch size")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to train for (0 or less trains forever)")

    # VAE parameters
    parser.add_argument("--vae_model", type=str,
                        default="vae/models/seg_bce_cnn_zdim64_beta1_kl_tolerance0.0_data/",
                        help="Trained VAE model to load")
    parser.add_argument("--vae_model_type", type=str, default="cnn", help="VAE model type (\"cnn\" or \"mlp\")")
    parser.add_argument("--vae_z_dim", type=int, default=None, help="Size of VAE bottleneck")

    # Environment settings
    parser.add_argument("--synchronous", type=int, default=False, help="Set this to True when running in a synchronous environment")
    parser.add_argument("--fps", type=int, default=100, help="Set this to the FPS of the environment")
    parser.add_argument("--action_smoothing", type=float, default=0.9, help="Action smoothing factor")
    parser.add_argument("-start_carla", action="store_true", help="Automatically start CALRA with the given environment settings")

    # Training parameters
    parser.add_argument("--model_name", type=str, help="Name of the model to train. Output written to models/model_name")
    parser.add_argument("--percentage", type=str, default=0.5, help="Percentage of human experience")
    parser.add_argument("--reward_fn", type=str,
                        default="reward_speed_centering_angle_add",
                        help="Reward function to use. See reward_functions.py for more info.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed to use. (Note that determinism unfortunately appears to not be garuanteed " +
                             "with this option in our experience)")
    parser.add_argument("--eval_interval", type=int, default=10, help="Number of episodes between evaluation runs")
    parser.add_argument("--record_eval", type=bool, default=True,
                        help="If True, save videos of evaluation episodes " +
                             "to models/model_name/videos/")
    
    parser.add_argument("-restart", action="store_true",
                        help="If True, delete existing model in models/model_name before starting training")

    params = vars(parser.parse_args())

    # Remove a couple of parameters that we dont want to log
    start_carla = params["start_carla"]; del params["start_carla"]
    restart = params["restart"]; del params["restart"]

    # Reset tf graph
    tf.reset_default_graph()

    # Start training
    train(params, start_carla, restart)
