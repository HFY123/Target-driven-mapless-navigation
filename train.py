import os
import random
import shutil
import time
import math
import scipy
import csv

import numpy as np
import tensorflow.compat.v1 as tf

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

    with open(f'oneturn1_record\data_nohuman_' + model_name[-1] + '.csv', "a+", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['step', 'reward', 'distance'])
    # Set seeds
    if isinstance(seed, int):
        tf.random.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(1)

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
    print("Creating environment")
    env = CarlaEnv(obs_res=(160, 80),
                   action_smoothing=action_smoothing,
                   encode_state_fn=encode_state_fn,
                   reward_fn1=reward_functions["reward_speed_centering_angle_add"],
                   reward_fn2=reward_functions["reward_speed_centering_angle_add_eval"],
                   synchronous=synchronous,
                   fps=fps,
                   start_carla=False)
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
            '''
            env.success_rate.append(env.success_num / episode_idx)
            env.collision_rate.append(env.collision_num / episode_idx)
            env.stop_rate.append(env.stop_num / episode_idx)
            env.off_track_rate.append(env.off_track_num / episode_idx)
            env.speeding_rate.append(env.speeding_num / episode_idx)
            
            f1=open('success_rate.txt','ab')
            f2=open('collision_rate.txt','a')
            f3=open('stop_rate.txt','a')
            f4=open('off_track_rate.txt','a')
            f5=open('speeding_rate.txt','a')

            np.savetxt(f1, env.success_rate)
            np.savetxt(f2, env.collision_rate)
            np.savetxt(f3, env.stop_rate)
            np.savetxt(f4, env.off_track_rate)
            np.savetxt(f5, env.speeding_rate)
            #print(scipy.special.rel_entr(a, taken_actions))
            '''
            # Calculate last value (bootstrap value)
            _, last_values = model.predict(state) # []
            
            # Compute GAE
            advantages = compute_gae(rewards, values, last_values, dones, discount_factor, gae_lambda)
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            

            # Flatten arrays
            states        = np.array(states)
            taken_actions = np.array(taken_actions)
            returns       = np.array(returns)
            advantages    = np.array(advantages)

            T = len(rewards)
            assert states.shape == (T, *input_shape)
            assert taken_actions.shape == (T, num_actions)
            assert returns.shape == (T,)
            assert advantages.shape == (T,)

            # Train for some number of epochs
            model.update_old_policy() # θ_old <- θ
            for _ in range(num_epochs):
                num_samples = len(states)
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
                    model.train(states[mb_idx], taken_actions[mb_idx],
                                returns[mb_idx], advantages[mb_idx])
        
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

        with open(f'oneturn1_record\data_nohuman_' + model_name[-1] + '.csv', "a+", newline='') as csvfile:
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
    #settings.fixed_delta_seconds = 0.05
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
    parser.add_argument("--action_smoothing", type=float, default=0.5, help="Action smoothing factor")
    parser.add_argument("-start_carla", action="store_true", help="Automatically start CALRA with the given environment settings")

    # Training parameters
    parser.add_argument("--model_name", type=str, help="Name of the model to train. Output written to models/model_name")
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
