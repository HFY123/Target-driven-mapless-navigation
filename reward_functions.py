import numpy as np
from CarlaEnv.wrappers import angle_diff, vector
import math

low_speed_timer = 0
max_distance    = 3.0  # Max distance from center before terminating
target_speed    = 10.0 # kmh

def create_reward_fn(reward_fn, max_speed=25):
    """
        Wraps input reward function in a function that adds the
        custom termination logic used in these experiments

        reward_fn (function(CarlaEnv)):
            A function that calculates the agent's reward given
            the current state of the environment. 
        max_speed:
            Optional termination criteria that will terminate the
            agent when it surpasses this speed.
            (If training with reward_kendal, set this to 20)
    """
    def func(env):
        fwd    = vector(env.vehicle.get_velocity())
        wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
        angle  = (angle_diff(fwd, wp_fwd)*180/math.pi)
        #print(angle)

        terminal_reason = "Running..."

        # Stop if speed is less than 1.0 km/h after the first 5s of an episode
        global low_speed_timer
        low_speed_timer += 1.0 / env.fps
        speed = env.vehicle.get_speed()
        speed_kmh = 3.6 * env.vehicle.get_speed()

        if low_speed_timer > 2.0 and speed < 0.5 / 3.6:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"
            env.stop_num += 1
        
        if env.collision_flag:
            env.terminal_state = True
            terminal_reason = "collision"
            env.collision_num += 1

        # Stop if distance from center > max distance
        
        if env.distance_from_center > max_distance:
            env.terminal_state = True
            terminal_reason = "Off-track"
            env.off_track_num += 1
        
        # Stop if speed is too high
        if max_speed > 0 and speed_kmh > max_speed:
            env.terminal_state = True
            terminal_reason = "Too fast"
            env.speeding_num += 1

        if env.distance_to_goal <= 3:  #--hfy
            env.terminal_state = True
            terminal_reason = "Target reached"
            env.success_num += 1


        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            if terminal_reason == "Target reached":  #--hfy
                low_speed_timer = 0.0
                reward += 500
            else:
                low_speed_timer = 0.0
                reward -= 1000

        if env.terminal_state:
            env.extra_info.extend([
                terminal_reason,
                ""
            ])
        return reward
    return func


def create_reward_fn_eval(reward_fn, max_speed=20):
    """
        Wraps input reward function in a function that adds the
        custom termination logic used in these experiments

        reward_fn (function(CarlaEnv)):
            A function that calculates the agent's reward given
            the current state of the environment. 
        max_speed:
            Optional termination criteria that will terminate the
            agent when it surpasses this speed.
            (If training with reward_kendal, set this to 20)
    """
    def func(env):
        fwd    = vector(env.vehicle.get_velocity())
        wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
        angle  = (angle_diff(fwd, wp_fwd)*180/math.pi)
        #print(angle)

        terminal_reason = "Running..."

        # Stop if speed is less than 1.0 km/h after the first 5s of an episode
        global low_speed_timer
        low_speed_timer += 1.0 / env.fps
        speed = env.vehicle.get_speed()
        speed_kmh = 3.6 * env.vehicle.get_speed()

        if low_speed_timer > 1.5 and speed < 0.5 / 3.6:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        if env.collision_flag:
            env.terminal_state = True
            terminal_reason = "collision"
        
        if env.distance_from_center > 3.0:
            env.terminal_state = True
            terminal_reason = "Off-track"
        
        # Stop if speed is too high
        if max_speed > 0 and speed_kmh > max_speed:
            env.terminal_state = True
            terminal_reason = "Too fast"

        if env.distance_to_goal <= 5:  #--hfy
            env.terminal_state = True
            terminal_reason = "Target reached"


        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            if terminal_reason == "Target reached":  #--hfy
                low_speed_timer = 0.0
                reward += 500
            else:
                low_speed_timer = 0.0
                reward -= 1000

        if env.terminal_state:
            env.extra_info.extend([
                terminal_reason,
                ""
            ])
        return reward
    return func

#---------------------------------------------------
# Create reward functions dict
#---------------------------------------------------

reward_functions = {}

# Kenall's (Learn to Drive in a Day) reward function
def reward_kendall(env):
    speed_kmh = 3.6 * env.vehicle.get_speed()
    return speed_kmh

reward_functions["reward_kendall"] = create_reward_fn(reward_kendall)

# Our reward function (additive)
def reward_speed_centering_angle_add(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               + centering factor (1 when centered, 0 when not)
               + angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)
    #print(angle)

    min_speed = 5.0 # km/h
    max_speed = 20.0 # km/h
    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh > max_speed:
        speed_reward = -speed_kmh
    else:
        if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
            speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]

        elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
            speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
        else:                                         # Otherwise
            speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0) 

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(15)), 0.0)

    Distance_factor1 = max(1.0 - env.distance_to_goal / env.distance, 0.0)  #--hfy

    Distance_factor2 = env.D  #--hfy

    #Check reward before scaling  --hfy
    #print(speed_reward, centering_factor, angle_factor, Distance_factor1, Distance_factor2)
    #print(angle_factor)

    # Final reward
    reward = speed_reward + 5*centering_factor + angle_factor + 15*Distance_factor1 + Distance_factor2 - 5 #--hfy

    return reward

reward_functions["reward_speed_centering_angle_add"] = create_reward_fn(reward_speed_centering_angle_add)

reward_functions["reward_speed_centering_angle_add_eval"] = create_reward_fn_eval(reward_speed_centering_angle_add)

# Our reward function (multiplicative)
def reward_speed_centering_angle_multiply(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 10.0 # km/h
    max_speed = 25.0 # km/h

    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh == 0:
        speed_reward = -1
    if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                  # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
    else:                                         # Otherwise
        speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)
    '''
    if intersection_offroad>0:
		self.reward-=100 
    if intersection_otherlane>0:
        self.reward-=100 
    elif collision_vehicles > 0:
        self.reward-=100
    elif collision_pedestrians >0:
        self.reward-=100  
    elif collision_other >0:
        self.reward-=50
    '''
    '''Distance_diff = env.D
    if Distance_diff > 0:
        Distance_factor = 0.5
    else:
        Distance_factor = 0.1 '''
    
    # Final reward
    reward = speed_reward * centering_factor * angle_factor - 0.1#* Distance_factor

    #reward = speed_reward * angle_factor * Distance_factor

    return reward

reward_functions["reward_speed_centering_angle_multiply"] = create_reward_fn(reward_speed_centering_angle_multiply)
