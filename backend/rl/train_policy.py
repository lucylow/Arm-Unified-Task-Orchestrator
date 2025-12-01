from stable_baselines3 import PPO
import gym

def train_ppo_policy(env_name='mobile_gui_env', total_timesteps=100000):
    env = gym.make(env_name)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_mobile_gui_policy")

if __name__ == '__main__':
    train_ppo_policy()
