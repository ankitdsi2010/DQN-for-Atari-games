
from datetime import datetime
from argparse import ArgumentParser
import gym
import torch
import random
import numpy as np 

import environment.atari as Atari
import environment.utils as env_utils
from utils.params_manager import ParamsManager
from utils.decay_schedule import LinearDecaySchedule
from utils.experience_memory import Experience, ExperienceMemory
import utils.weights_initializer
from function_approximator.perceptron import SLP
from function_approximator.cnn import CNN
from tensorboardX import SummaryWriter

args = ArgumentParser("deep_Q_learner")
args.add_argument("--params-file", help="Path to the parameters json file. Default is parameters.json",
                  default="parameters.json", metavar="PFILE")
args.add_argument("--env", help="ID of the Atari environment available in OpenAI Gym.Default is SeaquestNoFrameskip-v4",
                  default="SeaquestNoFrameskip-v4", metavar="ENV")
args.add_argument("--gpu-id", help="GPU device ID to use. Default=0", default=0, type=int, metavar="GPU_ID")
args.add_argument("--render", help="Render environment to Screen. Off by default", action="store_true", default=False)
args.add_argument("--test", help="Test mode. Used for playing without learning. Off by default", action="store_true",
                  default=False)
args.add_argument("--record", help="Enable recording (video & stats) of the agent's performance",
                  action="store_true", default=False)
args.add_argument("--recording-output-dir", help="Directory to store monitor outputs. Default=./trained_models/results",
                  default="./trained_models/results")
args = args.parse_args()

params_manager = ParamsManager(args.params_file)
seed = params_manager.get_agent_params()['seed']
summary_file_path_prefix = params_manager.get_agent_params()['summary_file_path_prefix']
summary_file_path = summary_file_path_prefix + args.env + "_" + datetime.now().strftime("%y-%m-%d-%H-%M")
writer = SummaryWriter(summary_file_path)

params_manager.export_env_params(summary_file_path + "/" + "env_params.json")
params_manager.export_agent_params(summary_file_path + "/" + "agent_params.json")
global_step_num = 0
use_cuda = params_manager.get_agent_params()['use_cuda']

device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)
    

class Deep_Q_Learner(object):
    
    def __init__(self, state_shape, action_shape, params):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.params = params
        self.gamma = self.params['gamma']
        self.learning_rate = self.params['lr']
        self.best_mean_reward = - float("inf")
        self.best_reward = - float("inf")
        self.training_steps_completed = 0
        
        if len(self.state_shape) == 1:
            self.DQN = SLP
        elif len(self.state_shape) == 3:
            self.DQN = CNN
            
        self.Q = self.DQN(state_shape, action_shape, device).to(device)
        self.Q.apply(utils.weights_initializer.xavier)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        if self.params['use_target_network']:
            self.Q_target = self.DQN(state_shape, action_shape, device).to(device)
            
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = params["epsilon_max"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = LinearDecaySchedule(initial_value=self.epsilon_max, final_value=self.epsilon_min,
                                                 max_steps=self.params['epsilon_decay_final_step'])
        self.step_num = 0
        self.memory = ExperienceMemory(capacity=int(self.params['experience_memory_capacity']))
        
    def get_action(self, observation):
        observation = np.array(observation)
        observation = observation / 255
        if len(observation.shape) == 3:
            if observation.shape[2] < observation.shape[0]:
                observation.reshape(observation[2], observation[1], observation[0])
            observation = np.expand_dims(observation, 0)
        return self.policy(observation)
    
    def epsilon_greedy_Q(self, observation):
        writer.add_scalar("DQL/epsilon", self.epsilon_decay(self.step_num), self.step_num)
        self.step_num += 1
        if random.random() < self.epsilon_decay(self.step_num) and not self.params["test"]:
            action = random.choice([i for i in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(observation).data.to(torch.device('cpu')).numpy())
        return action
    
    def learn(self, obs, action, reward, obs_next, done):
        if done:
            td_target = reward + 0.0
        else:
            td_target = reward + (self.gamma * torch.max(self.Q(obs_next)))
        td_error = td_target - self.Q(obs)[action]
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()
        
    def learn_from_batch_experience(self, experiences):
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs) / 255.0
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(obs_batch.reward)        
        if self.params["clip_rewards"]:
            reward_batch = np.sign(reward_batch)
        next_obs_batch = np.array(batch_xp.next_obs) / 255.0
        done_batch = np.array(batch_xp.done)
        
        if self.params['use_target_network']:
            if self.step_num % self.params['target_network_update_freq'] == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())
            td_target = reward_batch + ~done_batch * \
                np.tile(self.gamma, len(next_obs_batch)) * \
                self.Q_target(next_obs_batch).max(1)[0].data.cpu().numpy()
        else:
            td_target = reward_batch + ~done_batch * \
                np.tile(self.gamma, len(next_obs_batch)) * \
                self.Q(next_obs_batch).max(1)[0].data.cpu().numpy()
                
        td_target = torch.from_numpy(td_target).to(device)
        action_idx = torch.from_numpy(action_batch).to(device)
        td_error = torch.nn.functional.mse_loss(self.Q(obs_batch).gather(1, action_idx.view(-1, 1)),
                                                td_target.float().unsqueeze(1))
        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        writer.add_scalar("DQL/td_error", td_error.mean(), self.step_num)
        self.Q_optimizer.step()
        
    def replay_experience(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.params['replay_batch_size']
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)
        self.training_steps_completed += 1
        
    def save(self, env_name):
        file_name = self.params['save_dir'] + "DQL_" + env_name + ".ptm"
        agent_state = {"Q": self.Q.state_dict(),
                       "best_mean_reward": self.best_mean_reward,
                       "best_reward": self.best_reward};
        torch.save(agent_state, file_name)
        print("Agent's state saved to ", file_name)
        
    def load(self, env_name):
        file_name = self.params['load_dir'] + "DQL_" + env_name + ".ptm"
        agent_state = torch.load(file_name, map_location = lambda storage, loc: storage)
        self.Q.load_state_dict(agent_state["Q"])
        self.Q.to(device)
        self.best_mean_reward = agent_state["best_mean_reward"]
        self.best_reward = agent_state["best_reward"]
        print("Loaded Q model state from", file_name,
              " which fetched a best mean reward of:", self.best_mean_reward,
              " and an all time best reward of:", self.best_reward)
        
      
if __name__ == "__main__":
    
    env_conf = params_manager.get_env_params()
    env_conf["env_name"] = args.env
    if args.test:
        env_conf["episodic_life"] = False
    rew_type = "LIFE" if env_conf["episodic_life"] else "GAME"
    
    custom_region_available = False
    for key, value in env_conf['useful_region'].items():
        if key in args.env:
            env_conf['useful_region'] = value
            custom_region_available = True
            break
    if custom_region_available is not True:
        env_conf['useful_region'] = env_conf['useful_region']['Default']
        
    print("Using env_conf:", env_conf)
    atari_env = False
    for game in Atari.get_games_list():
        if game.replace("_", "") in args.env.lower():
            atari_env = True
    if atari_env:
        env = Atari.make_env(args.env, env_conf)
    else:
        print("Given environment name is not an Atari Env. Creating a Gym env")
        env = env_utils.ResizeReshapeFrames(gym.make(args.env))
        
    if args.record:
        env = gym.wrappers.Monitor(env, args.recording_output_dir, force=True)
        
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.n
    agent_params = params_manager.get_agent_params()
    agent_params["test"] = args.test
    agent = Deep_Q_Learner(observation_shape, action_shape, agent_params)
    
    episode_rewards = list()
    prev_checkpoint_mean_ep_rew = agent.best_mean_reward
    num_improved_episodes_before_checkpoint = 0
    print("Using agent_params:", agent_params)
    if agent_params['load_trained_model']:
        try:
            agent.load(env_conf["env_name"])
            prev_checkpoint_mean_ep_rew = agent.best_mean_reward
        except FileNotFoundError:
            print("WARNING: No trained model found for this environment. Training from scratch.")
            
    episode = 0
    while global_step_num <= agent_params['max_training_steps']:
        obs = env.reset()
        cum_reward = 0.0 
        done = False
        step = 0
        while not done:
            if env_conf['render'] or args.render:
                env.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.memory.store(Experience(obs, action, reward, next_obs, done))

            obs = next_obs
            cum_reward += reward
            step += 1
            global_step_num += 1
            
            if done is True:
                episode += 1
                episode_rewards.append(cum_reward)
                if cum_reward > agent.best_reward:
                    agent.best_reward = cum_reward
                if np.mean(episode_rewards) > prev_checkpoint_mean_ep_rew:
                    num_improved_episodes_before_checkpoint += 1
                if num_improved_episodes_before_checkpoint >= agent_params["save_freq_when_perf_improves"]:
                    prev_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                    agent.best_mean_reward = np.mean(episode_rewards)
                    agent.save(env_conf['env_name'])
                    num_improved_episodes_before_checkpoint = 0
                print("\nEpisode#{} ended in {} steps. Per {} stats: reward ={} ; mean_reward={:.3f} best_reward={}".
                      format(episode, step+1, rew_type, cum_reward, np.mean(episode_rewards), agent.best_reward))
                writer.add_scalar("main/ep_reward", cum_reward, global_step_num)
                writer.add_scalar("main/mean_ep_reward", np.mean(episode_rewards), global_step_num)
                writer.add_scalar("main/max_ep_rew", agent.best_reward, global_step_num)
                if agent.memory.get_size() >= 2 * agent_params['replay_start_size'] and not args.test:
                    agent.replay_experience()

                break
    env.close()
    writer.close()
    
    
        
            
        
        
        
            
        
        
