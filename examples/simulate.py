from environment.environment import PacmanEnv
import time

env = PacmanEnv(scale=7, time_bw_epi=5, move_left=.5, is_beneficial=.5, update_freq=15)
env.simulate(num_episodes=15)
hangtime = 3
# for i in range(5):
#     env.simulate_one_epi(hangtime)