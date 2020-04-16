import matplotlib.pyplot as plt

class LinearDecaySchedule(object):
    
    def __init__(self, initial_value, final_value, max_steps):
        assert initial_value > final_value, "initial_value should be greater than final_value"
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_factor = (initial_value - final_value) / max_steps
        
    def __call__(self, step_num):
        current_value = self.initial_value - (self.decay_factor * step_num)
        if current_value < self.final_value:
            current_value = self.final_value           
        return current_value
    
if __name__ == "__main__":
    epsilon_initial = 1.0
    epsilon_final = 0.05
    MAX_NUM_EPISODES = 10000
    MAX_STEPS_PER_EPISODE = 300
    MAX_STEPS = MAX_NUM_EPISODES * MAX_STEPS_PER_EPISODE
    linear_sched = LinearDecaySchedule(initial_value = epsilon_initial, final_value = epsilon_final, max_steps = MAX_STEPS)
    epsilon = [linear_sched(step) for step in range(MAX_STEPS)]
    plt.plot(epsilon)
    plt.show()
    
    
