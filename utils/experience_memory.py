
import random
from collections import namedtuple

Experience = namedtuple("Experience", ['obs', 'action', 'reward', 'next_obs', 'done'])

class ExperienceMemory(object):
    
    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.mem_idx = 0
        self.memory = []
        
    def store(self, experience):
        if self.mem_idx < self.capacity:
            self.memory.append(None)
        self.memory[self.mem_idx % self.capacity] = experience
        self.mem_idx += 1
       
    def sample(self, batch_size):
        assert batch_size <= len(self.memory), "Sample batch_size is more than available space in memory"
        return random.sample(self.memory, batch_size)
    
    def get_size(self):
        return len(self.memory)
        
        
