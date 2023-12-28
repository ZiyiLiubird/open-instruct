import numpy as np
import torch
import torch.multiprocessing as mp
from collections import deque



class DistributedReplayBuffer():
    def __init__(self, buffer_size:int, pad_token_id:int):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.pad_token_id = pad_token_id

    def add(self, batch_sequence):
        for sequence in batch_sequence:
            self.buffer.append(sequence)

    def sample(self, num_samples:int):
        indexes :list = np.random.randint(0, len(self.buffer), num_samples)
        sequence_list :list[torch.Tensor] = [self.buffer[index] for index in indexes]
    
        return sequence_list
    
    def __len__(self):
        return len(self.buffer)