import random
import torch
class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def push(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        # max_size 넘어가면 다시 인덱스 = 0
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        # 배치 사이즈만큼 랜덤하게 인덱스 추출
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

    def __len__(self):
        return self.size