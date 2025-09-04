import numpy as np
import random

class PrioritizedReplayBuffer:
    """
    A Prioritized Experience Replay (PER) buffer.
    This buffer stores transitions and samples them based on their TD error,
    allowing the agent to learn more efficiently from important experiences.
    """
    def __init__(self, capacity, alpha=0.6):
        """
        Initializes the PER buffer.

        Args:
            capacity (int): The maximum number of transitions to store.
            alpha (float): The prioritization exponent (0=uniform, 1=full priority).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.beta = 0.4  # Initial value for importance-sampling weights

    def add(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the buffer.
        New experiences are given the highest priority to ensure they are sampled soon.
        """
        max_priority = np.max(self.priorities) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Samples a batch of experiences from the buffer using prioritized sampling.

        Args:
            batch_size (int): The number of experiences to sample.
            beta (float): The importance-sampling exponent.

        Returns:
            tuple: A tuple containing the sampled states, actions, rewards,
                   next_states, dones, importance-sampling weights, and indices.
        """
        if len(self.buffer) < self.capacity:
            priorities = self.priorities[:self.position]
        else:
            priorities = self.priorities

        # Calculate sampling probabilities based on priorities
        probabilities = priorities ** self.alpha
        probabilities /= np.sum(probabilities)

        # Sample indices based on the calculated probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]

        # Calculate importance-sampling (IS) weights
        total_samples = len(self.buffer)
        weights = (total_samples * probabilities[indices]) ** (-beta)
        weights /= np.max(weights) # Normalize for stability
        weights = np.array(weights, dtype=np.float32)

        # Unpack the samples
        states, actions, rewards, next_states, dones = zip(*samples)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), weights, indices)

    def update_priorities(self, indices, errors, epsilon=1e-6):
        """
        Updates the priorities of sampled experiences based on their TD error.

        Args:
            indices (list): The indices of the experiences to update.
            errors (np.ndarray): The new TD errors for the experiences.
            epsilon (float): A small constant to ensure no priority is zero.
        """
        for i, error in zip(indices, errors):
            self.priorities[i] = np.abs(error) + epsilon

    def __len__(self):
        return len(self.buffer)