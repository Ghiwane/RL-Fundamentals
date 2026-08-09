class GridWorld():
    """Custom GridWorld environment for Reinforcement Learning tasks."""

    def __init__(self, size=5):
        """Initialize environment variables, grid parameters, and reset state."""
        self.size = size                            # Grid size (5x5 by default)
        self.start_pos = [0, 0]                     # Starting coordinates of the agent
        self.goal_pos = [4, 4]                      # Target destination coordinates
        self.walls = [[1, 1], [2, 2], [4, 2], [3, 4]] # List of non-traversable obstacle coordinates
        self.step_count = 0                         # Counter for current episode steps
        self.agent_pos = None                       # Current position of the agent
        self.reset()                                # Initialize state upon instantiation

    def reset(self):
        """Reset the environment state for a new episode."""
        self.agent_pos = self.start_pos.copy()      # Create a independent copy of starting position
        self.step_count = 0                         # Reset step counter to zero
        return self.agent_pos                       # Return initial observation

    def move(self, new_pos):
        """Evaluate movement validity, update state, and return (state, reward, done)."""
        self.step_count += 1                        # Increment total steps taken
        reward = 0                                  # Default reward value
        done = False                                # Episode termination flag

        # Check if the intended position is within grid bounds and not inside a wall
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and new_pos not in self.walls:
            self.agent_pos = new_pos.copy()         # Successfully update agent position

            # Case 1: Agent reaches the target goal
            if new_pos == self.goal_pos: 
                done = True                         # End episode
                reward = 10                         # Positive terminal reward
        
            # Case 2: Max step limit reached on valid move
            elif self.step_count >= 100:
                done = True                         # End episode due to truncation limit
                reward = -1                         # Step penalty
                        
            # Case 3: Standard valid step
            else:
                reward = -1                         # Standard movement cost to encourage shortest path
        
        # Invalid move: Out of bounds or wall collision
        else:
            reward = -5                             # Penalty for illegal move (agent position unchanged)
            if self.step_count >= 100:
                done = True                         # End episode if max step limit reached

        return self.agent_pos, reward, done

    def step(self, action):
        """Execute action, map it to grid coordinates, and trigger move logic."""
        x, y = self.agent_pos

        # Action mapping: 0 = Up, 1 = Down, 2 = Left, 3 = Right
        if action == 0:
            new_pos = [x, y - 1]                    # Move UP (decrease row/y index)
        elif action == 1:
            new_pos = [x, y + 1]                    # Move DOWN (increase row/y index)
        elif action == 2:
            new_pos = [x - 1, y]                    # Move LEFT (decrease col/x index)
        else:
            new_pos = [x + 1, y]                    # Move RIGHT (increase col/x index)

        return self.move(new_pos)

    def render(self):
        """Display ASCII visual representation of current grid state."""
        for y in range(self.size):
            line = ""
            for x in range(self.size):
                pos = [x, y]
                if pos == self.agent_pos:
                    line += " A "                   # Agent current position symbol
                elif pos == self.goal_pos:
                    line += " G "                   # Goal position symbol
                elif pos in self.walls:
                    line += " # "                   # Wall/obstacle symbol
                else:
                    line += " . "                   # Empty cell symbol
            print(line)