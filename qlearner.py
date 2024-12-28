import numpy as np
import logging
import pickle
import pygame
import sys
import matplotlib.pyplot as plt

AGENT, OPPONENT, BALL = 0, 1, 2
ACTIONS = {"UP": -1, "DOWN": 1, "STAY": 0}
DIRECTIONS = {
    "top_left": [-1, -1],
    "top_right": [1, -1],
    "bot_left": [-1, 1],
    "bot_right": [1, 1],
}
REWARDS = {"lose": -1, "win": 1, "default": 0}

class Learner:
    def __init__(self, env, alpha, gamma, epsilon, choose_unexplored_first):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.choose_unexplored_first = choose_unexplored_first
        self.Q = {}
        self.training_error = []
        print("Agent Strategy and Opponent", self.env.args.agent_strategy, self.env.args.opponent_strategy)

    
    def qlearning(self, train_episodes, eval_every, eval_episodes, plot_scores, eval_with_game):
        """
        Perform Q-learning for a specified number of episodes.
        """
        train_scores = []               # List to store scores for training episodes
        eval_scores = []                # List to store evaluation scores

        for episode in range(1, train_episodes + 1):
            score, num_iter, state = 0, 0, self.env.get_state(initial=True)  # Resets Environment for each episode

            while not self.env.is_final_state(state, num_iter):
                # Selecting actions for both agent and opponent
                action_agent = self.select_action(state, AGENT)
                action_opponent = self.select_action(state, OPPONENT)

                # Applying actions to the environment and getting the next state and reward
                state_next, reward = self.env.apply_actions(state, action_agent, action_opponent, num_iter)
                
                self.update_q_value(state, action_agent, state_next, reward) # Q-value update based on the received reward

                state = state_next      # Moving to the next state
                score += reward         # Accumulating the score
                num_iter += 1       

            train_scores.append(score)  # Storing the score for this episode

            # Evaluating policy periodically
            if episode % eval_every == 0:
                if eval_with_game:
                    # Evaluating and storing the average score over several episodes
                    avg_score = self.evaluate_policy(eval_episodes)
                    eval_scores.append(avg_score)
                    logging.info(f"Episode {episode}: Avg Eval Score = {avg_score}")
                else:
                    logging.info(f"Episode {episode}: Score = {score}")

        # Plotting the training and evaluation scores if requested
        if plot_scores:
            print("Number of states:", len(self.Q))
            self.eval_train_scores(eval_every, train_scores, eval_scores)
            self.plot_training_error()

        return self.Q


    def select_action(self, state, player_type):
        """
        Select an action based on the player's strategy and the current state.
        """
        strategy = self.env.args.agent_strategy if player_type == AGENT else self.env.args.opponent_strategy
        legal_actions = self.env.get_legal_actions(player_type, state)
        return self.choose_action_by_strategy(player_type, strategy, state, legal_actions)

    
    def choose_action_by_strategy(self, player_idx, strategy, state, actions):
        if strategy == "greedy":
            return self.get_best_action(state, actions)
        elif strategy == "eps_greedy":
            return self.epsilon_greedy(state, actions)
        elif strategy == "almost_perfect":
            return self.get_almost_perfect_action(player_idx, state, actions)
        elif strategy == "perfect":
            return self.env.get_perfect_action(player_idx, state)

        return np.random.choice(actions) # Strategy = Random

    
    def epsilon_greedy(self, state, legal_actions):
        """
        Decides whether to explore or exploit 
        """
        # Optionally perform epsilon decay to decrease exploration over time
        # *** Write your code here (For question 5) ***  

        if np.random.uniform(0, 1) <= self.epsilon:
            # If set, Chooses an unexplored state randomly
            if self.choose_unexplored_first:
                unexplored = [a for a in legal_actions if (state, a) not in self.Q]
                if unexplored:
                    return np.random.choice(unexplored)
            # Explore (Randomly return an action from the list of legal actions)
            # *** Write your code here ***
            return
        else:
            # Exploit (Call a function to return the best action with the highest Q-value)
            # *** Write your code here ***
            return
            

    def get_best_action(self, state, legal_actions):
        """
        Selects the best action based on the Q-values for the given state and legal actions.
        
        Parameters:
        - state (tuple): The current state, consisting of (agent_y, ball_pos).
        - legal_actions (list): A list of actions that the agent is allowed to take in the current state.
        
        Returns:
        - The action with the highest Q-value.
        """
        agent_state = (state[AGENT], state[BALL])
        
        # Get the Q-values for each legal action, defaulting to 0 if not available
        # You need to calculate the Q-values for each action. If the Q-value for the action is not available, it should default to 0.
        # *** Write your code here ***
        
        # Return the action with the maximum Q-value
        # *** Write your code here ***

        return best_action


    def get_almost_perfect_action(self, player_idx, state, legal_actions):
        if np.random.uniform(0, 1) <= 0.7:
            return self.env.get_perfect_action(player_idx, state)
        else:
            return np.random.choice(legal_actions)
 

    def update_q_value(self, state, action, state_next, reward):
        """
        Update the Q-value using the Q-learning update rule.
        """
        my_state = state[AGENT], state[BALL]
        my_state_next = state_next[AGENT], state_next[BALL]

        actions_next = self.env.get_legal_actions(AGENT, state_next)

        # Select the best action for the next state (state_next) using by calling a function
        # *** Write your code here ***

        # Get the Q-value for the best action in the next state (state_next). If not found, default to 0.
        # *** Write your code here ***

        # Get the current Q-value for the given state-action pair (my_state, action). (Set default to 0.)
        # *** Write your code here***

        # Compute the Temporal Difference (TD) error, which is the difference between the target and the current Q-value.
        # *** Write your code here ***

        # Update the Q-value using the Q-learning update formula: Q(s, a) = Q(s, a) + alpha * TD error.
        # *** Write your code here ***
        
        self.training_error.append(abs(td_error))

    def evaluate_policy(self, eval_episodes):
        """
        Evaluate the current policy for a number of episodes.
        """
        total_score = 0
        for _ in range(eval_episodes):
            total_score += self.play_game_evaluate()
        return total_score / eval_episodes

    def play_game_evaluate(self):
        score = num_iter = 0

        agent_eval_strategy, opponent_eval_strategy = self.env.get_eval_strategies()

        state = self.env.get_state(initial=True)

        while not self.env.is_final_state(state, num_iter):
            p_actions = self.env.get_legal_actions(AGENT, state)
            o_actions = self.env.get_legal_actions(OPPONENT, state)

            p_act = self.choose_action_by_strategy(AGENT, agent_eval_strategy, state, p_actions)
            o_act = self.choose_action_by_strategy(OPPONENT, opponent_eval_strategy, state, o_actions)

            state, reward = self.env.apply_actions(state, p_act, o_act, num_iter)
            score += reward
            num_iter += 1

        return score

    def play_final_game(self):
        """
        Play the game using the learned Q-table.
        """
        menu_options = """Press a number on keyboard:
        1) Play Game
        2) Opponent Strategy: Random
        3) Opponent Strategy: Almost Perfect
        4) Opponent Strategy: Perfect
        5) Exit
        """
        self.env.final_show(self, menu_options, logging)

        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logging.info("Quit event detected.")
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_key_press(event.key, menu_options)

            clock.tick(20)
            
        pygame.quit()
        sys.exit()


    def handle_key_press(self, key, menu_options):
        """
        Handle key press events during gameplay.
        """
        if key == pygame.K_1:
            logging.info("Starting game...")
            self.env.final_show(self, menu_options, logging)
        elif key == pygame.K_2:
            self.env.args.opponent_strategy = "random"
            logging.info("Opponent strategy set to Random.")
        elif key == pygame.K_3:
            self.env.args.opponent_strategy = "almost_perfect"
            logging.info("Opponent strategy set to Almost Perfect.")
        elif key == pygame.K_4:
            self.env.args.opponent_strategy = "perfect"
            logging.info("Opponent strategy set to Perfect.")
        elif key == pygame.K_5:
            logging.info("Exiting game...")
            pygame.quit()
            sys.exit()


    def eval_train_scores(self, eval_every, train_scores, eval_scores):
        """
        Plots the training and evaluation scores over episodes.

        Parameters:
        - train_scores (list): List of training scores over episodes.
        - eval_scores (list): List of evaluation scores at evaluation intervals.
        """
        plt.figure(figsize=(10, 6))
        
        smoothing_window = 20
        smoothed_train_scores = np.convolve(
            train_scores, np.ones(smoothing_window) / smoothing_window, mode="same"
        )
        
        train_episodes = np.arange(1, self.env.args.train_episodes + 1)
        eval_episodes = np.arange(eval_every, self.env.args.train_episodes + 1, eval_every)
        
        plt.plot(
            train_episodes,
            smoothed_train_scores,
            label="Training Scores",
            linewidth=1.0,
            color="blue",
        )
        
        plt.plot(
            eval_episodes,
            eval_scores,
            label="Evaluation Scores",
            linewidth=0.5,
            color="red",
            marker="o",
        )

        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Average Score", fontsize=12)
        plt.title("Training and Evaluation Scores Over Episodes", fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()


    def plot_training_error(self):
        """
        Plots the TD errors averaged over every 500 steps.
        """
        batch_size = 500
        avg_errors = [
            np.mean(self.training_error[i:i + batch_size])
            for i in range(0, len(self.training_error), batch_size)
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(avg_errors)), avg_errors, label="Average TD Error (every 500 steps)", alpha=0.7)
        plt.xlabel("Batch Index (500 steps per batch)")
        plt.ylabel("Average Temporal Difference Error")
        plt.title("TD Error Averaged Over Every 500 Steps")
        plt.legend()
        plt.grid()
        plt.show()

