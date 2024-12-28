import argparse
import logging
from qlearner import Learner
from graphics.World import PongWorld
import pandas as pd 

"""
You can install the necessary libraries by running the following command in your terminal:
pip install -r requirements.txt
"""
DEFAULT_CANVAS_SIZE = (32, 24)
DEFAULT_PADDLE_LENGTH = 7
DEFAULT_BALL_VELOCITY = 1

def main(args: argparse.Namespace) -> None:
    args.train_episodes = 1000                  # Number of Training Episodes
    args.opponent_strategy = 'almost_perfect'   # Opponent strategy
    args.discount = 0.95                        # Discount Factor
    args.epsilon = 0.1                          # Epsilon (Exploration probability)
    args.alpha = 0.2                            # Learning Rate

    env = PongWorld(args)

    learner = Learner(
        env=env,
        alpha=args.alpha,
        gamma=args.discount,
        epsilon=args.epsilon,
        choose_unexplored_first=args.choose_unexplored_first,
    )

    q_table = learner.qlearning(
        train_episodes=args.train_episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        plot_scores=args.plot_scores,
        eval_with_game=args.eval_with_game
    )

    if q_table is None:
        logging.error("The Q-learning method returned None. Check the implementation.")

    q_table_df = pd.DataFrame(
        [(state_action[0], state_action[1], value) for state_action, value in q_table.items()],
        columns=["State", "Action", "Value"]
    )

    logging.info("Q-table DataFrame:\n%s", q_table_df)
    
    # Play the game with the learned Q-table
    if args.final_show:
        learner.play_final_game()

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the Pong game and Q-learning agent.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a Q-learning agent on Pong.")

    # Game parameters
    game_group = parser.add_argument_group("Game Parameters")
    game_group.add_argument("--canvas_size", nargs=2, default=DEFAULT_CANVAS_SIZE, type=int, help="Size of the game canvas (width, height).")
    game_group.add_argument("--paddle_length", default=DEFAULT_PADDLE_LENGTH, type=int, help="Length of the paddle.")
    game_group.add_argument("--velocity", default=DEFAULT_BALL_VELOCITY, type=int, help="Ball velocity.")

    # Q-learning parameters
    qlearn_group = parser.add_argument_group("Q-learning Parameters")
    qlearn_group.add_argument("--max_iter", default=1000, type=int, help="Maximum iterations per episode.")
    qlearn_group.add_argument("--learning_rate", default=0.2, type=float, help="Learning rate for Q-learning.")
    qlearn_group.add_argument("--discount", default=1, type=float, help="Discount factor for future rewards.")
    qlearn_group.add_argument("--epsilon", default=0.09, type=float, help="Epsilon for epsilon-greedy policy.")
    qlearn_group.add_argument("--alpha", default=0.1, type=float, help="Exploration rate for the agent.")
    qlearn_group.add_argument("--train_episodes", default=1000, type=int, help="Number of training episodes.")

    qlearn_group.add_argument("--eval_episodes", default=10, type=int, help="Number of evaluation episodes.")
    qlearn_group.add_argument("--eval_every", default=10, type=int, help="Evaluate every N episodes.")
    qlearn_group.add_argument("--plot_scores", action="store_true", default=True, help="Plot training scores.")
    qlearn_group.add_argument("--final_show", action="store_true", default=True, help="Show the final gameplay after training.")
    qlearn_group.add_argument("--fps", default=30, type=int, help="Frames per second for the gameplay.")
    qlearn_group.add_argument("--eval_with_game", action="store_true", default=True, help="Show the final gameplay after training.")

    # Strategies
    strategy_group = parser.add_argument_group("Agent and Opponent Strategies")
    strategy_group.add_argument("--agent_strategy", default="eps_greedy", type=str, help="Strategy for the agent.")
    strategy_group.add_argument("--choose_unexplored_first", action="store_true", help="Prefer unexplored actions.")
    strategy_group.add_argument("--opponent_strategy", default="almost_perfect", type=str, help="Strategy for the opponent.")

    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    main(args)
