import matplotlib.pyplot as plt
import numpy as np
import pygame
from qlearner import AGENT, OPPONENT, BALL, REWARDS, ACTIONS, Learner
from graphics.GameObjects import Ball, Paddle

X, Y = 0, 1
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480

MOVE_KEYS = {
    pygame.K_UP: (OPPONENT, -1),
    pygame.K_DOWN: (OPPONENT, 1),
    pygame.K_w: (AGENT, -1),
    pygame.K_s: (AGENT, 1),
}


class PongWorld:
    def __init__(self, args):
        self.args = args
        self.Qlearn = Learner

        Ball.VELOCITY = args.velocity
        Paddle.VELOCITY = args.velocity
        Paddle.LENGTH = args.paddle_length

        self.canvas_width, self.canvas_height = args.canvas_size
        pygame.init()

        # Initializing ball and paddles
        self.ball = Ball((self.canvas_width // 2, self.canvas_height // 2))
        paddle_y = (self.canvas_height - Paddle.LENGTH) // 2
        self.paddles = [
            Paddle((0, paddle_y), "green"),
            Paddle((self.canvas_width - 1, paddle_y), "red"),
        ]
    
    def render_game_to_image(self):
      pygame_image = pygame.surfarray.array3d(self.window)
      return np.transpose(pygame_image, (1, 0, 2)) 

    def show_game_frame(self):
        image = self.render_game_to_image()
        plt.imshow(image)
        plt.axis('off') 
        plt.show()

    def handle_ball_movement(self, cx, cy):
        nx = cx + self.ball.direction[X] * Ball.VELOCITY
        ny = cy + self.ball.direction[Y] * Ball.VELOCITY

        in_p1_area = self.ball.in_paddle_area(self.paddles[AGENT])
        in_p2_area = self.ball.in_paddle_area(self.paddles[OPPONENT])

        touching_p1 = nx <= 1
        touching_p2 = nx >= self.canvas_width - 2

        # clamp and change ball direction
        if in_p1_area and touching_p1:
            nx = 1
            self.ball.direction[X] *= -1
        if in_p2_area and touching_p2:
            nx = self.canvas_width - 2
            self.ball.direction[X] *= -1

        # WALL THICKNESS = 1
        hit_top = ny <= 1
        hit_bottom = ny >= self.canvas_height - 1

        if hit_top:
            ny = 1
            self.ball.direction[Y] *= -1
        if hit_bottom:
            ny = self.canvas_height - 1
            self.ball.direction[Y] *= -1

        return nx, ny

    def handle_paddle_movement(self, paddle, direction):
        ny = paddle.rect.y + direction * Paddle.VELOCITY
        ny = max(ny, 0)
        ny = min(ny, self.canvas_height - Paddle.LENGTH)

        return ny

    def game_reset(self):
        self.ball.reset()
        for paddle in self.paddles:
            paddle.reset()

    def get_eval_strategies(self):
        if self.args.agent_strategy == "almost_perfect":
            agent_eval_strategy = "almost_perfect"
        elif self.args.agent_strategy in ["greedy", "eps_greedy"]:
            agent_eval_strategy = "greedy"
        else:
            agent_eval_strategy = "random"

        opponent_eval_strategy = self.args.opponent_strategy

        return agent_eval_strategy, opponent_eval_strategy

    def get_state(self, initial=False):
        if initial:
            self.game_reset()

        return (
            self.paddles[AGENT].get_position(),
            self.paddles[OPPONENT].get_position(),
            self.ball.get_position(),
        )

    def is_final_state(self, state, num_iter):
        bx, _ = state[BALL]

        return (num_iter >= self.args.max_iter) or (bx <= 0) or (bx >= self.canvas_width)

    def get_reward_description(self, state, num_iter):
        bx, _ = state[BALL]

        if num_iter >= self.args.max_iter:
            return "default"
        if bx <= 0:
            return "lose"
        if bx >= self.canvas_width:
            return "win"

        return "default"

    def get_legal_actions(self, player_idx, state):
        player_y = state[player_idx]
        legal_actions = ["STAY"]

        if player_y >= Paddle.VELOCITY:
            legal_actions.append("UP")
        if player_y + Paddle.LENGTH <= self.canvas_height - Paddle.VELOCITY:
            legal_actions.append("DOWN")

        return legal_actions

    def apply_actions(self, state, action_agent, action_opponent, num_iter):
        """
        Returns next_state, reward
        state = (my paddle pos, adv paddle pos, ball pos)
        """
        reward = 0

        # Move ball
        bx, by = state[BALL]
        nx, ny = self.handle_ball_movement(bx, by)
        next_ball = (nx, ny)
        self.ball.set_position(nx, ny)

        # Move paddles according to action
        agent_y = self.handle_paddle_movement(self.paddles[AGENT], ACTIONS[action_agent])
        opponent_y = self.handle_paddle_movement(self.paddles[OPPONENT], ACTIONS[action_opponent])

        self.paddles[AGENT].set_position(agent_y)
        self.paddles[OPPONENT].set_position(opponent_y)

        next_state = (agent_y, opponent_y, next_ball)

        if self.is_final_state(next_state, num_iter):
            r = self.get_reward_description(next_state, num_iter)
            reward = REWARDS.get(r, 0)

        return next_state, reward

    def get_perfect_action(self, player_idx, state):
        paddle_y = state[player_idx]
        _, by = state[BALL]

        legal_actions = self.get_legal_actions(player_idx, state)

        if by > paddle_y and "DOWN" in legal_actions:
            return "DOWN" # Follow ball down
        
        if by < paddle_y and "UP" in legal_actions:
            return "UP"   # Follow ball up

        if by == paddle_y:
            if self.ball.direction[Y] == -1:
                return "UP"
            if self.ball.direction[Y] == 1:
                return "DOWN"

        return "STAY"


    def set_text_params(self):
        self.text_params = [
            self.font.render(
                "canvas_size = {} x {}".format(self.canvas_width, self.canvas_height),
                True,
                pygame.Color("white"),
            ),
            self.font.render("paddle_length = {}".format(Paddle.LENGTH), True, pygame.Color("white")),
            self.font.render(
                "eps = {} / lr = {} / gamma = {} / max_iter = {}".format(
                    self.args.epsilon,
                    self.args.learning_rate,
                    self.args.discount,
                    self.args.max_iter,
                ),
                True,
                pygame.Color("white"),
            ),
            self.font.render(
                "train_episodes = {}".format(self.args.train_episodes),
                True,
                pygame.Color("white"),
            ),
            self.font.render(
                "agent_strategy = {}".format(self.args.agent_strategy),
                True,
                pygame.Color("white"),
            ),
            self.font.render(
                "opponent_strategy = {}".format(self.args.opponent_strategy),
                True,
                pygame.Color("white"),
            ),
            self.font.render(
                "alpha = {}".format(self.args.alpha),
                True,
                pygame.Color("white"),
            ),
        ]

    def final_show(self, Qlearner, menu_options, logging):

        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.canvas = pygame.surface.Surface((self.canvas_width, self.canvas_height))
        self.font = pygame.font.SysFont("Arial", 12)
        self.text_params = self.font.render("Testing fonts", True, pygame.Color("white"))
        self.set_text_params()
        pygame.display.set_caption("Pong Q-Learning")

        score = num_iter = 0
        agent_eval_strategy, opponent_eval_strategy = self.get_eval_strategies()

        state = self.get_state(initial=True)
        game_sleep = max(1, 1000 // self.args.fps)
        text_offset = 15

        while not self.is_final_state(state, num_iter):
            p_act = Qlearner.choose_action_by_strategy(
                AGENT,
                agent_eval_strategy,
                state,
                self.get_legal_actions(AGENT, state),
            )
            o_act = Qlearner.choose_action_by_strategy(
                OPPONENT,
                opponent_eval_strategy,
                state,
                self.get_legal_actions(OPPONENT, state),
            )

            py, oy, (bx, by) = state
            self.ball.set_position(bx, by)
            self.paddles[AGENT].set_position(py)
            self.paddles[OPPONENT].set_position(oy)

            self.canvas.fill(pygame.Color("black"))
            self.ball.draw(self.canvas)

            for paddle in self.paddles:
                paddle.draw(self.canvas)

            self.window.blit(
                pygame.transform.scale(self.canvas, (WINDOW_WIDTH, WINDOW_HEIGHT)),
                (0, 0),
            )

            for i, tp in enumerate(self.text_params):
                self.window.blit(tp, (WINDOW_WIDTH // 2 - 100, text_offset * (i + 1)))

            pygame.display.update()
            pygame.time.delay(game_sleep)

            state, reward = self.apply_actions(state, p_act, o_act, num_iter)
            score += reward
            num_iter += 1

        win_status = self.get_reward_description(state, num_iter)
        if win_status == 'win':
            logging.info("\n*** Win! ***")
        else:
            logging.info("\n*** Lose ***")
        print(menu_options)

