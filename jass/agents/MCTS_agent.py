from jass.agents.MCTS_logic import MCTS_logic
from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation


class MCTS_agent(Agent):
    def __init__(self, name, max_iterations=80):
        self.__play_strategy = MCTS_logic(name, max_iterations)

    def action_trump(self, obs: GameObservation) -> int:
        return self.__play_strategy.choose_card(obs)

    def action_play_card(self, obs: GameObservation) -> int:
        return self.__play_strategy.choose_card(obs)
