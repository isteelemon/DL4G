from jass.agents.Agent_rulebased import Agent_rulebased
from jass.game.game_util import *
from jass.game.game_sim import GameSim
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena


if __name__ == '__main__':
        rule = RuleSchieber()
        game = GameSim(rule=rule)
        agent = Agent_rulebased()

        np.random.seed(2)
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)
        obs = game.get_observation()
        cards = convert_one_hot_encoded_cards_to_str_encoded_list(obs.hand)
        print(cards)
        trump = agent.action_trump(obs)

        # tell the simulation the selected trump
        game.action_trump(trump)


        # play the game to the end and print the result
        while not game.is_done():
                game.action_play_card(agent.action_play_card(game.get_observation()))

        print(game.state.points)

