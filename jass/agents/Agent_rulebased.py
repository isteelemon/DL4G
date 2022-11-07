from jass.game.game_util import *
from jass.game.game_sim import GameSim
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena

TRUMP_THRESHOLD = 68

class Agent_rulebased(Agent):


    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        # add your code here using the function above



        # score if the color is trump
        trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
        # score if the color is not trump
        no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
        # score if obenabe is selected (all colors)
        obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0, ]
        # score if uneufe is selected (all colors)
        uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]


        best_cardsum = 0
        best_trump = -1
        card_list = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        for current_trump in trump_ints:
            current_cardsum = 0
            for card in card_list:
                color = color_of_card[card]
                offset = offset_of_card[card]
                if color == current_trump:
                    current_cardsum += trump_score[offset]
                elif (current_trump == 4):
                    current_cardsum += obenabe_score[offset]
                elif (current_trump == 5):
                    current_cardsum += uneufe_score[offset]
                else:
                    current_cardsum += no_trump_score[offset]

            if (best_cardsum < current_cardsum):
                best_cardsum = current_cardsum
                best_trump = current_trump

        if(best_cardsum >= TRUMP_THRESHOLD or obs.forehand == 0):

            print(f"Picking: {trump_strings_german_long[best_trump]} as Trump with score {best_cardsum}")
            return best_trump
        else:
            print(f"Pushing as best result was {trump_strings_german_long[best_trump]} as Trump with score {best_cardsum}")
            return PUSH


    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # we use the global random number generator here
        return np.random.choice(np.flatnonzero(valid_cards))
