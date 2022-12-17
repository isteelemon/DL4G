import numpy as np
from jass.agents.agent import Agent
from jass.game.const import card_strings, next_player, partner_player
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from anytree import Node


class MCTS_agent(Agent):

    def __init__(self, agent_name, iterations=40):

        self.iterations = iterations
        self.__rule = RuleSchieber()
        self.root: Node = None
        self.current_node: Node = None

    def choose_card(self, obs: GameObservation) -> int:

        self.init_tree(obs)
        for i in range(self.iterations):
            self.monte_carlo_tree_search(obs)
            if len(self.root.children) == 1 and self.root.expanded:
                break

        best_node = self.best_node()
        return best_node.card

    def cardnames(self, cards):
        cardnames = list()
        for card, value in enumerate(cards):
            if value == 1:
                cardnames.append(card_strings[card])
        return cardnames

    def init_tree(self, obs: GameObservation):

        scores = np.zeros(4)
        for trick, player in enumerate(obs.trick_winner):
            if player < 0:
                break
            scores[player] += obs.trick_points[trick]

        played_cards = np.zeros(36)
        for trick in obs.tricks:
            if trick[0] == -1:
                break
            for card in trick:
                if card == -1:
                    break
                played_cards[card] = 1

        own_cards = obs.hand.copy()

        self.root = Node("Root", player=next_player[next_player[next_player[obs.player_view]]], trick=obs.nr_tricks,
                         trick_cards=obs.current_trick.copy(), played_cards=played_cards, own_cards=own_cards,
                         expanded=False, scores=scores, count=0, card=None, total_payoff=0)
        self.current_node = self.root

    def monte_carlo_tree_search(self, obs: GameObservation):

        # 1. Selection
        self.selection()
        # 2. Expansion
        self.expansion(obs)
        # 3. Simulation
        payoff = self.simulation(obs)
        # 4. Backpropagation
        self.backpropagation(payoff)

    def selection(self):

        self.__current_node = self.root
        while self.current_node.expanded and not self.current_node.is_leaf:
            # select child node by tree policy
            best_node = None
            # Cannot be zero as early confidences can be zero as well
            best_confidence = -1
            for node in self.current_node.children:
                confidence = self.calculate_ucb1(self.node)
                if best_confidence < confidence:
                    best_confidence = confidence
                    best_node = node
            self.current_node = best_node

    def calc_ucb1(self, node):
        ucb1 = (node.total_payoff / node.count) + 1 * (np.sqrt(np.log(self.root.count) / node.count))
        return ucb1

    def expansion(self, obs):
        if self.current_node.expanded:
            return
        is_my_turn = next_player[self.current_node.player] == obs.player_view
        valid_cards = None
        if is_my_turn:
            own_cards = self.current_node.own_cards.copy()
            nr_cards_in_trick = int(
                np.sum(self.current_node.played_cards) % 4)
            valid_cards = self.rule.get_valid_cards(own_cards,
                                                    self.current_node.trick_cards,
                                                    nr_cards_in_trick,
                                                    obs.trump)
        else:
            valid_cards = np.ones(36)
            valid_cards = np.subtract(
                valid_cards, self.current_node.own_cards)
            valid_cards = np.subtract(
                valid_cards, self.current_node.played_cards)
        valid_actions = list()
        for card, value in enumerate(valid_cards):
            if value == 1:
                valid_actions.append(card)
        unexplored_actions = valid_actions.copy()
        for node in self.current_node.children:
            unexplored_actions.remove(node.card)

        # we now have fully expanded the node
        if len(unexplored_actions) <= 1:
            self.current_node.expanded = True

        if 0 < len(unexplored_actions):
            # select an unexplored action / card by policy (currently random)
            card = unexplored_actions[-1]
            player = next_player[self.current_node.player]
            own_cards = self.current_node.own_cards.copy()
            if player == obs.player_view:
                own_cards[card] = 0
            played_cards = self.current_node.played_cards.copy()
            trick_position = int(np.sum(played_cards)) % 4
            played_cards[card] = 1
            if trick_position == 0:
                current_trick = self.current_node.trick + 1
                trick_cards = np.full(4, -1)
            else:
                current_trick = self.current_node.trick
                trick_cards = self.current_node.trick_cards.copy()
            trick_cards[trick_position] = card
            scores = self.current_node.scores.copy()

            # a trick was finished
            if trick_position == 3:
                first_player = next_player[player]
                trick_points = self.rule.calc_points(
                    trick_cards, current_trick == 8, obs.trump)
                winner = self.rule.calc_winner(
                    trick_cards, first_player, obs.trump)
                scores[winner] += trick_points

            next_node = Node(f"{int(np.sum(played_cards))}.{card}", parent=self.current_node,
                             player=player, trick=current_trick, trick_cards=trick_cards,
                             played_cards=played_cards, own_cards=own_cards, card=card,
                             expanded=False, scores=scores, count=0, total_payoff=0)
            self.current_node = next_node

    def __do_simulation(self, obs):
        payoff = 0
        # a leaf we already calculated the payoff for
        if self.current_node.expanded:
            if 0 < self.current_node.total_payoff:
                payoff = self.current_node.total_payoff / self.current_node.count
            # a leaf need to calculate the payoff for
            else:
                payoff = self.calculate_payoff(obs, self.current_node.scores)
        else:
            current_player = next_player[self.current_node.player]
            trick_cards = self.current_node.trick_cards.copy()
            own_cards = self.current_node.own_cards.copy()
            played_cards = self.current_node.played_cards.copy()
            unplayed_cards_others = np.ones(36)
            unplayed_cards_others = np.subtract(unplayed_cards_others, own_cards)
            unplayed_cards_others = np.subtract(unplayed_cards_others, played_cards)
            simulated_scores = self.current_node.scores.copy()
            actions_others = list()
            for card, value in enumerate(unplayed_cards_others):
                if value == 1:
                    actions_others.append(card)

            # play remaining tricks
            for played_card_count in range(int(np.sum(self.current_node.played_cards)), 36):
                current_trick = int(played_card_count / 4)
                trick_position = played_card_count % 4
                if current_player == obs.player_view:
                    valid_cards = self.rule.get_valid_cards(own_cards,
                                                            trick_cards,
                                                            played_card_count % 4,
                                                            obs.trump)
                    valid_actions = list()
                    for card, value in enumerate(valid_cards):
                        if value == 1:
                            valid_actions.append(card)
                else:
                    valid_actions = actions_others

                card = np.random.choice(valid_actions)
                trick_cards[trick_position] = card
                played_cards[card] = 1
                if current_player == obs.player_view:
                    own_cards[card] = 0
                else:
                    unplayed_cards_others[card] = 0
                    actions_others.remove(card)
                if trick_position == 3:
                    first_player = next_player[current_player]
                    trick_points = self.rule.calc_points(
                        trick_cards, current_trick == 8, obs.trump)
                    winner = self.rule.calc_winner(
                        trick_cards, first_player, obs.trump)
                    simulated_scores[winner] += trick_points
                    trick_cards.fill(-1)
                    current_player = winner
                else:
                    current_player = next_player[current_player]
            payoff = self.calculate_payoff(obs, simulated_scores)
        return payoff

    def calculate_payoff(self, obs, scores):
        player = obs.player_view
        partner = partner_player[player]
        payoff = (scores[player] + scores[partner]) / 157
        return payoff

    def backpropagation(self, payoff):
        for node in self.current_node.iter_path_reverse():
            node.count += 1
            node.total_payoff += payoff

    def expansion(self, obs):
        if self.current_node.expanded:
            return
        is_my_turn = next_player[self.current_node.player] == obs.player_view
        valid_cards = None
        if is_my_turn:
            own_cards = self.current_node.own_cards.copy()
            nr_cards_in_trick = int(
                np.sum(self.current_node.played_cards) % 4)
            valid_cards = self.rule.get_valid_cards(own_cards,
                                                    self.current_node.trick_cards,
                                                    nr_cards_in_trick,
                                                    obs.trump)
        else:
            valid_cards = np.ones(36)
            valid_cards = np.subtract(
                valid_cards, self.current_node.own_cards)
            valid_cards = np.subtract(
                valid_cards, self.current_node.played_cards)
        valid_actions = list()
        for card, value in enumerate(valid_cards):
            if value == 1:
                valid_actions.append(card)
        unexplored_actions = valid_actions.copy()
        for node in self.current_node.children:
            unexplored_actions.remove(node.card)

        # we now have fully expanded the node
        if len(unexplored_actions) <= 1:
            self.current_node.expanded = True

        if 0 < len(unexplored_actions):
            # select an unexplored action / card by policy (currently random)
            card = unexplored_actions[-1]
            player = next_player[self.current_node.player]
            own_cards = self.current_node.own_cards.copy()
            if player == obs.player_view:
                own_cards[card] = 0
            played_cards = self.current_node.played_cards.copy()
            trick_position = int(np.sum(played_cards)) % 4
            played_cards[card] = 1
            if trick_position == 0:
                current_trick = self.current_node.trick + 1
                trick_cards = np.full(4, -1)
            else:
                current_trick = self.current_node.trick
                trick_cards = self.current_node.trick_cards.copy()
            trick_cards[trick_position] = card
            scores = self.current_node.scores.copy()

            # a trick was finished
            if trick_position == 3:
                first_player = next_player[player]
                trick_points = self.rule.calc_points(
                    trick_cards, current_trick == 8, obs.trump)
                winner = self.rule.calc_winner(
                    trick_cards, first_player, obs.trump)
                scores[winner] += trick_points

            next_node = Node(f"{int(np.sum(played_cards))}.{card}", parent=self.current_node,
                             player=player, trick=current_trick, trick_cards=trick_cards,
                             played_cards=played_cards, own_cards=own_cards, card=card,
                             expanded=False, scores=scores, count=0, total_payoff=0)
            self.current_node = next_node

    def simulation(self, obs):
        payoff = 0
        # a leaf we already calculated the payoff for
        if self.current_node.expanded:
            if 0 < self.current_node.total_payoff:
                payoff = self.current_node.total_payoff / self.current_node.count
            # a leaf need to calculate the payoff for
            else:
                payoff = self.__calculate_payoff(obs, self.current_node.scores)
        else:
            current_player = next_player[self.current_node.player]
            trick_cards = self.current_node.trick_cards.copy()
            own_cards = self.current_node.own_cards.copy()
            played_cards = self.current_node.played_cards.copy()
            unplayed_cards_others = np.ones(36)
            unplayed_cards_others = np.subtract(unplayed_cards_others, own_cards)
            unplayed_cards_others = np.subtract(unplayed_cards_others, played_cards)
            simulated_scores = self.current_node.scores.copy()
            actions_others = list()
            for card, value in enumerate(unplayed_cards_others):
                if value == 1:
                    actions_others.append(card)

            # play remaining tricks
            for played_card_count in range(int(np.sum(self.current_node.played_cards)), 36):
                current_trick = int(played_card_count / 4)
                trick_position = played_card_count % 4
                if current_player == obs.player_view:
                    valid_cards = self.rule.get_valid_cards(own_cards,
                                                            trick_cards,
                                                            played_card_count % 4,
                                                            obs.trump)
                    valid_actions = list()
                    for card, value in enumerate(valid_cards):
                        if value == 1:
                            valid_actions.append(card)
                else:
                    valid_actions = actions_others

                card = np.random.choice(valid_actions)
                trick_cards[trick_position] = card
                played_cards[card] = 1
                if current_player == obs.player_view:
                    own_cards[card] = 0
                else:
                    unplayed_cards_others[card] = 0
                    actions_others.remove(card)
                if trick_position == 3:
                    first_player = next_player[current_player]
                    trick_points = self.rule.calc_points(
                        trick_cards, current_trick == 8, obs.trump)
                    winner = self.rule.calc_winner(
                        trick_cards, first_player, obs.trump)
                    simulated_scores[winner] += trick_points
                    trick_cards.fill(-1)
                    current_player = winner
                else:
                    current_player = next_player[current_player]
            payoff = self.calculate_payoff(obs, simulated_scores)
        return payoff

    def best_node(self):
        best_node = self.root.children[0]
        for node in self.root.children:
            if best_node.count < node.count:
                best_node = node
        return best_node
