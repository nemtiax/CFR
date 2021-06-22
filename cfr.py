import numpy as np

#This class is responsible for keeping track of the game state and for evaluating payoffs at terminal states
#No part of the regret minimization algorithm lives in here.
class Game:
    def __init__(self):
        self.deck = ["1","2","3"]
        self.history = ["-","-","-","-","-"]
        self.current_player = -1 #chance node
        self.current_round = 0

    def deal(self):
        np.random.shuffle(self.deck)
        self.history[0] = self.deck[0]
        self.history[1] = self.deck[1]
        self.current_player = 0
        self.current_round = 2


    def get_masked_history(self,player):
        temp = self.history.copy()
        temp[1-player] = "?"
        return "".join(temp)

    def take_action(self,action_id):
        if action_id==0: #pass
            self.history[self.current_round] = "p"
        else: #bet
            self.history[self.current_round] = "b"

        self.current_round += 1
        self.current_player = 1 - self.current_player

    def undo(self):
        self.current_round -= 1
        self.history[self.current_round] = "-"
        self.current_player = 1 - self.current_player

    def is_terminal(self):
        if self.history[self.current_round - 2] == "b": #we had a bet and a response
            return True
        if self.history[self.current_round -2] == "p" and self.history[self.current_round-1]=="p": #two consecutive passes
            return True

        return False

    def get_terminal_values(self):
        assert(self.is_terminal())

        if self.history[2] == "p" and self.history[3]=="p": #two consecutive passes
            if self.history[0] > self.history[1]:
                return (1,-1)
            else:
                return (-1,1)
        elif (self.history[2] == "b" and self.history[3] == "b") or (self.history[3] == "b" and self.history[4] == "b"): #two consecutive bets
            if self.history[0] > self.history[1]:
                return (2,-2)
            else:
                return (-2,2)
        elif self.history[2] == "p": #player 1 passed, then player 2 bet (we already checked for pass-pass), then player 1 passed (we already checked for pass-bet-bet)
            return (-1,1)
        elif self.history[2] == "b": #player 1 bet, then player 2 passed (we already checked for bet-bet)
            return (1,-1)


#The strategyMap class is responsible for book-keeping our regrets and strategies for each information set
class StrategyMap:
    def __init__(self):

        #because Kuhn poker only has a handful of information sets, we specify them by hand.
        #The notation is as follows:
        #p1card p2card p1action1 p2action2 p1action3
        #p1card and p2card indicates the cards dealt to player 1 and player 2 respectively.
        #Known card values are 1 2 or 3
        #A question mark indicates an unknown card (i.e. the card belonging to the other player)
        #The actions are p for pass and b for bet. A - means that an action has not yet occurred.


        #For example, 1?pb- means player 1 was dealt a 1 (the fact that we know this means this infoset belongs to player 1),
        #player 2's card is unknown, player1 opened with a pass, and player2 responded with a bet.
        self.infosets = ("1?---","2?---","3?---","1?pb-","2?pb-","3?pb-","?1p--","?2p--","?3p--","?1b--","?2b--","?3b--")

        #these maps hold our cumulative regrets and strategies.
        #The keys for these maps are infoset strings (listed above)
        #The values are |A|-element vectors (in the case of Kuhn poker, 2 elements).

        #For cumulative_regret_map, a value vector represents the total regret we have felt for taking each action in a particular infoset
        #Index 0 corresponds to passing, index 1 corresponds to betting.
        #So, cumulative_regret_map['?1p--'] = [5,-2] means that, in aggregate, we have felt 5 regret for passing with a 1 in hand as player 2 after player 1 passed,
        # and -2 regret for betting in the same situation
        #cumulative_regret_map is used to calculate our current strategy at any particular training set.
        self.cumulative_regret_map = {}
        #In the case of cumulative_strategy_map, a value vector represents the cumulative frequency with which we have chosen each strategy in a particular infoset
        #Index 0 corresponds to passing, index 1 corresponds to betting.
        #So cumulative_strategy_map['1?---'] = [7,3] means that as player 1, when dealt a 1 card, we have, in aggregate, passed with weight 7 and bet with weight 3
        #If we normalize cumulative_strategy_map so that each value vector sums to 1, we get out final average strategy.
        self.cumulative_strategy_map = {}

        #we initialize our cumulative regret and strategies to be 0 for all infosets.
        for infoset in self.infosets:
            self.cumulative_regret_map[infoset] = [0,0]
            self.cumulative_strategy_map[infoset] = [0,0]

    #convenience method for pulling out the cumulative regret for a paritcular infoset as indicated by a masked history string
    def get_cumulative_regret(self,masked_history):
        return self.cumulative_regret_map[masked_history]

    #convenience method for incrementing our cumulative regret for a particualr infoset
    def add_cumulative_regret(self,masked_history,regrets):
        self.cumulative_regret_map[masked_history] += regrets


    #here we calculate our current strategy from our cumulative regrets using regret-matching.
    #if we have any actions with positive regret, we will choose each action proportionally to its positive regret. Actions with negative regret will not be chosen.
    #On the other hand, if we have no actions with positive regret, we will play uniformly.
    #following this method of generating strategies will garauntee that our average strategy approaches a nash equilibrium
    def get_current_strategy(self,masked_history):
        strategy = np.zeros(2)
        positive_regret_sum = 0
        cumulative_regrets = self.get_cumulative_regret(masked_history)
        for a in range(2):
            if cumulative_regrets[a] > 0:
                strategy[a] = cumulative_regrets[a]
                positive_regret_sum += cumulative_regrets[a]

        if positive_regret_sum > 0:
            strategy =  strategy / positive_regret_sum
        else:
            strategy = np.full((2,),0.5)

        return strategy

    #we need to remember our cumulative strategy so that at the end we can take the average and get an approximate equilibrium strategy
    #we will weight each individual local strategy by our contribution to how likely we were to reach the corresponding infoset.
    #intuitively, if we are currently playing a strategy that rarely plays to infoset X, we will not put much weight on what that strategy wants
    #to do when it arrives at X.  If instead our current strategy always plays towards infoset X, we will put more weight.
    #The exact value of the weight is the product of how likely we were to make each decision that lead to the indicated infoset.
    #We only weight according to our own action probabilities - we imagine that the opponent and the chance nodes had played towards
    #this infoset with probability 1.
    def add_to_cumulative_strategy(self,masked_history,local_strategy,weight):
        self.cumulative_strategy_map[masked_history] += weight*local_strategy

    #convenience method for pulling out our cumulative strategy for a particular infoset
    def get_cumulative_strategy(self,masked_history):
        return self.cumulative_strategy_map[masked_history]

    #our average strategy at the end of training will approach a nash equilibrium.  We compute it simply by normalizing the cumulative strategy
    #for each infoset
    def get_average_strategy(self,masked_history):
        cumulative_strategy = self.get_cumulative_strategy(masked_history)
        return cumulative_strategy/np.sum(cumulative_strategy)

#The core CFR algorithm.  This function takes the following arguments:
#   -game, a Game object containing the current state of the game
#   -strategy_map, a StrategyMap object responsible for calculating the current strategy and updating cumulative regret
#   -p1 and p2, the probability of reaching the current node (ignoring chance) given the current strategies, decomposed into each player's contribution.
#
#This function returns the expected values of the current position for each player, and does book-keeping to update regrets in the strategy_map
def cfr(game,strategy_map,p1,p2):
    current_player = game.current_player

    #if we're at a terminal state, there's no more moves to be made, and we can just return the current payoffs
    if game.is_terminal():
        return game.get_terminal_values()

    #otherwise, we need to consider which move to make.
    #we first extract a masked history string for the current player, which will be a representation of our current information set
    masked_history = game.get_masked_history(current_player)
    #given our information set, strategy_map computes the current regret-matching strategy we should follow
    #this strategy is a probability distribution over the moves available to us
    local_strategy = strategy_map.get_current_strategy(masked_history)
    #when updating our cumulative local strategy, we need to weight according to how likely our current strategy was to lead us to this node
    #This is the product of the probabilities that we selected the appropriate action at each ancestor of this game state where it was our turn
    #this is captured by p1 if we're currently player1, and p2 if we're currently player2
    if current_player == 0:
        strategy_weight = p1
    else:
        strategy_weight = p2

    #log our current strategy to our cumulative strategy so that we can compute our average strategy at the end
    strategy_map.add_to_cumulative_strategy(masked_history,local_strategy,strategy_weight)

    #we now need to compute regret for each action we could take.
    #this requires knowing two things - the expected payoff we get for taking each action given our current strategy profile, and the average utility we could get
    #from this state by just following our current mixed strategy.
    #in other words, how much do I regret playing my current strategy instead of swapping to a pure strategy that plays each action?
    action_utilities = np.zeros((2,2))
    expected_utilities = np.zeros((2,))
    #loop over actions, speculatively executing them, and recursively calling cfr() to compute the expected payoff given that action
    for a in range(2):
        #apply the action to move the game to the next state.  We will rever this change when we're done considering this action.
        game.take_action(a)
        #we need to update our contribution to the probability of reaching this state.
        if current_player==0: #adjust p1
            new_p1 = p1 * local_strategy[a]
            new_p2 = p2
        else:
            new_p1 = p1
            new_p2 = p2 * local_strategy[a]
        #compute the utility of taking action a using a recursive call to cfr
        action_utilities[a] = cfr(game,strategy_map,new_p1,new_p2)
        #the overall expected utility of this state is just the weighted average of each action's utility, weighted by the probability of taking that action under the current mixed strategy
        expected_utilities += local_strategy[a] * action_utilities[a]
        #now that we're done considering this action, we need to revert the game back to before we took it
        game.undo()

    #given the utility for each action, and the expected utility for the current state, we can compute the regret we feel for not playing each action as a pure strategy
    regrets = np.zeros((2,))
    for a in range(2):
        #our regret is simply the utility we would receive from the action, minus the utility we get from our current strategy
        regrets[a] = action_utilities[a][current_player] - expected_utilities[current_player]
        #when updating our cumulative regrets, we need to weight by how likely our opponent was to bring us to this state.
        #if the current state is very improbable, we shouldn't count our regret very much.  Intuitively, we don't want to over
        #adjust our strategy to handle rare cases.
        if current_player == 0:
            regrets[a] *= p2
        else:
            regrets[a] *= p1
        strategy_map.add_cumulative_regret(masked_history,regrets)

    return expected_utilities


#to train our strategy, we simply run a large number of games through the cfr algorithm
#we will monitor our expected utilities to see how well we approach the true EV of the game
s = StrategyMap()
total_expected_utils = np.zeros((2,))
for i in range(1000000):
    g = Game()
    g.deal()
    expected_utils = cfr(g,s,1,1)
    total_expected_utils += expected_utils
    print(i,total_expected_utils/(i+1))

#we can now extract our final strategy for each infoset.
for infoset in s.infosets:
    print(infoset,s.get_average_strategy(infoset),s.get_cumulative_regret(infoset))
