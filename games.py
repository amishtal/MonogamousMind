import numpy as np
import scipy
from scipy.stats import nbinom

'''
Interface for players. Any class/object/individual
that will be playing games should either implement this
interface or have a function that can create an 
appropriate `Player` object.

Certain games may require specific considerations when
implementing these functions (perhaps different forms
for `prev_results`) or additional methods.
'''
class Player(object):
    def reset(self):
        '''
        Called before playing a new game. Should
        reset any internal state of the player
        that is relevant to only one game.
        '''
        pass
        
    def get_initial_action(self):
        '''
        Used to get the initial actions of the
        player.
        '''
        raise NotImplementedError()

    def get_action(self, prev_results):
        '''
        Used to get an action from a player given
        the results of a previous round of play.

        `prev_results` : list of dictionaries
            Holds results of the previous
            round of play. `prev_results[0]`
            holds results of the current player.
        '''
        raise NotImplementedError()



'''
A two player game where each player has the
same two actions to take.

Players are assumed to have a `getAction` method
that returns a non-negative integer value. In
the case of two actions, these values can be either
0 or 1.
'''
class TwoPlayerGame(object):
    def __init__(self):
        raise NotImplementedError('Base game class \
                is not instantiable!')

    def play(self, player1, player2, n_rounds=None):
        if n_rounds is None:
            n_rounds = nbinom.rvs(1,0.2,1) + 1

        player1.reset()
        player2.reset()

        total_payoffs = np.zeros((1,2))

        p1_payoff = 0
        p1_action = None
        p1_results = {}

        p2_payoff = 0
        p2_action = None
        p2_results = {}

        payoffs = np.zeros((1,2))

        p1_trace = []
        p2_trace = []
        for i in range(0, n_rounds):
            if i == 0:
                p1_action = player1.get_initial_action()
                p2_action = player2.get_initial_action()
            else:
                p1_results['payoff'] = p1_payoff
                p1_results['action'] = p1_action

                p2_results['payoff'] = p2_payoff
                p2_results['action'] = p2_action

                p1_action = player1.get_action([p1_results, p2_results])
                p2_action = player2.get_action([p2_results, p1_results])

            p1_trace.append(p1_action)
            p2_trace.append(p2_action)

            p1_payoff = self.payoff_matrix[p1_action, p2_action]
            p2_payoff = self.payoff_matrix[p2_action, p1_action]

            total_payoffs += [[p1_payoff, p2_payoff]]

        traces = (p1_trace, p2_trace)
        avg_payoffs = total_payoffs / float(n_rounds)

        return avg_payoffs, traces

    def play_against_trace(self, player, trace):
        n_rounds = len(trace)

        player.reset()

        total_payoffs = np.zeros((1,2))

        p_payoff = 0
        p_action = None
        p_results = {}

        t_payoff = 0
        t_action = None
        t_results = {}

        payoffs = np.zeros((1,2))

        p_trace = []
        for i in range(0, n_rounds):
            if i == 0:
                p_action = player.get_initial_action()
                t_action = trace[0]
            else:
                p_results['payoff'] = p_payoff
                p_results['action'] = p_action

                t_results['payoff'] = t_payoff
                t_results['action'] = t_action

                p_action = player.get_action([p_results, t_results])
                t_action = trace[i]

            p_trace.append(p_action)

            p_payoff = self.payoff_matrix[p_action, t_action]
            t_payoff = self.payoff_matrix[t_action, p_action]

            total_payoffs += [[p_payoff, t_payoff]]

        traces = (p_trace, trace)
        avg_payoffs = total_payoffs / float(n_rounds)

        return avg_payoffs, traces

       

class Snowdrift(TwoPlayerGame):
    def __init__(self, payoff_matrix):
        # Matrix should be in form:
        #  [ [ P, T ],
        #    [ S, R ]]
        # Ensure relevant inequalities hold.
        P = payoff_matrix[0,0] # Mutual defection.
        R = payoff_matrix[1,1] # Mutual cooperation.
        S = payoff_matrix[1,0] # Cooperator.
        T = payoff_matrix[0,1] # Defector.
        if T > R and R > S and S > P:
            self.payoff_matrix = payoff_matrix
        else:
            raise ValueError('Snowdrift game requires T > R > S > P.')

class PrisonersDilemma(TwoPlayerGame):
    def __init__(self, payoff_matrix):
        # Matrix should be in form:
        #  [ [ P, T ],
        #    [ S, R ]]
        # Ensure relevant inequalities hold.
        P = payoff_matrix[0,0] # Mutual defection.
        R = payoff_matrix[1,1] # Mutual cooperation.
        S = payoff_matrix[1,0] # Cooperator.
        T = payoff_matrix[0,1] # Defector.
        if T > R and R > P and P > S:
            self.payoff_matrix = payoff_matrix
        else:
            raise ValueError('Prisoner\'s dilemma game requires T > R > P > S.')

class StagHunt(TwoPlayerGame):
    def __init__(self, payoff_matrix):
        # Matrix should be in form:
        #  [ [ P, T ],
        #    [ S, R ]]
        # Ensure relevant inequalities hold.
        P = payoff_matrix[0,0] # Mutual defection.
        R = payoff_matrix[1,1] # Mutual cooperation.
        S = payoff_matrix[1,0] # Cooperator.
        T = payoff_matrix[0,1] # Defector.
        if R > T and T >= P and P > S:
            self.payoff_matrix = payoff_matrix
        else:
            raise ValueError('Stag hunt game requires R > T >= P > S.')
