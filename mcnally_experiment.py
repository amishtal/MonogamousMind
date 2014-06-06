import array
import operator
import pprint
import random


import numpy as np
import scipy
from scipy.stats import nbinom

import h5py

from deap import base, creator, tools


import base as mm_base
import games
import neural_nets as nnets
import neural_nets.mcnally_net
import neural_nets.fast_mcnally_net


def evaluate(pop, intel_penalty=0.01, game=None):
    n = len(pop)
    total_payoffs = np.zeros((1, n))
    rate_of_coop = np.zeros((1,n))

    class NNPlayer(games.Player):
        def __init__(self, nnet):
            self.nnet = nnet

        def reset(self):
            self.nnet.reset()

        def get_initial_action(self):
            return self.nnet.initial_move

        def get_action(self, prev_results):
            prev_payoffs = [r['payoff'] for r in prev_results]
            output = self.nnet.activate(prev_payoffs)
            if type(output) == np.ndarray:
                output = output[0,0]
            else:
                output = output

            if np.random.rand() < output:
                return 1
            else:
                return 0

    for i in range(n):
        for j in range(i+1, n):
            n_rounds = nbinom.rvs(1, 0.2, 1) + 1
            payoffs, traces = game.play(NNPlayer(pop[i]), NNPlayer(pop[j]), n_rounds)
            total_payoffs[0,i] += payoffs[0,0]
            total_payoffs[0,j] += payoffs[0,1]

            rate_of_coop[0, i] += np.mean(traces[0])
            rate_of_coop[0, j] += np.mean(traces[1])

    total_payoffs /= float(n-1)
    rate_of_coop /= float(n-1)

    for i in range(n):
        #pop[i].fitness.values = [total_payoffs[0,i] - intel_penalty*pop[i].get_intelligence()] 
        pop[i].fitness.values = [total_payoffs[0,i], pop[i].get_intelligence()] 
        pop[i].rate_of_coop = rate_of_coop[0, i]


    # Examine the strategies in the population.
    class AlwaysCooperatePlayer(games.Player):
        def get_initial_action(self):
            return 1
        def get_action(self, prev_payoffs):
            return 1

    class AlwaysDefectPlayer(games.Player):
        def get_initial_action(self):
            return 0
        def get_action(self, prev_results):
            return 0

    class TitForTatPlayer(games.Player):
        def get_initial_action(self):
            return 1
        def get_action(self, prev_results):
            # Default cooperate, but defect if opponent defected.
            if prev_results[1]['action'] == 0:
                return 0
            else:
                return 1

    class TitForTwoTatsPlayer(games.Player):
        def __init__(self):
            self.opponent_defected = False
        def reset(self):
            self.opponent_defected = False
        def get_initial_action(self):
            return 1
        def get_action(self, prev_results):
            # Default cooperate, but defect if opponent defected twice
            # in a row.
            if prev_results[1]['action'] == 0 and self.opponent_defected:
                return 0
            elif prev_results[1]['action'] == 0:
                self.opponent_defected = True
                return 1
            else:
                self.opponent_defected = False
                return 1

    class PavlovPlayer(games.Player):
        def get_initial_action(self):
            return 1
        def get_action(self, prev_results):
            return prev_results[1]['action']

    class ProbabilisticPlayer(games.Player):
        def __init__(self, prob):
            # prob -> probability of cooperating
            self.prob = prob
        def get_initial_action(self):
            self.get_action(None)
        def get_action(self, prev_results):
            if np.random.rand() <= prob:
                return 1
            else:
                return 1

    probs = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_games = 5
    n_rounds = 20
    n_total = len(probs)*n_games*n_rounds

    test_players = [AlwaysCooperatePlayer(),
                    AlwaysDefectPlayer(),
                    TitForTatPlayer(),
                    TitForTwoTatsPlayer(),
                    PavlovPlayer()]
    test_player_moves = np.zeros((len(test_players), n_total))
    pop_moves = np.zeros((len(pop), n_total))
    opp_moves = np.zeros(len(probs)*n_games*n_rounds)

    start_idx = 0
    for p in probs:
        for i in range(n_games):
            stop_idx = start_idx + n_rounds

            random_trace = [np.random.rand() < p for i in range(n_rounds)]
            random_trace = np.array(random_trace, dtype='float')

            opp_moves[start_idx:stop_idx] = random_trace

            for (j, player) in enumerate(test_players):
                _, traces = game.play_against_trace(player, random_trace)
                test_player_moves[j, start_idx:stop_idx] = traces[0]

            for (j, indiv) in enumerate(pop):
                _, traces = game.play_against_trace(NNPlayer(indiv), random_trace)
                pop_moves[j, start_idx:stop_idx] = traces[0]

            start_idx += n_rounds

    # (X - Y)^2 = X^2 - 2XY + Y^2
    pop_squared = np.square(pop_moves).sum(axis=1)
    test_squared = np.square(test_player_moves).sum(axis=1)
    pop_test = np.dot(pop_moves, test_player_moves.T)
    sq_dists = pop_squared[:,np.newaxis] - 2 * pop_test + test_squared[np.newaxis,:]
    sq_dists /= n_total

    closest_strat = sq_dists.argmin(axis=1)
    for (i, indiv) in enumerate(pop):
        indiv.closest_strategy = closest_strat[i]
        indiv.strategy_dists = sq_dists[i,:]

    return pop

def main():
    nn_class = nnets.fast_mcnally_net

    sd_game = games.Snowdrift(  np.array([[1,8], [2,5]]))
    pd_game = games.PrisonersDilemma(np.array([[2,7], [1,6]]))
    sh_game = games.StagHunt(  np.array([[1,1], [0,2]]))

    creator.create('FitnessMax', mm_base.WeightedSumFitness, weights=[1., -0.01])
    creator.create('Individual', nn_class.McNallyNet, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    n_input = 2
    max_n_cognitive = 10
    max_init_cognitive = 3
    n_output = 1
    toolbox.register('individual', creator.Individual,
            n_input, max_n_cognitive, n_output, max_init_cognitive)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate, intel_penalty=0.01, game=sh_game)
    toolbox.register('mate', nn_class.crossover)

    prob_mutate_weights = 0.1
    prob_mutate_structure = 0.02
    toolbox.register('mutate', nn_class.mutate, prob_mutate_weights=prob_mutate_weights, prob_mutate_structure=prob_mutate_structure)
    toolbox.register('select', tools.selRoulette)

    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_fit.register('min', np.min, axis=0)
    stats_fit.register('max', np.max, axis=0)
    stats_fit.register('std', np.std, axis=0)
    stats_fit.register('mean', np.mean, axis=0)

    stats_intel = tools.Statistics(key=lambda ind: ind.get_intelligence())
    stats_intel.register('min', np.min)
    stats_intel.register('max', np.max)
    stats_intel.register('std', np.std)
    stats_intel.register('mean', np.mean)

    stats_coop = tools.Statistics(key=lambda ind: ind.rate_of_coop)
    stats_coop.register('min', np.min)
    stats_coop.register('max', np.max)
    stats_coop.register('std', np.std)
    stats_coop.register('mean', np.mean)

    stats_strat_percent_diff = tools.Statistics(key=lambda ind: ind.strategy_dists)
    stats_strat_percent_diff.register('min', np.min, axis=0)
    stats_strat_percent_diff.register('max', np.max, axis=0)
    stats_strat_percent_diff.register('std', np.std, axis=0)
    stats_strat_percent_diff.register('mean', np.mean, axis=0)

    mstats = tools.MultiStatistics(
        fitness=stats_fit,
        intelligence=stats_intel,
        cooperation=stats_coop,
        strategy_percent_diff=stats_strat_percent_diff
    )
    #mstats.register('min', np.min)
    #mstats.register('max', np.max)
    #mstats.register('std', np.std)
    #mstats.register('mean', np.mean)

    pp = pprint.PrettyPrinter(width=50)

    # Main Algorithm
    NGEN = 10000
    CXPB = 0.1
    MUTPB = 1.0

    # Initialize population.
    pop = toolbox.population(n=50)
    records = []

    # Evaluate population.
    #pop = toolbox.evaluate(pop)
    #record = stats.compile(pop)
    #print record
    for g in range(NGEN):
        print '-- Generation {} --'.format(g)
        pop = toolbox.evaluate(pop)
        record = mstats.compile(pop)
        records.append(record)
        pp.pprint(record)
        fitnesses = [ind.fitness.wsum() for ind in pop]
        intels   = [ind.get_intelligence() for ind in pop]
        #print fitnesses
        #print intels
        print '  Average fitness:      {}'.format(np.mean(fitnesses))
        print '  Average intelligence: {}'.format(np.mean(intels))

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        #offspring = toolbox.evaluate(offspring)
        pop[:] = offspring

        save_file = h5py.File('results.h5', 'w')
        mm_base.dictToHDFS(records, save_file)
        save_file.close()

if __name__ == '__main__':
    main()
