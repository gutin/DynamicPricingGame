import numpy as np
import numpy.random as rn
import multiprocessing as mp


def simulate_trial(args):
    teams, horizon, x_0, price_scale, seed = args
    rn.seed(seed)
    num_buyers = len(teams) - 1
    statistics = np.zeros(2*(num_buyers + 1))
    reserve_prices = rn.exponential(price_scale, size=num_buyers)
    reserve_prices = np.append(reserve_prices, -100.)
    # the team assigned with -100 means that this team should act as a seller in this round

    ##############
    ## Dynamics ##
    ##############
    for n in range(len(teams)):
        reserve_price_n = np.roll(reserve_prices, n+1)  # Assign reservation price in a round-robin way

        # reset initial values
        x_t = x_0
        x_h_t = [x_0]
        p_h_t = []

        b_t_1 = np.zeros(num_buyers + 1)  # b_t_1[n] indicates if buyer makes a purchase at time t-1
        b_h = np.zeros(num_buyers + 1)  # b_h[n] indicates the number of items the buyer purchases

        purchase_hist = [[] for _ in range(num_buyers+1)]

        # game starts
        for t in range(horizon):
            # seller determines price p_t at time t
            seller = teams[n][1]
            p_t = seller.get_price(t, x_h_t, p_h_t, price_scale, horizon, num_buyers)
            assert p_t >= 0., "Team {0} posted a negative price of {1}!".format(seller.get_name(), p_t)
            p_h_t.append(p_t)

            b_t = np.zeros(num_buyers+1)  # b_t[m] indicates if buyer m has willingness to buy at time t
            for m in range(num_buyers+1):
                if m != n and b_h[m] == 0:  # we should exclude team n since it plays as the seller in this round
                    buyer = teams[m][0]
                    b_t[m] = buyer.get_decision(t, x_h_t, p_h_t, reserve_price_n[m], b_t_1[m], horizon, num_buyers, purchase_hist[m])

            # randomly decide who make a purchase if inventory is not enough to
            # serve all demands in the current period

            if x_t < np.sum(b_t):
                buyer_indices = [i for i in range(len(teams)) if i != n and b_t[i] == 1]
                b_t = np.zeros(num_buyers+1)
                b_t[rn.choice(buyer_indices, int(x_t), replace=False)] = 1
            b_h += b_t

            statistics[n] += p_t * np.sum(b_t)
            b_t_1 = b_t
            statistics[len(teams):] += np.multiply(b_t, reserve_price_n - p_t)
            x_t -= sum(b_t)
            x_h_t.append(x_t)
            if x_t == 0:
                break
    return statistics


def simulate_samples(teams, horizon, x_0, num_trials, price_scale, num_threads):
    pool = mp.Pool(num_threads)
    try:
        args = zip([teams]*num_trials, [horizon]*num_trials, [x_0]*num_trials, [price_scale]*num_trials,
                   rn.randint(100, 10000000, size=num_trials))
        revenue_cs_sample = np.array(pool.map(simulate_trial, args)) if num_threads > 1 else \
            np.array(map(simulate_trial, args))
    finally:
        pool.close()
    return revenue_cs_sample


def simulate(teams, horizon, x_0, num_trials, price_scale, num_threads=1):
    """
    Computes the average revenue and consumer surplus of the N teams
    :param teams: a list [(b_i, s_i), i=1,...,N] of pairs (b_i, s_i) of buyer, seller pairs of each team
    :param horizon: length of game
    :param x_0: initial seller's inventory
    :param num_trials: number of Monte Carlo trials for estimating the mean revenue and consumer surplus
    :param price_scale: distributional parameter for the reserve price. More precisely the reserve prices are
                        i.i.d from an Exponential(1/price_scale) distribution, so that E[reserve_price] == price_scale
    :param num_threads: Number of parallel threads used to run simulation
    :return:
    """
    # team n's revenue yield when playing as the seller is stored in revenue[n]
    revenue_cs_sample = simulate_samples(teams, horizon, x_0, num_trials, price_scale, num_threads)
    # return performance results
    revenue_cs = np.mean(revenue_cs_sample, axis=0)
    return revenue_cs[:len(teams)], revenue_cs[len(teams):]
