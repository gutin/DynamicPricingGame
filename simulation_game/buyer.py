from abc import ABCMeta
from abc import abstractmethod

import signal
import time

EXECUTION_TIME_LIMIT_IN_SECONDS = 2


def signal_handler(signum, frame):
    raise Exception("Execution timed out!")


class Buyer(object):
    """
    Abstract class of a simulation game buyer. Defines a common interface for every team
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.time_out = False

    def get_decision(self, t, inventory_h, price_h, reserve_price, b_t_1_n, horizon, num_buyers, purchase_history):
        """
        Queries the buyer for its willingness to buy the product given the information below
        :param t: the current time
        :param inventory_h: the vector (x_0, X_1, ..., X_{t-1}) of the past evolution of the seller's inventory
        :param price_h: the vector (p_1, ..., p_{t-1}) of the past evolution of the seller's prices
        :param reserve_price: this buyer's current reservation price
        :param b_t_1_n: binary flag for whether the seller bought in the previous time step
        :param horizon: the 'T' parameter
        :param num_buyers: number of buyers in every round
        :param purchase_history:
        :return: 1 if this buyer intends to buy the product now, else 0
        """
        purchase_history.append(b_t_1_n)
        if sum(purchase_history) >= 1:
            # if the buyer bought one item already, then we prevent him from doing so again
            return 0

        # Limit execution to 1 second!
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(EXECUTION_TIME_LIMIT_IN_SECONDS)
        try:
            return self._get_decision_impl(t, inventory_h, price_h, reserve_price, b_t_1_n, horizon, num_buyers)
        except Exception as e:
            print "Exception caught for buyer", self.get_name()
            self.time_out = True
            return 0

    @abstractmethod
    def _get_decision_impl(self, t, inventory_h, price_h, reserve_price, b_t_1_n, horizon, num_buyers):
        """
        Implementation of the get_decision method which takes the same parameters.
        Encapsulates the logic behind buying decisions. Because this is an abstract method, it returns None.
        """
        return

    @abstractmethod
    def get_name(self):
        """
        Returns the name for this buyer. Helps in analyzing results
        """
        return


class MyopicBuyer(Buyer):

    """
    An example buyer bot for the purposes of testing the game
    """

    def get_name(self):
        return "MyopicBuyer(" + self.name + ")"

    def __init__(self, name):
        super(MyopicBuyer, self).__init__()
        self.name = name

    def _get_decision_impl(self, t, inventory_h, price_h, reserve_price, b_t_1_n, horizon, num_buyers):
        """
        This simple buyer bot will always buy as long as the seller's posted price is below its reserve price
        """
        return 1 if reserve_price > price_h[t] else 0


class FaultyBuyer(Buyer):
    """
    A bad example of a buyer. This is to test that we do stop long running bots
    """
    def get_name(self):
        return "FaultyBuyer"

    def _get_decision_impl(self, t, inventory_h, price_h, reserve_price, b_t_1_n, horizon, num_buyers):
        while True:
            time.sleep(5)
        return 1
