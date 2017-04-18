from abc import ABCMeta
from abc import abstractmethod

import signal
import sys
import traceback

EXECUTION_TIME_LIMIT_IN_SECONDS = 2


def signal_handler(signum, frame):
    raise Exception("Execution timed out!")


class Seller(object):
    """
    Abstract class of a simulation game seller. Defines a common interface for every team
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.time_out = False

    def get_price(self, t, inventory_h, price_h, price_scale, horizon, num_buyers):
        # Limit execution time of the pricing function to 1 second
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(EXECUTION_TIME_LIMIT_IN_SECONDS)
        try:
            return self._get_price_impl(t, inventory_h, price_h, price_scale, horizon, num_buyers)
        except Exception as e:
            print "Exception thrown in seller {0}!".format(self.get_name())
            print traceback.print_exc(file=sys.stdout)
            self.time_out = True
            return 1e7

    @abstractmethod
    def _get_price_impl(self, t, inventory_h, price_h, price_scale, horizon, num_buyers):
        """
        Queries the seller for the price it's going to post for the current time step
        :param t: current time step
        :param inventory_h: the vector (x_0, X_1, ..., X_{t-1}) of the past evolution of the seller's inventory
        :param price_h: the vector (p_1, ..., p_{t-1}) of the past evolution of the seller's prices
        :param price_scale: the parameter of the Exponential distribution of reserve prices. Check numpy.random
                            documentation for the exact definition. Note that the mean of this distribution equals price_scale
        :param horizon: length of a game
        :param num_buyers: number of buyers
        :return: the price the seller posts now
        """
        return

    @abstractmethod
    def get_name(self):
        """
        Returns the name of the seller. Helps in analyzing results
        """
        return


class DummySeller(Seller):
    """
    A dummy seller bot for the purposes of testing the game
    """
    def get_name(self):
        return "DummySeller(" + self.name + ")"

    def __init__(self, name, constant_price=1.):
        self.name = name
        self.constant_price = constant_price

    def _get_price_impl(self, t, inventory_h, price_h, price_scale, horizon, num_buyers):
        """
        This seller always posts a fixed price regardless of anything
        """
        return self.constant_price
