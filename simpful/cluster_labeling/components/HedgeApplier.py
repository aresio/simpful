from configparser import ConfigParser
from typing import Callable, List, Tuple, Dict, Any

from simpful.cluster_labeling.components.CFGManager import CFGManager, HedgeChain


def as_dict(cfg_parser: ConfigParser) -> dict:
    d: Dict[str, Any] = dict(cfg_parser._sections)
    for k in d:
        for subk in d[k]:
            d[k][subk] = float(d[k][subk])
    return d


def load_ini_to_dict(file_path: str) -> dict:
    config = ConfigParser()
    config.read(file_path)
    return as_dict(config)


class HedgeApplier:
    def __init__(self, universe_of_discourse: List[float],
                 cfg: CFGManager):
        self.universe_of_discourse: List[float] = universe_of_discourse
        # Extent is useful to scale the transformation to any order of magnitude
        low, high = universe_of_discourse
        self.__extent = abs(high - low)
        # Grammar, hedges
        self.cfg: CFGManager = cfg
        self.hedges_dict: dict = load_ini_to_dict('simpful/cluster_labeling/_hedges.ini')

    def __unary_transform(self, mem_fun: Callable, q: float, r: float, p: float) -> Callable:
        """
        Hedge application formula

        :param mem_fun: membership function being modified
        :param q: Deflates function (modifies peak of gaussian up and down)
        :param r: Shifts funciton (left/right)
        :param p: Makes function larger or thinner
        :return: modified membership function
        """
        return lambda x: q * (mem_fun(x - (self.__extent * r)) ** p)

    def apply_single_hedge(self, mem_fun: Callable,
                           hedge_to_apply: str,
                           current_hedge: str,
                           shift_enabled: bool,
                           shift_mult: float):
        """
        Apply a single hedge to a membership function

        :param mem_fun: membership function to modify
        :param hedge_to_apply: hedge to apply
        :param current_hedge: current hedge (before modification)
        :param shift_enabled: whether to utilize the shift portion of the hedge (if moving in positive or negative)
        :param shift_mult: [-1, 1] whether to move [left, right]
        :return: modified hedge chain, modified membership function
        """
        # Fetch values
        # q - Deflates function (not used much)
        # r - Shift quantity
        # p - Exponent (makes thinner/larger)
        q, r, p = self.hedges_dict[hedge_to_apply].values()
        r = shift_mult * r if shift_enabled else 0.0
        p = p if not shift_enabled else 1.0
        current_hedge = f"{hedge_to_apply} {current_hedge}"
        return current_hedge, self.__unary_transform(mem_fun=mem_fun, q=q, r=r, p=p)

    def apply_hedge_chain(self, membership_function: Callable, hedge_chain: HedgeChain) -> Tuple[str, Callable]:
        """
        Apply a hedge chain to a membership function

        :param membership_function: membership function to modify
        :param hedge_chain: hedge chain to apply
        :return: current hedge chain, modified membership function
        """
        current_hedge: str = ""
        current_membership: Callable = membership_function
        shift_enabled: bool = False  # To enable shift modifications for stack modifiers
        shift_mult: float = 1.0  # To move in the negative direction

        # --- APPLY HEDGES in reverse order (last to first, skip if empty) ---
        # ([hedge for hedge in hedge_chain if hedge != self.cfg.get_empty_hedge()]):
        for hedge in reversed(hedge_chain):
            # Hedges are always [stack] shift [stack] modifier
            # This allows [stack] before the shift to enhance the shift

            # If the hedge is a shift of any kind
            if hedge in self.cfg.get_positive_shift() + self.cfg.get_negative_shift():
                shift_enabled: bool = True  # Enable
            # Apply the current hedge
            current_hedge, current_membership = self.apply_single_hedge(mem_fun=current_membership,
                                                                        hedge_to_apply=hedge,
                                                                        current_hedge=current_hedge,
                                                                        shift_enabled=shift_enabled,
                                                                        shift_mult=shift_mult)
            # This makes the [stack] before negative shifts help to move to the left
            if hedge in self.cfg.get_negative_shift():
                shift_mult: float = -1.0
        return current_hedge, current_membership
