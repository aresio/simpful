from typing import List, Tuple

from simpful import FuzzySet
from simpful.cluster_labeling.components.CFGManager import HedgeChain
from simpful.cluster_labeling.components.HedgeApplier import HedgeApplier
from simpful.cluster_labeling.components.similarity import union_intersection_sim


class Individual:

    def __init__(self, target_set: FuzzySet,
                 template_set: FuzzySet,
                 universe_of_discourse: List[float],
                 hedge_applier: HedgeApplier):
        self.universe_of_discourse: List[float] = universe_of_discourse
        self.target_set: FuzzySet = target_set
        self.base_template: FuzzySet = template_set
        self.current_fuzzy_set: FuzzySet = template_set
        self.fitness: float = 0.0
        self.current_chain: HedgeChain = []

        self.hedge_applier: HedgeApplier = hedge_applier

    def apply_hedges(self, hedge_chain: HedgeChain):
        """
        Apply a hedge chain (list of strings) to the current individual

        :param hedge_chain: (list of strings) hedges to apply
        :return: None
        """
        # The new hedges are prepended (new additions are extended with already present hedges)
        # Extend is also in place, so hedge_chain is also updated
        self.current_chain = hedge_chain.extend(self.current_chain) if self.current_chain else hedge_chain
        # Apply hedge
        current_hedge, modified_membership = self.hedge_applier.apply_hedge_chain(
            membership_function=self.base_template._funpointer,
            hedge_chain=hedge_chain)
        # Modify fuzzy set
        self.current_fuzzy_set = FuzzySet(function=modified_membership,
                                          term=f"{current_hedge}"
                                               f"{self.base_template.get_term()}")
                                               # f"({self.target_set.get_term()})")
        # Update fitness
        self.fitness = union_intersection_sim(self.current_fuzzy_set,
                                              self.target_set,
                                              self.universe_of_discourse)

    def __repr__(self):
        return f"Invidual " \
               f"[f: {self.fitness}] " \
               f"{self.current_chain} {self.current_fuzzy_set.get_term()}"
