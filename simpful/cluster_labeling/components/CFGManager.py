import random
import re
from typing import List

try:
    from nltk import CFG, parse, Nonterminal
    from nltk.parse.generate import generate

    nltk = True
except ImportError:
    nltk = False

# Type alias
HedgeChain = List[str]


class CFGManager:

    def __init__(self, max_generation_depth: int = None, template_names: List[str] = None):
        if not nltk:
            raise Exception("ERROR: please, install nltk for cluster labeling facilities")
        if template_names is None:
            self.template_names = ["low", "medium", "high"]
        # Define a context-free grammar (CFG) for generating propositions
        self.grammar = CFG.fromstring("""
        PROPOSITION -> MOD_PROP | SHIFT_PROP | SHIFT_PROP MOD_PROP 

        MOD_PROP -> MODIFIER | STACK_MOD MODIFIER 
        MODIFIER -> CONCENTRATE_MOD | DILATE_MOD 

        SHIFT_PROP -> SHIFT | STACK_MOD SHIFT
        SHIFT -> SHIFT_ABOVE | SHIFT_BELOW

        SHIFT_ABOVE -> 'above'
        SHIFT_BELOW -> 'below'

        STACK_MOD -> 'very' | 'very_very' 

        CONCENTRATE_MOD -> 'absolutely' | 'extremely' | 'definitely'
        DILATE_MOD -> 'approximatively' | 'slightly' | 'more_or_less' | 'sort_of' | 'fairly' | 'somewhat' | 'likely'
        """)
        self.parser = parse.BottomUpChartParser(self.grammar)
        # Precompute a list of sentences
        self.precomputed_individuals = self.generate_sentences(max_generation_depth)

    def generate_sentences(self, max_depth: int = None) -> List[HedgeChain]:
        """
        Generate a list of sentences from the grammar up to a specified depth

        :param max_depth: max depth, unlimited if None
        :return: generated sentences
        """
        sentences: List[HedgeChain] = list(generate(self.grammar, depth=max_depth))
        return sentences

    def generate_individuals(self, n: int = 1) -> List[HedgeChain]:
        """
        Randomly sample 'n' individuals from the precomputed list of sentences

        :param n: sampled individuals
        :return: n individuals
        """
        return random.sample(self.precomputed_individuals, k=n)

    def get_negative_shift(self) -> List[str]:
        """
        Retrieve the right-hand side (RHS) elements of the grammar rules for SHIFT_BELOW (negative shift)
        In the simplest case, just 'below'

        :return: rhs for negative shift
        """
        return [x._rhs[0] for x in self.grammar._lhs_index[Nonterminal("SHIFT_BELOW")]]

    def get_positive_shift(self) -> List[str]:
        """
        Retrieve the right-hand side (RHS) elements of the grammar rules for SHIFT_ABOVE (positive shift)
        In the simplest case, just 'above'

        :return: rhs for positive shift
        """
        return [x._rhs[0] for x in self.grammar._lhs_index[Nonterminal("SHIFT_ABOVE")]]

    def get_priority_terms(self, term: str):
        """
        Get the priority terms (i.e., terms to maintain when simplifying labels)

        :param term: TODO
        :return: TODO
        """
        above_shifts: List[str] = self.get_positive_shift()
        negative_shifts: List[str] = self.get_negative_shift()
        # TODO: fix
        cluster_names = re.findall(r"\(cluster\d+\)", term)
        return above_shifts + negative_shifts + self.template_names + cluster_names  # + self.get_uncertainty_mods() + cluster_names

    # def get_uncertainty_mods(self):
    #     return [x._rhs[0] for x in self.grammar._lhs_index[Nonterminal("PEAK_MOD")]]
    #
    # def get_empty_hedge(self):
    #     return self.grammar._lhs_index[Nonterminal("EMPTY_HEDGE")][0]._rhs[0]

    # NEGATION_HEDGE: str = 'not'
    # @classmethod
    # def get_negation_hedge(cls):
    #     return cls.NEGATION_HEDGE
