import random
import re
from typing import List

try:
    from nltk import CFG, parse, Nonterminal
    from nltk.parse.generate import generate

    nltk = True
except ImportError:
    nltk = False

# Type aliases
HedgeChain = List[str]


class CFGManager:
    NEGATION_HEDGE: str = 'not'

    def __init__(self, **kwargs):
        if not nltk:
            raise Exception("ERROR: please, install nltk for cluster labeling facilities")
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
        self.precomputed_individuals = self.generate_sentences(**kwargs)

    def generate_sentences(self, max_depth: int = None):
        sentences = list(generate(self.grammar, depth=max_depth))
        return sentences

    def generate_individuals(self, n: int = 1):
        return random.sample(self.precomputed_individuals, k=n)

    def get_negative_shift(self):
        return [x._rhs[0] for x in self.grammar._lhs_index[Nonterminal("SHIFT_BELOW")]]

    def get_positive_shift(self):
        return [x._rhs[0] for x in self.grammar._lhs_index[Nonterminal("SHIFT_ABOVE")]]

    # def get_uncertainty_mods(self):
    #     return [x._rhs[0] for x in self.grammar._lhs_index[Nonterminal("PEAK_MOD")]]
    #
    # def get_empty_hedge(self):
    #     return self.grammar._lhs_index[Nonterminal("EMPTY_HEDGE")][0]._rhs[0]

    @classmethod
    def get_negation_hedge(cls):
        return cls.NEGATION_HEDGE
    def get_priority_terms(self, term: str = ""):
        above_shifts = self.get_positive_shift()
        negative_shifts = self.get_negative_shift()
        # uncertainty = self.get_uncertainty_mods()
        # TODO: less hard-coded? From file?
        templates = ["low", "medium", "high"]
        cluster_names = re.findall(r"\(cluster\d+\)", term)
        return above_shifts + negative_shifts + templates + cluster_names  # + uncertainty + cluster_names



