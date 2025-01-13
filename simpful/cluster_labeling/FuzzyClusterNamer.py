from __future__ import annotations

from datetime import datetime
from typing import List

import seaborn as sns

from simpful import FuzzySet, SingletonsSet, LinguisticVariable
from simpful.cluster_labeling.components import CFGManager, GaussianFuzzyTemplates, GrammarGuidedHedgeBuilder


class FuzzyClusterNamer:
    def __init__(self, FS_list: List[FuzzySet] | LinguisticVariable,
                 universe_of_discourse: List[float]):
        # force_different: bool = False):
        """
        Performs fuzzy cluster naming

        :param FS_list: List of fuzzy sets (clusters pertaining a certain feature) to name.
                        Can be a list of fuzzy sets or a linguistic variable that contains them.
        :param universe_of_discourse: Boundaries of the known universe of discourse
        """
        if isinstance(FS_list, LinguisticVariable):
            self._FS_list: list[FuzzySet] = FS_list._FSlist
        else:
            self._FS_list: list[FuzzySet] = FS_list
        self._universe_of_discourse = universe_of_discourse
        # self.force_different = force_different  #

        self._params: dict = {
            "num_individuals": -1,
            "minimum_k_candidates": 10,
            "accept_similarity_threshold": 0.8,
            "simplify_names": True
        }
        self._simplify: bool = self._params["simplify_names"]
        self._cfg: CFGManager = CFGManager()

    @staticmethod
    def _fix_term(fuzzy_set: FuzzySet) -> None:
        """
        Fix the nomenclature of the term of a fuzzy set

        :param fuzzy_set: Fuzzy set whose term must be fixed
        :return: None
        """
        # Term (approximation)
        term = fuzzy_set._term[:fuzzy_set._term.find("(")].strip()
        # Cluster (just brackets that indicate which cluster the approximation refers to)
        cluster = fuzzy_set._term[fuzzy_set._term.find("("):]
        # Find the possible "surely" string
        # TODO: rename, or remove
        if not (index := term.find(GaussianFuzzyTemplates.SURELY_STRING)) == -1:
            new_term = (f"{term[index + len(GaussianFuzzyTemplates.SURELY_STRING):]} "
                        f"and {term[:index]}{cluster}").strip()

            fuzzy_set._term = new_term
        # Fix underscores
        fuzzy_set._term = fuzzy_set._term.replace("_", " ")

    def _simplify_terms(self, approximated_sets: list[FuzzySet]) -> list[FuzzySet]:
        """
        Simplify the terms of an approximated fuzzy set.
        Only priority terms as defined in the ContextFreeGrammarManager are kept.

        :param approximated_sets: sets to simplify
        :return: approximated sets with simplified terms
        """
        for approximated_set in approximated_sets:
            term: str = approximated_set.get_term()
            priority_list: list[str] = self._cfg.get_priority_terms(term)
            term_tokens: list[str] = term.split(" ")
            simplified_term: list[str] = [x for x in term_tokens if x in priority_list]
            approximated_set._term = " ".join(simplified_term).strip()
        return approximated_sets

    def run_approximator(self, **plot_args) -> list[FuzzySet]:
        """
        Run the cluster name approximators

        :param plot_args: optional parameters for plotting
        :return: a list of approximated sets
        """

        # --------------------------------------------------------
        if any([isinstance(f, SingletonsSet) for f in self._FS_list]):
            raise ValueError("Currently cannot approximate singleton sets")
        # --------------------------------------------------------
        plot: bool = plot_args.get("plot", False)
        if plot:
            sns.set_style("whitegrid")
            LinguisticVariable(self._FS_list, universe_of_discourse=self._universe_of_discourse).plot(
                outputfile=plot_args.get("name_before", f"old_{datetime.now().strftime('%H_%M_%S')}"))
        templates = GaussianFuzzyTemplates(universe_of_discourse=self._universe_of_discourse)
        # ------------------------------------------------------------------------------
        approximated_sets: list[FuzzySet] = []
        for f_set in self._FS_list:
            # Get two best templates
            best_templates: list[FuzzySet] = templates.match_likely_candidates(f_set)
            # Run grammar guided hedge builder
            hedge_builder = GrammarGuidedHedgeBuilder(sets_to_modify=best_templates, set_to_approximate=f_set,
                                                      universe_of_discourse=self._universe_of_discourse,
                                                      use_tqdm=plot_args.get("use_tqdm", False))
            simil_set = hedge_builder.run_complete(num_individuals=self._params["num_individuals"],
                                                   accept_similarity_threshold=self._params[
                                                       "accept_similarity_threshold"],
                                                   minimum_k_candidates=self._params["minimum_k_candidates"])
            # Force different ?
            self._fix_term(simil_set)
            approximated_sets.append(simil_set)

        if plot:
            LinguisticVariable(self._FS_list + approximated_sets,
                               universe_of_discourse=self._universe_of_discourse).plot(
                outputfile=plot_args["name_after"])
        if self._simplify:
            approximated_sets = self._simplify_terms(approximated_sets)
        return approximated_sets
