from typing import List

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x  # Fallback to a no-op

from simpful import FuzzySet
from simpful.cluster_labeling.components.CFGManager import CFGManager, HedgeChain
from simpful.cluster_labeling.components.HedgeApplier import HedgeApplier
from simpful.cluster_labeling.components.Individual import Individual
from simpful.cluster_labeling.components.similarity import union_intersection_sim


class GrammarGuidedHedgeBuilder:
    def __init__(self, sets_to_modify: List[FuzzySet],
                 set_to_approximate: FuzzySet,
                 universe_of_discourse: list,
                 use_tqdm: bool = False):
        # --- Start and target set ---
        self.sets_to_modify: List[FuzzySet] = sets_to_modify
        self.set_to_approximate: FuzzySet = set_to_approximate
        # --- Useful to calculate shifts ---
        self.universe_of_discourse: list = universe_of_discourse
        # --- Context Free Grammar ---
        self.cfg = CFGManager()
        # --- Hedge applier class object ---
        self.hedge_applier = HedgeApplier(universe_of_discourse=self.universe_of_discourse,
                                          cfg=self.cfg)
        # --- Parameters ---
        self.params: dict = {
            # Hedge generation parameters
            "required_improvement": 0.1,
            "compromise_criteria": "shortest",
        }
        self._tqdm = use_tqdm

    def __generate_hedge_population(self, num_individuals: int = -1) -> list:
        """
        Generate a population of hedges

        :param num_individuals: int Number of individual hedges. -1 uses all possible hedges
        :return: list[list[str]] population of hedges
        """
        max_invidiuals: int = len(self.cfg.precomputed_individuals)
        # All (or clamp)
        if num_individuals == -1 or num_individuals > max_invidiuals:
            num_individuals = max_invidiuals
        if num_individuals <= 0:
            raise ValueError(f"Found {num_individuals} num_individuals: must be either -1 or a positive value")
        population: list = self.cfg.generate_individuals(n=num_individuals)
        return population

    def run_naive_best(self, num_individuals: int = -1) -> FuzzySet:
        """
        Runs naive approach (just pick the best overall without modifications)

        :param num_individuals: number of individuals generated (-1: all in grammar)
        :return: most similar hedge chain
        """
        # Generate hedges
        population: List[HedgeChain] = self.__generate_hedge_population(num_individuals)
        chain_population = self.__hedge_population_fitness(population)
        elite_candidates: List[Individual] = self.__pick_top_candidates(chain_population)
        return elite_candidates[0].current_fuzzy_set

    def __hedge_population_fitness(self, hedge_population: List[HedgeChain]) -> List[Individual]:
        """
        Calculate the fitness of a hedge population

        :param hedge_population: A list containing possible hedges
        :return: a list of individuals
        """
        individuals_with_fitness = []
        for individual_hedge in tqdm(hedge_population, disable=not self._tqdm):
            individuals_with_fitness += self.__similarity_fitness(individual_hedge)
        return individuals_with_fitness

    def __similarity_fitness(self, hedge_chain: HedgeChain) -> List[Individual]:
        """
        Calculate the similarity (fitness) of a hedge chain to the target set.
        Hedge chains are applied to all current template candidates (in sets_to_modify).

        :param hedge_chain: hedge chain to test
        :return:
        """
        modified_templates = []
        # Iterate over templates
        for template in self.sets_to_modify:
            # Generate individual with that template
            individual = Individual(target_set=self.set_to_approximate,
                                    template_set=template,
                                    universe_of_discourse=self.universe_of_discourse,
                                    hedge_applier=self.hedge_applier)
            # Apply hedge chain to the individual
            individual.apply_hedges(hedge_chain)
            modified_templates.append(individual)
        return modified_templates

    @staticmethod
    def __pick_top_candidates(population: List[Individual], k: int = 5) -> List[Individual]:
        #  Select the top best candidates based on the aptitude of individuals.
        return sorted(population, key=lambda x: x.current_fitness, reverse=True)[:k]

    def run_conservative(self, num_individuals: int = -1,
                         accept_similarity_threshold: float = 0.8,
                         minimum_k_candidates: int = 10) -> FuzzySet:
        """
        Run conservative approach. Best hedges in terms of fitness with a compromise (e.g., shorter are preferred)

        :param num_individuals: number of individuals generated (-1: all in grammar)
        :param accept_similarity_threshold: similarity threshold after which an approximation is accepted
        :param minimum_k_candidates: minimum candidates that satisfy the threshold required move to the next step
                                     If not satisfied, the threshold is lowered
        :return: best candidate fuzzy set
        """
        # Generate hedges
        population: List[HedgeChain] = self.__generate_hedge_population(num_individuals)
        individuals_with_fitness: List[Individual] = self.__hedge_population_fitness(population)
        # Sort individuals by their fitness (descending order, higher first)
        sorted_best_candidates = [x for x in sorted(individuals_with_fitness,
                                                    key=lambda x: x.fitness, reverse=True)]

        while True:
            # Filter by threshold
            filtered_candidates = [x for x in
                                   sorted_best_candidates
                                   if x.fitness > accept_similarity_threshold]

            # If the list is too small, try again with lower thresholding
            if len(filtered_candidates) < minimum_k_candidates:
                accept_similarity_threshold -= 0.05
            # Otherwise, find the best candidate in the list
            else:
                best_candidate = self.__find_best_compromise(filtered_candidates)
                return best_candidate.current_fuzzy_set

    def __find_best_compromise(self, sorted_best_candidates: List[Individual]) -> Individual:
        """
        Find the best compromise hedge chain following a criteria.
        - "Shortest" - Sort by chain length. Compromise in length and fitness.
                       A longer one is chosen over a shorter one only if it has a required_improvement improvement

        :param sorted_best_candidates: Good candidates sorted in descending order of fitness
        :param required_improvement: Required fitness improvement for a hedge chain to be chosen
                                     over the compromise criteria (e.g., a longer chain when criteria is shortest)
        :return:
        """
        # TODO: Add different criteria?
        if self.params["compromise_criteria"] == "shortest":
            # Sort by length
            sorted_best_candidates.sort(key=lambda x: x.current_fuzzy_set.get_term())
            # The first best candidate is the shortest
            best_candidate = sorted_best_candidates[0]
            # Go through other good candidates, accept if there is a substantial improvement
            for candidate in sorted_best_candidates[1:]:
                # Not better, skip
                if candidate.fitness < best_candidate.fitness:
                    continue
                # Improvement by at least required_improvement
                elif (candidate.fitness - best_candidate.fitness) > self.params["required_improvement"]:
                    best_candidate = candidate
            return best_candidate
        else:
            raise ValueError("compromise_criteria must be in [shortest,]")

    def run_complete(self, num_individuals: int = -1,
                     accept_similarity_threshold: float = 0.8,
                     minimum_k_candidates: int = 10) -> FuzzySet:
        """
        Run complete approach. Like conservative, but keeps template as is if it already satisfied similarity threshold

        :param num_individuals: number of individuals generated (-1: all in grammar)
        :param accept_similarity_threshold: similarity threshold after which an approximation is accepted
        :param minimum_k_candidates: minimum candidates that satisfy the threshold required move to the next step
                                     If not satisfied, the threshold is lowered
        :return: best candidate fuzzy set
        """
        # TODO: not return first
        for basic_template in self.sets_to_modify:
            sim = union_intersection_sim(basic_template, self.set_to_approximate, self.universe_of_discourse)
            # If base template is already good enough
            if sim >= accept_similarity_threshold:
                accepted_template = FuzzySet(function=lambda x: basic_template.get_value(x),
                                             term=f"{basic_template.get_term()}")
                # ({self.set_to_approximate.get_term()})")
                return accepted_template
        # Now run conservative
        return self.run_conservative(num_individuals, accept_similarity_threshold, minimum_k_candidates)
