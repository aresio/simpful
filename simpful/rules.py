import numpy as np
import operator
import regex as re
from .rule_parsing import preparse, postparse
import itertools
import random
from random import randint, randrange

from simpful import rules


def proba_generator(n):
    """
    Global method for generating a list of probabilities (uniformly distributed).

    Args:
        n ([integer]): the length of the list containing probabilities to be created.

    Returns:
        [list]: a list of probabilities.
    """
    list_of_random_floats = np.random.random(n)
    sum_of_values = list_of_random_floats.sum()
    normalized_values = list_of_random_floats / sum_of_values
    return normalized_values


def duplicate(testList, n):
    return [ele for ele in testList for _ in range(n)]


class RuleGen:
    """
    
    This module contains helper methods to construct rules for a fuzzy system
    in a random way. An example use case would be a search algorithm for a fuzzy
    rule base which tests to see what rules given a list of variables and linguistic terms 
    are best. 

    """

    def __init__(self, n_consequents, cluster_centers, all_var_names=None, threshold=None, var_names=None, probas=None,
                 generateprobas=None, operators=['AND_p', 'OR', 'AND', 'NOT'], ops=['AND_p', 'OR', 'AND'], var_len=False,
                 unique_vars=None):
        
        """ Contructor methods that should initialize at least the following:
            cluster_centers, n_consequents

                        
        Args:
            cluster_centers ([ndarray]): An ndarray which can be obtained for example using sklearn fuzzy cmeans
            all_var_names ([list], optional): A list of all the variables which can (possibly) be included in the rules. Defaults to None.
            threshold ([integer]): A threshold to determine how many variables should be included in the rules. Defaults to None.
            n_consequents ([list]): a list containing the outcomes that are possible (outcomes should be strings)
            var_names ([list], optional): Instead of giving all_var_names one could specify just specify ONLY the variables you want included. Defaults to None.
            probas ([ndarray], optional): Shape should be (n_rules, n_consequents). Defaults to None.

            Note: either all_var_names and a threshold should be given or just the var_names you want included.
        """

        self.threshold = threshold
        self.all_var_names = all_var_names
        self.cluster_centers = cluster_centers.shape[0] if isinstance(cluster_centers, (np.ndarray)) else cluster_centers
        self.var_names = var_names
        self.n_consequents = n_consequents
        self.p_rules = []
        self.probas = probas if probas is None else np.round(
            probas, decimals=3, out=None)
        self.generateprobas = generateprobas
        self.genprobas = None
        self.operators = operators
        self.ops = ops
        self.models = {}
        self.pfs = None
        self.var_len = var_len
        self.unique_vars = None if unique_vars is None else unique_vars
    
    def var_selector(self):
        """
        
        This method does the selection of variables given to be included in the rules given
        a list of variables.

        """
        winner = []
        full_dict = dict(
            zip(self.all_var_names, proba_generator(len(self.all_var_names))))
        while len(winner) < (self.threshold):
            winner.append(
                max(full_dict.items(), key=operator.itemgetter(1))[0])
            full_dict.pop(winner[-1])

        self.var_names = winner

    def interpret_consequents(self):
        """

        If a ndarray of probabilities was NOT given of shape (n_rules, n_consequents)
        then this method can generate probabilities randomly in a uniform way.

        
        """
        if self.probas is None:
            self.genprobas = proba_generator(len(self.n_consequents))

    def generate_rules(self):
        """
        This method can be used to generate rules for a Zero Order Takagi Sugeno System with a crisp output value
        as in Fialho(2016).
        Keep in mind that using this method the crisp values would still have to be set manually.

        Returns:
            n_rules: number of rules where the number is equivalent to the number of clusters.
        """
        RULES = []
        for i in range(self.cluster_centers):
            RULE = 'IF '
            for j, var_name in enumerate(self.var_names):
                if (j != len(self.var_names) - 1):
                    RULE += '({} IS cluster{}) AND_p '.format(var_name, i)
                else:
                    RULE += '({} IS cluster{}) THEN (OUTPUT IS fun{})'.format(var_name, i, i)
            RULES.append(RULE)
        self.rules = RULES
        return RULES

    def generate_multiple_ts(self):
        """
        Takes the probabilistic rules and generates rules for seperate zero order takagi sugeno systems
        that can be used to obtain estimates seperately using zero order takagi sugeno systems 
        and ordinary least squares to estimate the parameters. 

        """
        antecedents = map(preparse, self.p_rules)
        preparsed = [*antecedents]
        fixed_preparse = [['IF ' + ant + ' THEN (OUTPUT IS fun{})'.
                           format(i)] for i, ant in enumerate(preparsed)]
        consequents = map(postparse, self.p_rules)
        postparsed = [*consequents]
        just_probas = [item[0] for item in postparsed]
        converted_models = [list(itertools.product(rule, probas))
                            for rule, probas in zip(fixed_preparse, just_probas)]
        transposed_models = list(map(list, zip(*converted_models)))
        models = {i: rule for i, rule in enumerate(transposed_models)}
        self.models = models
        return models

    def generate_zero_pfs(self):
        """
        Takes the probabilistic rules and generates rules for seperate zero order takagi sugeno systems
        that can be used to obtain estimates seperately using zero order takagi sugeno systems 
        and ordinary least squares to estimate the parameters. 

        """
        antecedents = map(preparse, self.p_rules)
        preparsed = [*antecedents]
        fixed_preparse = [['IF ' + ant + ' THEN P(OUTPUT IS SOMETHING)='.
                           format(i)] for i, ant in enumerate(preparsed)]
        consequents = map(postparse, self.p_rules)
        postparsed = [*consequents]

        if postparsed[0][1] == True and postparsed[0][0] > 0:
            numb_consequent = postparsed[0][0]
            holder = [None]*numb_consequent
            just_probas = duplicate(holder, len(preparsed))
            just_probas = [just_probas[i:i+numb_consequent]
                           for i in range(0, len(just_probas), numb_consequent)]

        else:
            just_probas = [item for item in postparsed]

        converted_models = [list(itertools.product(rule, probas))
                            for rule, probas in zip(fixed_preparse, just_probas)]
        transposed_models = list(map(list, zip(*converted_models)))
        models = {i: rule for i, rule in enumerate(transposed_models)}
        models2 = {}
        for model, rules in models.items():
            temp_rules = []
            for i, rule in enumerate(rules):
                new_rule = ''.join(map(str, rule))
                temp_rules.append(new_rule)
            models2[model] = temp_rules
        return models2

    def get_ts_probas(self):
        zero_order_probas = {}
        for model, rules in self.models.items():
            holder = []
            for rule in rules:
                holder.append(rule[1])
            zero_order_probas[model] = np.asarray(holder)
        return zero_order_probas

    def generate_operator(self):
        """
        Automatically generates the operators that Simpful supports. This method should
        be updated if changes are implemented in Simpful.

        Returns:
            [operator]: normally returns one operator. Except when NOT is returned,
            in which case 2 operators are returned. NOT is then applied to the next 
            variable and a connecting operator is added in addition to NOT. Keep in mind
            that NOT can't be returned twice.
        """
        winner = []
        ops_probas = dict(
            zip(self.operators, proba_generator(len(self.operators))))
        winner.append(max(ops_probas.items(), key=operator.itemgetter(1))[0])

        # Ensures Not can't be returned twice.
        if winner[0] == 'NOT':
            ops_probs = dict(zip(self.ops, proba_generator(len(self.ops))))
            winner.append(
                max(ops_probs.items(), key=operator.itemgetter(1))[0])

        return winner

    def generate_proba_rules(self, select = False):
        """
        Generates PFS rules randomly. It uses the operators as defined in the contructor (__init__) and thus 
        can easily be updated. The end result will be of shape (n_clusters, n_outcomes). 
        In laymans terms: this method will generate a rule for every cluster partition. The probabilities will 
        sum up to one and are either given or they can be randomly generated (uniformly distributed).
        """
        if select is True:
            self.var_selector()

        RULES = []

        for i in range(self.cluster_centers):
            RULE = 'IF '
            probas_for_rule = list(proba_generator(len(self.var_names)))
            indexed = list(enumerate(probas_for_rule))
            vars_to_be_incl = randint(1, len(self.var_names))
            top_vars = sorted(indexed, key=operator.itemgetter(1))[-vars_to_be_incl:]
            incl_var_indexes = [i[0] for i in top_vars]
            sorted_incl_indexes = sorted(incl_var_indexes)


            for j, var_name in enumerate(self.var_names):

                if j in sorted_incl_indexes:
                    pass
                else:
                    continue

                _operator = self.generate_operator()
                

                if j == sorted_incl_indexes[-1]:
                    # last part of rule

                    if _operator[0] == 'NOT':

                        RULE += '({} ({} IS cluster{}))'.format(
                        _operator[0], var_name, i)
                    
                    else:

                        RULE += '({} IS cluster{})'.format(var_name, i)

                    continue

                    # when not is selected execute this part with an additional operator is executed
                if _operator[0] == 'NOT':

                    RULE += '({} ({} IS cluster{})) {} '.format(
                        _operator[0], var_name, i, _operator[1])

                else:

                    # normal execution (meaning not was not choses as operator)
                    RULE += '({} IS cluster{}) {} '.format(var_name,
                                                           i, _operator[0])

            for k in range(len(self.n_consequents)):

                if self.generateprobas is True:

                    self.interpret_consequents()

                    if k == 0:

                        # first part of end of rule
                        RULE += ' THEN P(OUTCOME IS {})={}'.format(
                            self.n_consequents[k], self.genprobas[k])
                    else:

                        # if problem is multiclass
                        RULE += ', P(OUTCOME IS {})={}'.format(
                            self.n_consequents[k], self.genprobas[k])
                else:

                    # usefull for estiamting probabilties
                    if self.probas is None:

                        if self.generateprobas is False:

                            if k == 0:

                                RULE += ' THEN P(OUTCOME IS {})=None'.format(
                                    self.n_consequents[k])

                            else:

                                RULE += ', P(OUTCOME IS {})=None'.format(
                                    self.n_consequents[k])
                    else:

                        if self.probas is None:
                            raise Exception("Error: No probabilities were given. If you want to generate probabilities"
                                            + " randomly automatically try setting the generate probabilities parameter to true. ")

                        # if generation of probabilities is set to on (generate probas automatically)
                        if k == 0:

                            RULE += ' THEN P(OUTCOME IS {})={}'.format(
                                self.n_consequents[k], self.probas[i][k])
                        else:

                            RULE += ', P(OUTCOME IS {})={}'.format(
                                self.n_consequents[k], self.probas[i][k])

            RULES.append(RULE)

        # for clustering    
        antecedents_mut = [preparse(ant) for ant in RULES]
        var_check_mut = [re.findall(r"\w+(?=\sIS)", ant)
                            for ant in antecedents_mut]

        flatten = [
            item for sublist in var_check_mut for item in sublist]
        unique = list(set(flatten))
        self.unique_vars = unique

        self.p_rules = RULES
        return RULES
