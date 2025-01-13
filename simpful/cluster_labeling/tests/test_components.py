import simpful as sf
from simpful.cluster_labeling.components.CFGManager import CFGManager
from simpful.cluster_labeling.components.FuzzyTemplates import GaussianFuzzyTemplates
from simpful.cluster_labeling.components.GrammarGuidedHedgeBuilder import GrammarGuidedHedgeBuilder
from simpful.cluster_labeling.components.HedgeApplier import HedgeApplier
from simpful.cluster_labeling.components.Individual import Individual
from simpful.cluster_labeling.components.similarity import union_intersection_sim

try:
    from nltk import CFG, parse, Nonterminal, ChartParser
    from nltk.parse.generate import generate

    nltk = True
except ImportError:
    nltk = False


def test_CFG_manager():
    cfg = CFGManager()
    # Create a parser with your grammar
    # for c in cfg.precomputed_individuals:
    #     print(c)
    parser = ChartParser(cfg.grammar)
    #
    # # Parse a sentence
    sentence = ['below', 'very', 'slightly']
    for tree in parser.parse(sentence):
        tree.pretty_print()  # Print the parse tree
        # tree.draw()  # Draw the parse tree
    # editor.mainloop()
    cfg = CFGManager()
    for i in cfg.precomputed_individuals:
        print(i)
    print(len(cfg.precomputed_individuals))


def test_fuzzy_templates():
    GaussianFuzzyTemplates(universe_of_discourse=[-10., 10.]).show()
    GaussianFuzzyTemplates(universe_of_discourse=[50, 100]).show()
    GaussianFuzzyTemplates(universe_of_discourse=[801.0, 1145.0]).show()
    GaussianFuzzyTemplates(universe_of_discourse=[-10000, 50]).show()
    GaussianFuzzyTemplates(universe_of_discourse=[40, 700]).show()
    GaussianFuzzyTemplates(universe_of_discourse=[0, 1]).show()
    GaussianFuzzyTemplates(universe_of_discourse=[-100000, 1000000]).show()


def test_grammar_guided_builder():
    gs = sf.InvGaussianFuzzySet(7, 2, "C_1")
    templates = GaussianFuzzyTemplates([0, 10])._template_list
    gg = GrammarGuidedHedgeBuilder(templates, gs, [0, 10], )

    best = gg.run_complete()

    sf.LinguisticVariable([gs, best],
                          universe_of_discourse=[0, 10]).plot()


def test_individual():
    print(" *** TESTING INDIVIDUAL ***")

    cfg = CFGManager()
    templates = GaussianFuzzyTemplates([0, 10])._template_list
    G_2 = sf.GaussianFuzzySet(mu=7, sigma=2, term="real")

    hedge_applier = HedgeApplier([0, 10], cfg)
    for G_1 in templates[1:2]:
        ind = Individual(target_set=G_2,
                         template_set=G_1,
                         universe_of_discourse=[0, 10],
                         hedge_applier=hedge_applier)
        # chain = ['absolutely']
        # ind.apply_hedges(chain)
        chain = ['above']
        ind.apply_hedges(chain)

        C_2 = ind.current_fuzzy_set

        print(union_intersection_sim(G_2, C_2, [0, 10]))
        sf.LinguisticVariable([G_2, C_2], universe_of_discourse=[0, 10]).plot()


if __name__ == "__main__":
    if not nltk:
        raise Exception("ERROR: please, install nltk for cluster labeling facilities")
    test_CFG_manager()
    test_fuzzy_templates()
    test_grammar_guided_builder()
    test_individual()
