import re
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple

from simpful import FuzzySystem, LinguisticVariable, FuzzySet, Gaussian_MF
from simpful.cluster_labeling import AutoReport, FuzzyClusterNamer
from simpful.cluster_labeling.components.plot_utils import plot_clusters_and_approx

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x  # Fallback to a no-op


def _approximate_lv_labels(linguistic_variable: LinguisticVariable,
                           new_names: Optional[dict] = None) -> LinguisticVariable:
    """
    Create a linguistic variable with approximated labels

    :param linguistic_variable: original linguistic variable containing fuzzy sets.
    :param new_names: Optional dictionary mapping original concepts to new names for the labels
    :return: A new LinguisticVariable with approximated labels for the fuzzy sets
    """
    # Create approximator and run
    approximator = FuzzyClusterNamer(linguistic_variable, linguistic_variable.get_universe_of_discourse())
    # Get approximate sets
    approximate_sets: List[FuzzySet] = approximator.run_approximator(plot=False)
    # Get original fuzzy sets
    original_fuzzy_sets: List[FuzzySet] = linguistic_variable._FSlist
    new_fuzzy_sets = []
    for original_fs, approx_fs in zip(original_fuzzy_sets, approximate_sets):
        # Note: We keep the original function, but use the labels of the new ones
        new_fs = FuzzySet(function=Gaussian_MF(original_fs._funpointer._mu, original_fs._funpointer._sigma),
                          term=approx_fs.get_term())
        new_fuzzy_sets.append(new_fs)
    concept = linguistic_variable._concept if new_names is None else new_names[linguistic_variable._concept]
    concept = concept.replace(" ", "_")

    return LinguisticVariable(new_fuzzy_sets, concept=concept,
                              universe_of_discourse=linguistic_variable.get_universe_of_discourse())


def _make_modified_rules(new_fuzzy_system: FuzzySystem,
                         old_fuzzy_system: FuzzySystem,
                         variable_rename_map: dict = None) -> Tuple[List[str], List[str], dict]:
    """
    Update the fuzzy system rule with new names.
    Variables are ordered as before, so we can seek a correspondence.

    :param new_fuzzy_system: a FuzzySystem whose linguistic variables have been updated BUT without rules
    :param old_fuzzy_system: the old FuzzySystem, with the same variables but renamed and the old rules
    :param variable_rename_map: dictionary that defines new names for variables
    :return:  A tuple containing the newly formulated rules, pretty rules, and function correspondences
    """

    old_to_new_correspondences: Dict[Any, dict] = {}
    function_correspondences: dict = {}
    # Iterate through linguistic variable sets
    for (old_key, old_var), (new_key, new_var) in zip(old_fuzzy_system._lvs.items(),
                                                      new_fuzzy_system._lvs.items()):
        assert old_key == new_key, "Misalignment in Fuzzy systems (something has gone terribly wrong)."
        old_to_new_correspondences[old_key] = {}
        for old_fs, new_fs in zip(old_var._FSlist, new_var._FSlist):
            old_to_new_correspondences[old_key][old_fs.get_term()] = new_fs.get_term()
    # E.g. of old_to_new_correspondences:
    # {'Age':
    #   {'cluster1': 'below low', 'cluster2': 'above medium' ...
    # ...
    newly_formulated_rules = []
    pretty_rules = []
    then_blocks = []
    # Iterate original rules
    for old_rule in old_fuzzy_system.get_rules():
        # Find if and then block
        if_block, then_block = __extract_if_then_block(old_rule)
        then_blocks.append(then_block)

        single_pretty_rules, single_rules = __process_if_then_block(if_block, then_block, old_to_new_correspondences,
                                                                    function_correspondences, variable_rename_map)
        newly_formulated_rules.append(single_rules)
        pretty_rules.append(single_pretty_rules)
    overall_new_rules = [f'IF {" AND ".join(rr)} {tb}' for rr, tb in zip(newly_formulated_rules, then_blocks)]
    overall_pretty_rules = []
    for rr, tb in zip(pretty_rules, then_blocks):
        pretty_newline_rules = "\nAND ".join(rr)
        overall_pretty_rules.append(f'IF {pretty_newline_rules}'
                                    f'\n{tb.replace("(", "").replace(")", "")}')

    return overall_new_rules, overall_pretty_rules, function_correspondences


def __process_if_then_block(if_block: str, then_block: str,
                            old_to_new_correspondences: Dict[Any, dict],
                            function_correspondences: dict,
                            variable_rename_map: dict) -> Tuple[List[str], List[str]]:
    """
    Process the 'if' and 'then' blocks of a fuzzy rule to create new and pretty rules

    :param if_block: The 'if' part of the fuzzy rule
    :param then_block: The 'then' part of the fuzzy rule
    :param old_to_new_correspondences: A mapping of old fuzzy set terms to new fuzzy set terms
    :param function_correspondences: A dictionary to store function correspondences
    :param variable_rename_map: Optional dictionary that defines new names for variables
    :return: A tuple containing the new pretty rules and the new rules
    """
    single_rules = []
    single_pretty_rules = []
    # Iterate concepts
    for concept, concept_term_map in old_to_new_correspondences.items():
        concept_match = re.search(concept, if_block, flags=re.IGNORECASE)
        # If the concept is in the rule
        if concept_match:
            concept_start = concept_match.start()
            # Isolate rule (first closing bracket)
            bracket_match = re.search(r"\)", if_block[concept_start:], flags=re.IGNORECASE).start()
            individual_rule = if_block[concept_start:bracket_match + concept_start]
            # Rule will be "variable IS cluster"
            variable_name, sanity_is, cluster_name = individual_rule.split()
            assert sanity_is.upper() == "IS", f"Error in rule decomposition for {individual_rule}"
            if variable_rename_map is not None:
                new_var_pretty_name = variable_rename_map[variable_name]
                new_var_name = new_var_pretty_name.replace(" ", "_")
            else:
                new_var_name = variable_name
                new_var_pretty_name = variable_name

            # Simpful does NOT like spaces in cluster names, add underscore
            corresponding_label = concept_term_map[cluster_name].replace(' ', '_')
            new_rule = f"({new_var_name} IS {corresponding_label})"
            new_rule_pretty = f"{new_var_pretty_name} IS {concept_term_map[cluster_name]}"

            __build_function_correspondences(corresponding_label, function_correspondences, new_var_name, then_block)
            # ---------------------
            single_rules.append(new_rule)
            single_pretty_rules.append(new_rule_pretty)
    return single_pretty_rules, single_rules


def __build_function_correspondences(corresponding_label: str,
                                     function_correspondences : dict,
                                     new_var_name: str,
                                     then_block: str) -> None:
    """
    Build function correspondences from the 'then' block of a fuzzy rule.

    :param corresponding_label: The new label corresponding to the fuzzy set
    :param function_correspondences: A dictionary to store function correspondences
    :param new_var_name: The new variable name after renaming
    :param then_block: The 'then' part of the fuzzy rule, which contains the function name
    :return: None (works inplace)
    """
    # Find the end of the function name (first closing parenthesis)
    end = re.search(r"\)", then_block, flags=re.IGNORECASE).start()
    # Find the start of the function name (after "IS")
    start = re.search("IS", then_block, flags=re.IGNORECASE).end()
    fun_name = then_block[start:end].strip()
    # rule_function_correspondences.append((new_var_name, corresponding_label, fun_name))
    function_correspondences.setdefault(new_var_name, {})
    function_correspondences[new_var_name].setdefault(corresponding_label, []).append(fun_name)


def __extract_if_then_block(old_rule: str) -> Tuple[str, str]:
    """
    Extract the 'if' and 'then' blocks from a fuzzy rule

    :param old_rule: The fuzzy rule string containing 'if' and 'then' parts
    :return: A tuple containing the 'if' block and the 'then' block
    """
    if_block_start = re.search("if", old_rule, flags=re.IGNORECASE).start()
    then_block_start = re.search("then", old_rule, flags=re.IGNORECASE).start()
    # Isolate rules
    if_block = old_rule[if_block_start:then_block_start].strip()
    then_block = old_rule[then_block_start:].strip()
    return if_block, then_block


def _auto_label(original_fs: FuzzySystem,
                out_path: Optional[Path] = None,
                new_variable_names: Dict[str, str] = None):
    """
    Automatically label a fuzzy system by approximating its linguistic variables to templates.

    :param original_fs: original fuzzy system to be labeled
    :param out_path: Optional path to save the generated rules to a file
    :param new_variable_names: Optional dictionary mapping old variable names to new names
    :return: Tuple containing the relabeled fuzzy system and a dictionary of function correspondences
    """
    # Create a new fuzzy system
    new_fs = FuzzySystem(show_banner=False)
    # Copy the output functions weights, the variables remain the same for now
    for fun, value in original_fs._outputfunctions.items():
        new_fs.set_output_function(fun, value)
    # Iterate through language variables, approximate
    for i, (lv_name, lv_item) in tqdm(enumerate(original_fs._lvs.items()),
                                      desc="Approximating sets...",
                                      total=len(original_fs._lvs)):
        lv = _approximate_lv_labels(lv_item, new_variable_names)
        new_fs.add_linguistic_variable(lv_name, lv)

    new_rules, pretty_rules, fun_correspondences = _make_modified_rules(new_fs, original_fs)  # , new_names)
    new_fs.add_rules(new_rules)
    if out_path is not None:
        out_rule_path = out_path / 'rules.txt'
        with open(out_rule_path, 'w') as file1:
            file1.write("\n\n".join(pretty_rules))

    return new_fs, fun_correspondences


def _generate_plots_from_fs(fs: FuzzySystem, out_path: Path, fun_correspondences):
    """
     Generate plots for the linguistic variables in a fuzzy system and save them to a specified directory.

    :param fs: fuzzy system containing linguistic variables to plot
    :param out_path: path where the plots will be saved
    :param fun_correspondences: dictionary mapping functions to their correspondences for plotting
    :return:
    """
    plots_foder: Path = out_path / "plots"
    plots_foder.mkdir(exist_ok=True, parents=True)
    for idx, linguistic_variable in enumerate(fs._lvs.values()):
        # linguistic_variable.plot()
        plot_clusters_and_approx(linguistic_variable,
                                 plots_foder / f"{idx}.png",
                                 correspondences=fun_correspondences,
                                 num_functions=len(fs.get_rules())
                                 )


def _auto_report_generate(out_path: Path,
                          **report_args) -> None:
    """
    Generate an automatic report in PDF format

    :param out_path: path where the report will be saved
    :param report_args: Additional arguments for report customization ["title"]
    :return: None
    """
    pdf = AutoReport()
    pdf.set_title(report_args.get('title', "Report title"))
    pdf.set_author('Automatically generated from PyFume data')

    pdf.print_figures(1, 'Features', figures=[str(x) for x in (out_path / "plots").glob("*.png")])
    # TODO [IMPROVEMENT]: color functions with color in the actual graph
    pdf.print_text(2, 'Rules', str(out_path / 'rules.txt'))

    pdf.output(str(out_path / f'report.pdf'), 'F')


def approximate_fs_labels(system_to_label: FuzzySystem,
                          output_path: Union[str, Path] = "output",
                          generate_report: bool = False,
                          **report_args) -> None:
    if not isinstance(output_path, Path):
        output_path: Path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    # ---------------------------------------
    out_path: Path = output_path / "report"
    out_path.mkdir(exist_ok=True, parents=True)
    new_fs, fun_correspondences = _auto_label(system_to_label, out_path)
    _generate_plots_from_fs(new_fs, out_path, fun_correspondences)
    if generate_report:
        _auto_report_generate(out_path, **report_args)
