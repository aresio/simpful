import re
from pathlib import Path
from typing import Any, List, Optional, Dict

from simpful import FuzzySystem, LinguisticVariable, FuzzySet, Gaussian_MF
from tqdm import tqdm

from simpful.cluster_labeling import FuzzyClusterNamer, AutoReport
from simpful.cluster_labeling.components.plot_utils import plot_clusters_and_approx
from simpful.cluster_labeling.tests.Cement_Simpful_code import get_cement_fs
# from simpful.cluster_labeling.tests.Simpful_code_2rules import wrap_2rules


def approximate_lv(linguistic_variable: LinguisticVariable, new_names: Optional[dict] = None):
    # Create approximator and run
    approximator = FuzzyClusterNamer(linguistic_variable, linguistic_variable.get_universe_of_discourse())
    approximate_sets: List[FuzzySet] = approximator.run_approximator(plot=False)
    original_fuzzy_sets: List[FuzzySet] = linguistic_variable._FSlist
    new_fuzzy_sets = []
    for original_fs, approx_fs in zip(original_fuzzy_sets, approximate_sets):
        # Note: previously I changed the term, but maybe best to just create new ones
        new_fs = FuzzySet(function=Gaussian_MF(original_fs._funpointer._mu, original_fs._funpointer._sigma),
                          term=approx_fs.get_term())
        new_fuzzy_sets.append(new_fs)
        # original_fs._term = approx_fs.get_term()
    lv = LinguisticVariable(new_fuzzy_sets, concept=linguistic_variable._concept if new_names is None
    else new_names[linguistic_variable._concept].replace(" ", "_"),
                            universe_of_discourse=linguistic_variable.get_universe_of_discourse())
    return lv


def make_modified_rules(new_fuzzy_system: FuzzySystem,
                        old_fuzzy_system: FuzzySystem,
                        variable_rename_map: dict = None):
    """
    Update the fuzzy system rule with new names. Variables are ordered as before, so we can seek
    a correspondence.

    :param new_fuzzy_system: a FuzzySystem whose linguistic variables have been updated and without rules
    :param old_fuzzy_system: the old FuzzySystem, with the same variables but renamed and the old rules
    :param variable_rename_map: dictionary that defines new names for variables
    :return:
    """

    old_to_new_correspondences: Dict[Any, dict] = {}
    function_correspondences: dict = {}
    for (old_key, old_var), (new_key, new_var) in zip(old_fuzzy_system._lvs.items(),
                                                      new_fuzzy_system._lvs.items()):
        assert old_key == new_key
        old_to_new_correspondences[old_key] = {}
        for old_fs, new_fs in zip(old_var._FSlist, new_var._FSlist):
            old_term = old_fs.get_term()
            new_term = new_fs.get_term()
            old_to_new_correspondences[old_key][old_term] = new_term
    newly_formulated_rules = []
    pretty_rules = []
    then_blocks = []
    for old_rule in old_fuzzy_system.get_rules():
        # Find if and then block
        if_block_start = re.search("if", old_rule, flags=re.IGNORECASE).regs[0][0]
        then_block_start = re.search("then", old_rule, flags=re.IGNORECASE).regs[0][0]
        # Isolate rules
        if_block = old_rule[if_block_start:then_block_start].strip()
        then_block = old_rule[then_block_start:].strip()
        then_blocks.append(then_block)
        single_rules = []
        single_pretty_rules = []
        # Iterate concepts
        for concept in old_to_new_correspondences.keys():
            concept_match = re.search(concept, if_block, flags=re.IGNORECASE)
            # If the concept is in the rule
            if concept_match is not None:
                concept_start = concept_match.regs[0][0]
                # Isolate rule (first closing bracket)
                bracket_match = re.search("\)", if_block[concept_start:], flags=re.IGNORECASE).regs[0][0]
                individual_rule = if_block[concept_start:bracket_match + concept_start]
                # Rule will be "variable IS cluster"
                variable_name, sanity_is, cluster_name = individual_rule.split()
                assert sanity_is == "IS", f"Error in rule decomposition for {old_rule}"
                if variable_rename_map is not None:
                    new_var_pretty_name = variable_rename_map[variable_name]
                    new_var_name = new_var_pretty_name.replace(" ", "_")
                else:
                    new_var_name = variable_name
                    new_var_pretty_name = variable_name

                # Simpful does NOT like spaces in cluster names, add underscore
                corresponding_label = old_to_new_correspondences[concept][cluster_name].replace(' ', '_')
                new_rule = f"({new_var_name} IS {corresponding_label})"
                new_rule_pretty = f"{new_var_pretty_name} IS {old_to_new_correspondences[concept][cluster_name]}"

                # ----- TEST ---------
                end = re.search("\)", then_block, flags=re.IGNORECASE).regs[0][0]
                start = re.search("IS", then_block, flags=re.IGNORECASE).regs[0][1]

                fun_name = then_block[start:end].strip()
                # rule_function_correspondences.append((new_var_name, corresponding_label, fun_name))
                function_correspondences.setdefault(new_var_name, {})
                function_correspondences[new_var_name].setdefault(corresponding_label, []).append(fun_name)
                # ---------------------
                single_rules.append(new_rule)
                single_pretty_rules.append(new_rule_pretty)
        newly_formulated_rules.append(single_rules)
        pretty_rules.append(single_pretty_rules)
    overall_new_rules = [f'IF {" AND ".join(rr)} {tb}' for rr, tb in zip(newly_formulated_rules, then_blocks)]
    overall_pretty_rules = []
    for rr, tb in zip(pretty_rules, then_blocks):
        pretty_newline_rules = "\nAND ".join(rr)
        overall_pretty_rules.append(f'IF {pretty_newline_rules}'
                                    f'\n{tb.replace("(", "").replace(")", "")}')
    # Iterate over rules
    # for idx, rule in enumerate(old_rules):
    #     # Iterate over concepts
    #     for lv_concept, lv in new_linguistic_variables.items():
    #         # Search this concept within the rule string
    #         lv_name_start, lv_name_end = re.search(lv_concept, rule).regs[0]
    #         # Find the end of the current sub-rule
    #         end_of_subrule = rule[lv_name_end:].find(")")
    #         # Split the components, which should be "IS" and "clusterX"
    #         is_str, cluster_str = rule[lv_name_end:lv_name_end + end_of_subrule].split()
    #         # Sanity check
    #         assert is_str == "IS"
    #         # Get the names of these renamed cluster
    #         for renamed_cluster in [x.get_term() for x in lv._FSlist]:
    #             # Find the name of the cluster (clusterX)
    #             position: int = renamed_cluster.find(cluster_str)
    #             # If it has been found, replace it
    #             if position > 0:
    #                 # New cluster string (remove the (clusterX) nomeclature
    #                 # TODO: Is it ok to keep internally?
    #                 # TODO: CHECK
    #                 new_cluster_str = renamed_cluster[:position - 1].strip()
    #                 # new_cluster_str = renamed_cluster  # TODO: maybe better to keep it as is for the internal rules
    #                 replaced_bit = rule[lv_name_start:lv_name_end + end_of_subrule].replace(cluster_str,
    #                                                                                         new_cluster_str)
    #                 rule = rule[:lv_name_start] + replaced_bit + rule[lv_name_end + end_of_subrule:]
    #     new_fs_rules.append(rule)
    #     rule = rule.replace("(", "").replace(")", "")
    #     rule = rule.replace("AND", "\nAND").replace("THEN", "\nTHEN")
    #     new_rule = f"[{idx}] {rule}\n\n"
    #     new_rules += new_rule
    #     print(new_rule)

    return overall_new_rules, overall_pretty_rules, function_correspondences


def auto_label(fuzzy_system: FuzzySystem, out_path: Optional[Path] = None):
    # new_names = {
    #     "ETA_RISCONTRO_PRIMOI_SEGNI_DI_SVILUPPO_PUBERALE_anamnesi_patologica_prossima": "Age at onset of puberty",
    #     "STATURA_SDS_Cacciari": "Cacciari standard deviation score",
    #     "PICCO_LH_UL": "LH peak",
    #     "Delta_et_ossea__et_cronologica": "Difference between bone age and cronological age",
    #     "VOLUME_UTERINO_ml": "Uterine volume",
    #     "DIAMETRO_UTERINO_UNO_corrispondente_a_diametro_maggiore__mm": "Main uterine diameter",
    # }
    # Create the new fuzzy system
    new_fs = FuzzySystem(show_banner=False)
    # Copy the output functions weights, the variables remain the same for now
    for fun, value in fuzzy_system._outputfunctions.items():
        new_fs.set_output_function(fun, value)
    # Iterate through language variables, approximate
    for i, (lv_name, lv_item) in tqdm(enumerate(fuzzy_system._lvs.items()),
                                      desc="Approximating sets...",
                                      total=len(fuzzy_system._lvs)):
        # lv = approximate_lv(lv_item, new_names)
        lv = approximate_lv(lv_item)
        new_fs.add_linguistic_variable(lv_name, lv)

    new_rules, pretty_rules, fun_correspondences = make_modified_rules(new_fs, fuzzy_system) #, new_names)
    new_fs.add_rules(new_rules)
    if out_path is not None:
        out_rule_path = out_path / 'rules.txt'
        with open(out_rule_path, 'w') as file1:
            file1.write("\n\n".join(pretty_rules))

    return new_fs, fun_correspondences


def generate_plots_from_fs(fs: FuzzySystem, out_path: Path, fun_correspondences):
    plots_foder: Path = out_path / "plots"
    plots_foder.mkdir(exist_ok=True, parents=True)
    for idx, linguistic_variable in enumerate(fs._lvs.values()):
        # linguistic_variable.plot()
        plot_clusters_and_approx(linguistic_variable,
                                 plots_foder / f"{idx}.png",
                                 correspondences=fun_correspondences,
                                 num_functions=len(fs.get_rules())
                                 )


def auto_report_generate(out_path: Path, name: str = "test_report"):
    pdf = AutoReport()
    pdf.set_title("Puberty Dataset Report")
    pdf.set_author('Automatically generated from PyFume data')

    pdf.print_figures(1, 'Features', figures=[str(x) for x in (out_path / "plots").glob("*.png")])
    # TODO: color fun with color in the actual graph
    pdf.print_text(2, 'Rules', str(out_path / 'rules.txt'))

    pdf.output(str(out_path / f'{name}.pdf'), 'F')


def main():
    # # ---------------------------------------
    # fs_2_system: FuzzySystem = wrap_2rules()
    fs_2_system: FuzzySystem = get_cement_fs()
    # ---------------------------------------
    base_path: Path = Path('output')
    base_path.mkdir(exist_ok=True, parents=True)
    # ---------------------------------------
    name = "report"
    out_path: Path = base_path / name
    out_path.mkdir(exist_ok=True, parents=True)
    new_fs, fun_correspondences = auto_label(fs_2_system, out_path)
    generate_plots_from_fs(new_fs, out_path, fun_correspondences)
    auto_report_generate(out_path, name)
    # ---------------------------------------


if __name__ == '__main__':
    main()
