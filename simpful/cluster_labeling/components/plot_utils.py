import re
from pathlib import Path
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d

from simpful import LinguisticVariable, FuzzySystem
from simpful.cluster_labeling.components import GaussianFuzzyTemplates


def plot_clusters_and_approx(fs: FuzzySystem,
                             ling_var: LinguisticVariable,
                             outputfile: Union[str, Path] = "",
                             plot_color_by_rule_output: bool = False,
                             xscale: str = "linear",
                             fig_title: str = "Default"):
    sns.set_style("whitegrid")
    u_o_d = ling_var.get_universe_of_discourse()
    templates = GaussianFuzzyTemplates(u_o_d)._template_list

    fig, ax = plt.subplots(figsize=(6.4, 5.8), layout='constrained')

    # (x-axis generation code...)
    mi, ma = u_o_d
    if xscale == "linear":
        x = np.linspace(mi, ma, 10000)
    elif xscale == "log":
        if mi < 0 < ma:
            x = np.geomspace(mi, -np.finfo(np.float64).eps, 5000) + np.geomspace(np.finfo(np.float64).eps, ma, 5000)
        elif mi == 0:
            x = np.geomspace(np.finfo(np.float64).eps, ma, 10000)
        else:
            x = np.geomspace(mi, ma, 10000)
    else:
        raise Exception("ERROR: scale " + xscale + " not supported.")

    # --- FUNCTIONAL COLORING LOGIC ---
    # 1. Find all unique output terms from the rules to create a consistent color palette
    all_output_terms = set()
    for rule in fs.get_rules():
        match = re.search(r"THEN\s*\(\s*\w+\s+IS\s+([\w_]+)\s*\)", rule, re.IGNORECASE)
        if match:
            all_output_terms.add(match.group(1).strip())

    # 2. Create the color map based on all possible outputs
    prop_cycler = plt.rcParams['axes.prop_cycle']
    colors = {term: color['color'] for term, color in zip(sorted(list(all_output_terms)), prop_cycler)}

    # ... after the `colors` dictionary is created ...
    # --- Logic for varying line styles ---
    linestyles = ['-', '--', ':', '-.']
    # A dictionary to track the next style index for each color
    color_to_style_index = {c: 0 for c in colors.values()}

    main_lines = []
    if plot_color_by_rule_output:
        # --- BEHAVIOR 1: Color by Rule Output ---
        all_output_terms = set()
        for rule in fs.get_rules():
            match = re.search(r"THEN\s*\(\s*\w+\s+IS\s+([\w_]+)\s*\)", rule, re.IGNORECASE)
            if match:
                all_output_terms.add(match.group(1).strip())

        prop_cycler = plt.rcParams['axes.prop_cycle']
        colors = {term: color['color'] for term, color in zip(sorted(list(all_output_terms)), prop_cycler)}
        linestyles = ['-', '--', ':', '-.']
        color_to_style_index = {c: 0 for c in colors.values()}

        for fuzzy_set in ling_var._FSlist:
            output_term_for_this_set = "default"
            term_in_rule_format = fuzzy_set.get_term().replace(' ', '_')
            rule_antecedent = f"({ling_var._concept} IS {term_in_rule_format})"
            for rule in fs.get_rules():
                if rule_antecedent in rule:
                    match = re.search(r"THEN\s*\(\s*\w+\s+IS\s+([\w_]+)\s*\)", rule, re.IGNORECASE)
                    if match:
                        output_term_for_this_set = match.group(1).strip()
                        break
            color = colors.get(output_term_for_this_set, 'black')
            style_index = color_to_style_index.get(color, 0)
            linestyle = linestyles[style_index % len(linestyles)]
            if color in color_to_style_index:
                color_to_style_index[color] += 1

            if fuzzy_set._type == "function":
                y = [fuzzy_set.get_value(_xx) for _xx in x]
                line = ax.plot(x, y, linestyle=linestyle, lw=2, label=fuzzy_set.get_term(), color=color)
                main_lines.extend(line)
            else:
                f = interp1d(fuzzy_set._points.T[0], fuzzy_set._points.T[1], bounds_error=False,
                             fill_value=(fuzzy_set.boundary_values[0], fuzzy_set.boundary_values[1]))
                line = ax.plot(x, f(x), linestyle=linestyle, label=fuzzy_set.get_term(), color=color)
                main_lines.extend(line)

        if colors:
            output_handles = [mpatches.Patch(color=c, label=t.replace('_', ' ')) for t, c in colors.items()]
            fig.legend(handles=output_handles, loc='upper right', bbox_to_anchor=(1, 1), fontsize='small',
                       title='Rule Outputs')

    else:
        # --- BEHAVIOR 2: Simple Coloring ---
        for fuzzy_set in ling_var._FSlist:
            if fuzzy_set._type == "function":
                y = [fuzzy_set.get_value(_xx) for _xx in x]
                line = ax.plot(x, y, "-", lw=2, label=fuzzy_set.get_term())
                main_lines.extend(line)
            else:
                f = interp1d(fuzzy_set._points.T[0], fuzzy_set._points.T[1], bounds_error=False,
                             fill_value=(fuzzy_set.boundary_values[0], fuzzy_set.boundary_values[1]))
                line = ax.plot(x, f(x), "-", label=fuzzy_set.get_term())
                main_lines.extend(line)

    t_lines = []
    for t in templates:
        t_line = ax.plot(x, [t.get_value(xx) for xx in x], "-.", lw=1, label=t.get_term(), color="lightgray")
        t_lines.extend(t_line)

    plt.xlabel(ling_var._concept.replace("_", " "), fontsize=12)

    # (Legend and labels code...)
    fig.legend(handles=main_lines, loc='lower left', bbox_to_anchor=(0.1, 0),
               fontsize='small', title='Membership Functions')
    fig.legend(handles=t_lines, loc='lower right', bbox_to_anchor=(0.9, 0),
               fontsize='small', title='Template references')

    plt.xlabel(ling_var._concept.replace("_", " "), fontsize=12)
    plt.ylabel("Membership degree", fontsize=12)
    plt.ylim(bottom=-0.05, top=1.05)
    if xscale == "log":
        plt.xscale("symlog", linthresh=10e-2)
        plt.xlim(x[0], x[-1])

    if outputfile != "":
        fig.savefig(outputfile, dpi=300)

    if fig_title != "":
        plt.title(fig_title)
    else:
        universe_bounds = " to ".join([str(x) for x in u_o_d])
        terms = ", ".join([x._term.title() for x in ling_var._FSlist])
        plt.title(f"{terms} in [{universe_bounds}]")
