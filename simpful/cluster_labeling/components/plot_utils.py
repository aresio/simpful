from pathlib import Path
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d

from simpful import LinguisticVariable
from simpful.cluster_labeling.components import GaussianFuzzyTemplates


def plot_clusters_and_approx(ling_var: LinguisticVariable,
                             outputfile: Union[str, Path] = "",
                             xscale: str = "linear",
                             fig_title: str = "Default",
                             correspondences: Optional[dict] = None,
                             num_functions: int = 2):
    sns.set_style("whitegrid")
    u_o_d = ling_var.get_universe_of_discourse()
    templates = GaussianFuzzyTemplates(u_o_d)._template_list

    fig, ax = plt.subplots(figsize=(6.4, 5.8), layout='constrained')

    # ----------------
    mi, ma = u_o_d
    if xscale == "linear":
        x = np.linspace(mi, ma, 10000)
    elif xscale == "log":
        if mi < 0 < ma:
            x = np.geomspace(mi, -np.finfo(np.float64).eps, 5000) + np.geomspace(np.finfo(np.float64).eps, ma, 5000)
            # raise Exception("ERROR: cannot plot in log scale with negative universe of discourse")
        elif mi == 0:
            x = np.geomspace(np.finfo(np.float64).eps, ma, 10000)
        else:
            x = np.geomspace(mi, ma, 10000)
    else:
        raise Exception("ERROR: scale " + xscale + " not supported.")

    linestyles = ["-"]
    ax = plt.gca()
    if hasattr(ax._get_lines, '_cycler_items'):
        color_cycler = iter(ax._get_lines._cycler_items)
    else:
        color_cycler = iter(ax._get_lines.prop_cycler)
    colors = {f"fun{x}": col["color"] for x, col in zip(range(1, num_functions + 1), color_cycler)}
    main_lines = []

    for nn, fs in enumerate(ling_var._FSlist):
        corresponding_function = correspondences[ling_var._concept][fs.get_term().replace(" ", "_")].pop(0)
        if fs._type == "function":
            y = [fs.get_value(xx) for xx in x]
            color = colors[corresponding_function]
            main_line = ax.plot(x, y, "-", lw=2, label=fs._term, color=color)
        else:
            sns.regplot(x=fs._points.T[0], y=fs._points.T[1], marker="d", color="red", fit_reg=False)
            f = interp1d(fs._points.T[0], fs._points.T[1], bounds_error=False,
                         fill_value=(fs.boundary_values[0], fs.boundary_values[1]))
            main_line = ax.plot(x, f(x), linestyles[nn % 4], label=fs._term, )
        main_lines.extend(main_line)

    t_lines = []
    for t in templates:
        t_line = ax.plot(x, [t.get_value(xx) for xx in x], "-.", lw=1, label=t.get_term(), color="lightgray")
        t_lines.extend(t_line)

    # Create another legend for the second line.
    fig.legend(handles=main_lines, loc='center left')
    fig.legend(handles=t_lines, loc='center right')

    plt.xlabel(ling_var._concept.replace("_", " "), fontsize=12)
    plt.ylabel("Membership degree", fontsize=12)
    plt.ylim(bottom=-0.05, top=1.05)
    plt.ylim(bottom=-0.05, top=1.05)
    if xscale == "log":
        plt.xscale("symlog", linthresh=10e-2)
        plt.xlim(x[0], x[-1])

    # ----------------

    if outputfile != "":
        fig.savefig(outputfile, dpi=300)

    if fig_title != "":
        plt.title(fig_title)
    else:
        universe_bounds = " to ".join([str(x) for x in u_o_d])
        terms = ", ".join([x._term.title() for x in ling_var._FSlist])
        plt.title(f"{terms} in [{universe_bounds}]")
    # plt.show()
