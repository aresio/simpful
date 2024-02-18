from .fuzzy_sets import FuzzySet, MF_object, Triangular_MF, SingletonsSet
from .rule_parsing import recursive_parse, preparse, postparse
from numpy import array, linspace, meshgrid, geomspace, log10, finfo, float64
from scipy.interpolate import interp1d
from copy import deepcopy
from collections import defaultdict, OrderedDict
import re
import string
from math import prod
try:
    from matplotlib.pyplot import figure, axes, plot, show, title, subplots, legend, colorbar
    matplotlib = True
except ImportError:
    matplotlib = False
try:
    import seaborn as sns
except:
    pass

# constant values
linestyles= ["-", "--", ":", "-."]

# for sanitization
valid_characters = string.ascii_letters + string.digits + "()_ "

# constants for templating system 
TEMPLATES_ENGAGED = 0
TEMPLATES_DISENGAGED = 1
TEMPLATES_MISSING_INFO = 2

class UndefinedUniverseOfDiscourseError(Exception):

    def __init__(self, message):
        self.message = message


class LinguisticVariable(object):
    """
        Creates a new linguistic variable.

        Args:
            FS_list: a list of FuzzySet instances.
            concept: a string providing a brief description of the concept represented by the linguistic variable (optional).
            universe_of_discourse: a list of two elements, specifying min and max of the universe of discourse. Optional, but it must be specified to exploit plotting facilities.
    """

    def __init__(self, FS_list=[], concept=None, universe_of_discourse=None):
        
        if FS_list==[]:
            raise Exception("ERROR: please specify at least one fuzzy set")
        self._universe_of_discourse = universe_of_discourse
        self._FSlist = FS_list
        self._concept = concept


    def get_values(self, v):
        result = {}
        for fs in self._FSlist:
            result[fs._term] = fs.get_value(v)
        return result


    def get_index(self, term):
        for n, fs in enumerate(self._FSlist):
            if fs._term == term: return n
        return -1


    def get_universe_of_discourse(self):
        """
        This method provides the leftmost and rightmost values of the universe of discourse of the linguistic variable.

        Returns:
            the two extreme values of the universe of discourse.
        """
        if self._universe_of_discourse is not None:
            return self._universe_of_discourse
        mins = []
        maxs = []
        try:
            for fs in self._FSlist:
                mins.append(min(fs._points.T[0]))
                maxs.append(max(fs._points.T[0]))
        except AttributeError:
            raise UndefinedUniverseOfDiscourseError("Cannot get the universe of discourse. Please, use point-based fuzzy sets or explicitly specify a universe of discourse")
        return min(mins), max(maxs)


    def draw(self, ax, TGT=None, element=None, highlight=None, xscale="linear"):
        """
        This method returns a matplotlib ax, representing all fuzzy sets contained in the liguistic variable.

        Args:
            ax: the matplotlib axis to plot to.
            TGT: (deprecated) show the memberships of a specific element of discourse TGT in the figure.
            element: show the memberships of a specific element of discourse in the figure.
            highlight: string, indicating the linguistic term/fuzzy set to highlight in the plot.
            xscale: default "linear", supported scales "log". Changes the scale of the xaxis.
        Returns:
            A matplotlib axis, representing all fuzzy sets contained in the liguistic variable.
        """
        if matplotlib == False:
            raise Exception("ERROR: please, install matplotlib for plotting facilities")
        if TGT is not None:
            print("Warning: argument 'TGT' is deprecated and will be discontinued in future relseases. Please use 'element'")
            element = TGT

        mi, ma = self.get_universe_of_discourse()
        if xscale == "linear":
            x = linspace(mi, ma, 10000)
        elif xscale == "log":
            if mi < 0 and ma > 0:
                x = geomspace(mi, -finfo(float64).eps, 5000) + geomspace(finfo(float64).eps, ma, 5000)
                # raise Exception("ERROR: cannot plot in log scale with negative universe of discourse")
            elif mi == 0:
                x = geomspace(finfo(float64).eps, ma, 10000)
            else:
                x = geomspace(mi, ma, 10000)
        else:
            raise Exception("ERROR: scale "+xscale+" not supported.")

        
        if highlight is None:
            linestyles= ["-", "--", ":", "-."]
        else:
            linestyles= ["-"]*4

        if element is not None:
            ax.plot([element, element], [0.0, 1], color="red", linestyle="--", linewidth=2.0, label="Value")
        

        for nn, fs in enumerate(self._FSlist):
            
            # singleton
            if isinstance(fs, SingletonsSet):
                xs = [pair[0] for pair in fs._funpointer._pairs]
                ys = [pair[1] for pair in fs._funpointer._pairs]
                ax.vlines(x=xs, ymin=0.0, ymax=ys, linestyles=linestyles[nn%4], color=ax._get_lines.get_next_color(), label=fs._term)
            

            elif fs._type == "function":
                y = [fs.get_value(xx) for xx in x]
                color = ax._get_lines.get_next_color()
                lw = 1

                if highlight==fs._term: 
                    color="red"
                    lw =5 
                elif highlight is not None:
                    color="lightgray"
                ax.plot(x,y, linestyles[nn%4], lw=lw, label=fs._term, color=color)
                 
                # plot the membership degree on the fuzzy set
                if element is not None:
                    top = fs.get_value(element)
                    cut_y = [min(top, yy) for yy in y] 
                    ax.fill([x[0]]+list(x)+[x[-1]],[0]+cut_y+[0] ,  color=color, alpha=0.5,)


            else:
                sns.regplot(x=fs._points.T[0], y=fs._points.T[1], marker="d", color="red", fit_reg=False, ax=ax)
                f = interp1d(fs._points.T[0], fs._points.T[1], bounds_error=False, fill_value=(fs.boundary_values[0], fs.boundary_values[1]))
                ax.plot(x, f(x), linestyles[nn%4], label=fs._term,)
               
        
        ax.set_xlabel(self._concept)
        ax.set_ylabel("Membership degree")
        ax.set_ylim(bottom=-0.05, top=1.05)
        if xscale == "log":
            ax.set_xscale("symlog", linthresh=10e-2)
            ax.set_xlim(x[0], x[-1])
        if highlight is None: ax.legend(loc="best")
        return ax


    def plot(self, outputfile="", TGT=None, element=None, highlight=None, xscale="linear"):
        """
        Shows a plot representing all fuzzy sets contained in the liguistic variable.

        Args:
            outputfile: path and filename where the plot must be saved.
            TGT: (deprecated) show the memberships of a specific element of discourse TGT in the figure.
            element: show the memberships of a specific element of discourse in the figure.
            highlight: string, indicating the linguistic term/fuzzy set to highlight in the plot.
            xscale: default "linear", supported scales "log". Changes the scale of the xaxis.
        """
        if matplotlib == False:
            raise Exception("ERROR: please, install matplotlib for plotting facilities")
        if TGT is not None:
            print("Warning: argument 'TGT' is deprecated and will be discontinued in future relseases. Please use 'element'")
            element = TGT

        fig, ax = subplots(1,1)
        self.draw(ax=ax, element=element, highlight=highlight, xscale=xscale)

        if outputfile != "":
            fig.savefig(outputfile)

        show()
        
        

    def __repr__(self):
        if self._concept is None:
            text = "N/A"
        else:
            text = self._concept
        return "<Linguistic variable '"+text+"', contains fuzzy sets %s, universe of discourse: %s>" % (str(self._FSlist), str(self._universe_of_discourse))


class AutoTriangle(LinguisticVariable):
    """
        Creates a new linguistic variable, whose universe of discourse is automatically divided in a given number of fuzzy sets.
        The sets are all symmetrical, normalized, and for each element of the universe their memberships sum up to 1.
        
        Args:
            n_sets: (integer) number of fuzzy sets in which the universe of discourse must be divided.
            terms: list of strings containing linguistic terms for the fuzzy sets (must be appropriate to the number of fuzzy sets).
            universe_of_discourse: a list of two elements, specifying min and max of the universe of discourse.
            verbose: True/False, toggles verbose mode.
    """

    def __init__(self, n_sets=3, terms=None, universe_of_discourse=[0,1], verbose=False):
        
        if n_sets<2:
            raise Exception("Cannot create linguistic variable with less than 2 fuzzy sets.")

        control_points = [x*1/(n_sets-1) for x in range(n_sets)]
        low = universe_of_discourse[0]
        high = universe_of_discourse[1]
        control_points = [low + (high-low)*x for x in control_points]
        
        if terms is None:
            terms = ['case %d' % (i+1) for i in range(n_sets)]

        FS_list = []

        FS_list.append(FuzzySet(function=Triangular_MF(low,low,control_points[1]), term=terms[0]))

        for n in range(1, n_sets-1):
            FS_list.append(
                FuzzySet(function=Triangular_MF(control_points[n-1], control_points[n], control_points[n+1]), 
                    term=terms[n])
            )

        FS_list.append( FuzzySet(function=Triangular_MF(control_points[-2], high, high), term=terms[-1] ))

        super().__init__(FS_list, universe_of_discourse=universe_of_discourse)

        if verbose:
            for fs in FS_list:
                print(fs, fs.get_term())


class FuzzySystem(object):

    """
        Creates a new fuzzy system.

        Args:
            operators: a list of strings, specifying fuzzy operators to be used instead of defaults. Currently supported operators: 'AND_PRODUCT'.
            show_banner: True/False, toggles display of banner.
            sanitize_input: sanitize variables' names to eliminate non-accepted characters (under development).
            verbose: True/False, toggles verbose mode.
    """

    def __init__(self, operators=None, show_banner=True, sanitize_input=False, verbose=True):

        self._rules = []
        self._lvs = OrderedDict()
        self._variables = OrderedDict()
        self._crispvalues = OrderedDict()
        self._outputfunctions = OrderedDict()
        self._outputfuzzysets = OrderedDict()

        self._constants = []
        
        self._operators = operators

        self._detected_type = None

        self._replacement_dictionary = {}
        self._templates_enabled = False

        self._sanitize_input = sanitize_input
        if sanitize_input and verbose:
            print (" * Warning: Simpful rules sanitization is enabled, please pay attention to possible collisions of symbols.")

        if show_banner: self._banner()

    def _banner(self):
        import pkg_resources
        vrs = pkg_resources.get_distribution('simpful').version 
        print("  ____  __  _  _  ____  ____  _  _  __   ")
        print(" / ___)(  )( \\/ )(  _ \\(  __)/ )( \\(  ) v%s " % vrs)
        print(" \\___ \\ )( / \\/ \\ ) __/ ) _) ) \\/ (/ (_/\\ ")
        print(" (____/(__)\\_)(_/(__)  (__)  \\____/\\____/")
        print()
        print(" https://github.com/aresio/simpful")
        print()


    def get_fuzzy_sets(self, variable_name):
        """
            Returns the list of FuzzySet objects associated to one linguistic variable.
            
            Args:
                variable_name: name of the linguistic variable.

            Returns:
                a list containing FuzzySet objects.
        """
        try:
            return self._lvs[variable_name]._FSlist
        except ValueError:
            raise Exception("ERROR: linguistic variable %s does not exist" % variable_name)


    def get_fuzzy_set(self, variable_name, fs_name):
        """
            Returns a FuzzySet object associated to a linguistic variable.
            
            Args:
                variable_name: name of the linguistic variable.
                fs_name: linguistic term associated to the fuzzy set.

            Returns:
                a FuzzySet object.
        """
        try:
            LV =  self._lvs[variable_name]
        except ValueError:
            raise Exception("ERROR: linguistic variable %s does not exist" % (variable_name))

        for fs in LV._FSlist:
            if fs._term==fs_name:       return fs

        raise Exception("ERROR: fuzzy set %s of linguistic variable %s does not exist" % (fs_name, variable_name))


    def set_variable(self, name, value, verbose=False):
        """
        Sets the numerical value of a linguistic variable.

        Args:
            name: name of the linguistic variables to be set.
            value: numerical value to be set.
            verbose: True/False, toggles verbose mode.
        """
        if self._sanitize_input: name = self._sanitize(name)
        try: 
            self._variables[name] = float(value)
            if verbose: print(" * Variable %s set to %f" % (name, value))
        except ValueError:
            raise Exception("ERROR: specified value for "+name+" is not an integer or float: "+value)


    def set_constant(self, name, value, verbose=False):
        """
        Sets the numerical value of a linguistic variable to a constant value (i.e. ignore fuzzy inference).

        Args:
            name: name of the linguistic variables to be set to a constant value.
            value: numerical value to be set.
            verbose: True/False, toggles verbose mode.
        """
        if self._sanitize_input: name = self._sanitize(name)
        try: 
            value = float(value)
            self._variables[name] = value
            self._constants.append(name)
            if verbose: print(" * Variable %s set to a constant value %f" % (name, value))
        except ValueError:
            raise Exception("ERROR: specified value for "+name+" is not an integer or float: "+value)

    def _check_templates(self):
        return self._templates_enabled

    def add_rules_from_file(self, path, verbose=False):
        """
        Imports new fuzzy rules by reading the strings from a text file.

        Args:
            path: path to the file containing the rules.
            verbose: True/False, toggles verbose mode.
        """
        if path[-3:].lower()!=".xls" and path[-4:].lower()!=".xlsx":
            with open(path) as fi:
                rules_strings = fi.readlines()
            self.add_rules(rules_strings, verbose=verbose)
        else:
            raise NotImplementedError("Excel support not available.")


    def _sanitize(self, rule):
        new_rule = "".join(ch for ch in rule if ch in valid_characters)
        return new_rule


    def add_rules(self, rules, verbose=False):
        """
        Adds new fuzzy rules to the fuzzy system.

        Args:
            rules: list of fuzzy rules to be added. Rules must be specified as strings, respecting Simpful's syntax.
            verbose: True/False, toggles verbose mode.
        """
        for rule in rules:
            
            # optional: remove invalid symbols
            if self._sanitize_input: rule = self._sanitize(rule)

            parsed_antecedent = recursive_parse(preparse(rule), verbose=verbose, operators=self._operators)
            parsed_consequent = postparse(rule, verbose=verbose)
            self._rules.append( [parsed_antecedent, parsed_consequent] )
            if verbose:
                print(" * Added rule IF", parsed_antecedent, "THEN", parsed_consequent)
                print()
        if verbose: print(" * %d rules successfully added" % len(rules))


    def get_rules(self):
        """
        Returns the rule base of the fuzzy system.

        Returns:
                a list containing the fuzzy rules as strings, in the same order they were added.
        """
        rule_base = []

        for r in self._rules:
            ant = str(r[0])
            ant = ant.replace("c.(", "(")
            ant = ant.replace("f.(", "(")
            ant = "IF "+ant

            cons = r[1]
            cons = " THEN ("+cons[0]+" IS "+cons[1]+")"

            rule = ant + cons
            rule_base.append(rule)

        return rule_base


    def replace_rule(self, i, new_rule, verbose=False):
        """
        Replaces the i-th rule in the FuzzySystem object. Rules are stored in the same order they were added. 

        Args:
            i: index of the rule to be replaced in the fuzzy system.
            new_rule: fuzzy rule to be used for the replacement.
            verbose: True/False, toggles verbose mode.
        """

        if len(self._rules)<i or i<0: 
            raise Exception("ERROR: rule number %d does not exist")

        if self._sanitize_input: 
            new_rule = self._sanitize(new_rule)
        parsed_antecedent = recursive_parse(preparse(new_rule), verbose=verbose, operators=self._operators)
        parsed_consequent = postparse(new_rule, verbose=verbose)
        self._rules[i]=[parsed_antecedent, parsed_consequent] 
        
        if verbose:
            print( " * Rule", i, "replaced with new rule: IF", parsed_antecedent, "THEN", parsed_consequent)
            print()


    def add_linguistic_variable(self, name, LV, verbose=False):
        """
        Adds a new linguistic variable to the fuzzy system.

        Args:
            name: string containing the name of the linguistic variable.
            LV: linguistic variable object to be added to the fuzzy system.
            verbose: True/False, toggles verbose mode.
        """
        if self._sanitize_input: name = self._sanitize(name)
        self._lvs[name]=deepcopy(LV)
        if self._lvs[name]._concept is None: 
            self._lvs[name]._concept = name
        # self._lvs[name]=deepcopy(LV)
        if verbose: print(" * Linguistic variable '%s' successfully added" % name)


    def set_crisp_output_value(self, name, value, verbose=False):
        """
        Adds a new crisp output value to the fuzzy system.

        Args:
            name: string containing the identifying name of the crisp output value.
            value: numerical value of the crisp output value to be added to the fuzzy system.
            verbose: True/False, toggles verbose mode.
        """
        if self._sanitize_input: name = self._sanitize(name)
        self._crispvalues[name]=value
        if verbose: print(" * Crisp output value for '%s' set to %f" % (name, value))
        self._set_model_type("Sugeno")


    def set_output_function(self, name, function, verbose=False):
        """
        Adds a new output function to the fuzzy system.

        Args:
            name: string containing the identifying name of the output function.
            function: string containing the output function to be added to the fuzzy system.
                The function specified in the string must use the names of linguistic variables contained in the fuzzy system object.
            verbose: True/False, toggles verbose mode.
        """
        if self._sanitize_input: name = self._sanitize(name)
        self._outputfunctions[name]=function
        if "{" in function:
            self._templates_enabled = TEMPLATES_ENGAGED
            if verbose: 
                print(" * Template system engaged")

        if verbose: print(" * Output function for '%s' set to '%s'" % (name, function))
        self._set_model_type("Sugeno")


    def _set_model_type(self, model_type):
        if self._detected_type == "inconsistent": return
        if self._detected_type is  None:
            self._detected_type = model_type
            print (" * Detected %s model type" % model_type )
        elif self._detected_type != model_type:
            print("WARNING: model type is unclear (simpful detected %s, but a %s output was specified)" % (self._detected_type, model_type))
            self._detected_type = 'inconsistent'

    def get_firing_strengths(self, input_values=None):
        """
            If "input_values" is not provided, it returns a list of the firing strengths of the the rules, 
            given the current state of input variables.
            "input_values" is an optional argument, in the form of a dictionary containing a list of input states for each variable.
            In this second case, it returns a 2D list of firing strengths of the given states.

            Args:
                input_values: dictionary where the keys are names of linguistic variables and the values are a list of input values for that variable.

            Returns:
                a list containing rules' firing strengths, or a 2D list containing rules' firing strengths for each given input state.
        """
        if input_values is None:
            results = [float(antecedent[0].evaluate(self)) for antecedent in self._rules]
            return results
        else:
            firings = []
            for i in range(0, len(list(input_values.values())[0])):
                for var in input_values.keys():
                    self.set_variable(var, input_values[var][i])
                res = self.get_firing_strengths()
                firings.append(res)
            return firings


    def _replace_values(self, function, verbose=False):
        res_string=function[:]
        newstring = ""
        while "{" in res_string:
            
            try:
                prestring = res_string[:res_string.find("{")]
                substring = res_string[res_string.find("{")+1:res_string.find("}")]
                if verbose: print("   Pre- and sub-strings:", prestring, substring)
            except:
                print("ERROR: missing curly brace in template, aborting.")
                exit()

            variable = substring[2: substring.find("IS")].strip()
            case = float(substring[substring.find("IS")+2:substring.find("THEN")].strip())
            value = substring[substring.find("THEN")+4:].strip()
            if verbose: print(" * Analysing rule: IF %s IS %s THEN %s" % (variable, case, value))

            # checking everything all the time is not a good idea, optimize later
            #detected = False


            candidate = self._variables[variable]
            if candidate==case:
                newstring  = newstring+prestring+ str(value)
                res_string = res_string[res_string.find("}")+1:]
                if verbose: print(" - case detected for '%s IS %s" % (variable, case))
            else:
                res_string = res_string[res_string.find("}")+1:]
                newstring += prestring + "0"
                if verbose: print(" - case NOT detected for '%s IS %s'" % (variable, case))

            """
            for k,v in self._replacement_dictionary.items():
                if k==variable and v==case:
                    if verbose: print(" - case detected for", k,v)
                    newstring = newstring+prestring+ str(value)
                    res_string = res_string[res_string.find("}")+1:]
                    detected = True
            """

            #if detected: continue

            

        newstring += res_string
        return newstring
       

    def mediate(self, outputs, antecedent, results, ignore_errors=False, ignore_warnings=False, verbose=False):

        final_result = {}

        list_crisp_values = [x[0] for x in self._crispvalues.items()]
        list_output_funs  = [x[0] for x in self._outputfunctions.items()]

        for output in outputs:
            """
            if verbose:
                print(" * Processing output for variable '%s'" %  output)
                print("   whose universe of discourse is:", self._lvs[output].get_universe_of_discourse())
                print("   contains the following fuzzy sets:", self._lvs[output]._FSlist )
            """

            num = 0
            den = 0
            
            for (ant, res) in zip(antecedent, results):
                outname = res[0]
                outterm = res[1]
                weight = float(res[2])
                crisp = True

                if verbose: 
                    print(" ** Rule composition:", ant, "->", res, ", output variable: '%s'" % outname, "with term: '%s'" % outterm)            

                if outname==output:
                    if outterm not in list_crisp_values:
                        crisp = False
                        if outterm not in list_output_funs:
                            raise Exception("ERROR: one rule calculates an output named '"
                                + outterm
                                + "', but I cannot find it among the output terms.\n"
                                + " --- PROBLEMATIC RULE:\n"
                                + "IF " + str(ant) + " THEN " + str(res))
                    if crisp:
                        crispvalue = self._crispvalues[outterm]
                    elif isinstance(self._outputfunctions[outterm], MF_object):
                        raise Exception("ERROR in consequent of rule %s.\nSugeno reasoning does not support output fuzzy sets." % ("IF " + str(ant) + " THEN " + str(res)))
                    else:
                        string_to_evaluate = self._outputfunctions[outterm]
                        
                        # replacement here
                        if self._check_templates() == TEMPLATES_ENGAGED:
                            string_to_evaluate = self._replace_values(string_to_evaluate, verbose=verbose)
                            if verbose:
                                print(" * Replacing templates in function for '%s'" % res[0])
                                print("   name of function: '%s'" % res[1])
                                print(" * Final version of the '%s' rule: %s" % (res[1], string_to_evaluate))


                        for k,v in self._variables.items():
                            # old version
                            # string_to_evaluate = string_to_evaluate.replace(k,str(v))

                            # match a variable name preceeded or followed by non-alphanumeric and _ characters
                            # substitute it with its numerical value
                            string_to_evaluate = re.sub(r"(?P<front>\W|^)"+k+r"(?P<end>\W|$)", r"\g<front>"+str(v)+r"\g<end>", string_to_evaluate)
                        crispvalue = eval(string_to_evaluate)                       

                    try:
                        value = ant.evaluate(self) 
                    except RuntimeError: 
                        raise Exception("ERROR: one rule could not be evaluated\n"
                        + " --- PROBLEMATIC RULE:\n"
                        + "IF " + str(ant) + " THEN " + str(res) + "\n")

                    temp = value*crispvalue*weight
                    num += temp
                    den += value

            try:
                if den == 0.0:
                    final_result[output] = 0.0
                    if not ignore_warnings:
                        print("WARNING: the sum of rules' firing for variable '%s' is equal to 0. The result of the Sugeno inference was set to 0." % output)
                else:
                    final_result[output] = num / den

            except ArithmeticError:
                if ignore_errors:
                    print("WARNING: cannot perform Sugeno inference for variable '%s'. The variable appears only as antecedent in the rules or an arithmetic error occurred." % output)
                else:
                    raise Exception("ERROR: cannot perform Sugeno inference for variable '%s'. The variable appears only as antecedent in the rules or an arithmetic error occurred." % output)
        
        return final_result

    def mediate_Mamdani(self, 
        outputs, 
        antecedent, 
        results, 
        ignore_errors=False, 
        ignore_warnings=False, 
        verbose=False, 
        subdivisions=1000,
        aggregation_function=max):

        final_result = {}

        for output in outputs:
            if verbose:
                print(" * Processing output for variable '%s'" %  output)
                print("   whose universe of discourse is:", self._lvs[output].get_universe_of_discourse())
                print("   contains the following fuzzy sets:", self._lvs[output]._FSlist )
            cuts_list = defaultdict(list)

            x0, x1 = self._lvs[output].get_universe_of_discourse()

            for (ant, res) in zip(antecedent, results):
                outname = res[0]
                outterm = res[1]
                weight = float(res[2])

                if verbose: 
                    print(" ** Rule composition:", ant, "->", res, ", output variable: '%s'" % outname, "with term: '%s'" % outterm)            

                if outname==output:
                    try:
                        value = ant.evaluate(self)
                    except RuntimeError: 
                        raise Exception("ERROR: one rule could not be evaluated\n"
                        + " --- PROBLEMATIC RULE:\n"
                        + "IF " + str(ant) + " THEN " + str(res) + "\n")

                    #degree of satisfaction of the rule is multiplied by rule weight
                    cuts_list[outterm].append(value*weight)

            values = []
            weightedvalues = []
            integration_points = linspace(x0, x1, subdivisions)

            convenience_dict = {}
            for k in cuts_list.keys():
                convenience_dict[k] = self._lvs[output].get_index(k)
            if verbose: print ( " * Indices:", convenience_dict)

            for u in integration_points:
                comp_values = []
                for k, v_list in cuts_list.items():
                    for v in v_list:
                        n = convenience_dict[k]                 
                        fs_term = self._lvs[output]._FSlist[n]
                        result = float(fs_term.get_value_cut(u, cut=v))
                        comp_values.append(result)
                #keep = max(comp_values)
                keep = aggregation_function(comp_values)
                values.append(keep)
                weightedvalues.append(keep*u)

            sumwv = sum(weightedvalues)
            sumv = sum(values)
            
            try:
                if sumv == 0.0:
                    CoG = 0
                    if not ignore_warnings:
                        print("WARNING: the sum of rules' firing for variable '%s' is equal to 0. The result of the Mamdani inference was set to 0." % output)
                else:
                    CoG = sumwv/sumv
            
            except ArithmeticError:
                if ignore_errors:
                    print("WARNING: cannot perform Mamdani inference for variable '%s'. The variable appears only as antecedent in the rules or an arithmetic error occurred." % output)
                else:
                    raise Exception("ERROR: cannot perform Mamdani inference for variable '%s'. The variable appears only as antecedent in the rules or an arithmetic error occurred." % output)

            if verbose: print (" * Weighted values: %.2f\tValues: %.2f\tCoG: %.2f"% (sumwv, sumv, CoG))
            final_result[output] = CoG 

        return final_result


    def Sugeno_inference(self, terms=None, ignore_errors=False, ignore_warnings=False, verbose=False):
        """
        Performs Sugeno fuzzy inference.

        Args:
            terms: list of the names of the variables on which inference must be performed. If empty, all variables appearing in the consequent of a fuzzy rule are inferred.
            ignore_errors: True/False, toggles the raising of errors during the inference.
            ignore_warnings: True/False, toggles the raising of warnings during the inference.
            verbose: True/False, toggles verbose mode.

        Returns:
            a dictionary, containing as keys the variables' names and as values their numerical inferred values.
        """
        if self._sanitize_input and terms is not None: 
            terms = [self._sanitize(term) for term in terms]
        
        # default: inference on ALL rules/terms
        if terms == None:
            temp = [rule[1][0] for rule in self._rules] 
            terms = list(set(temp))
        else:
            # get rid of duplicates in terms to infer
            terms = list(set(terms))
            for t in terms:
                if t not in set([rule[1][0] for rule in self._rules]):
                    raise Exception("ERROR: Variable "+t+" does not appear in any consequent.")

        array_rules = array(self._rules, dtype='object')
        # cheking if the rule base is weighted, if not add dummy weights
        for n, cons in enumerate(array_rules.T[1]):
            if len(cons)<3:
                cons = cons + ("1.0",)
                array_rules.T[1][n] = cons

        if len(self._constants)==0:
            result = self.mediate(terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors, ignore_warnings=ignore_warnings, verbose=verbose)
        else:
            #remove constant variables from list of variables to infer
            ncost_terms = [t for t in terms if t not in self._constants]
            result = self.mediate(ncost_terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors, ignore_warnings=ignore_warnings, verbose=verbose)
            #add values of constant variables
            cost_terms = [t for t in terms if t in self._constants]
            for name in cost_terms:
                result[name] = self._variables[name]
        
        return result


    def Mamdani_inference(self, terms=None, subdivisions=1000, aggregation_function=max, ignore_errors=False, ignore_warnings=False, verbose=False):
        """
        Performs Mamdani fuzzy inference.

        Args:
            terms: list of the names of the variables on which inference must be performed.If empty, all variables appearing in the consequent of a fuzzy rule are inferred.
            subdivisions: the number of integration steps to be performed for calculating fuzzy set area (default: 1000).
            aggregation_function: pointer to function used to aggregate fuzzy sets during Mamdani inference, default is max. Use Python sum function, or simpful's probor function for sum and probabilistic OR, respectively.
            ignore_errors: True/False, toggles the raising of errors during the inference.
            ignore_warnings: True/False, toggles the raising of warnings during the inference.
            verbose: True/False, toggles verbose mode.

        Returns:
            a dictionary, containing as keys the variables' names and as values their numerical inferred values.
        """
        if self._sanitize_input and terms is not None: 
            terms = [self._sanitize(term) for term in terms]
        
        # default: inference on ALL rules/terms
        if terms == None:
            temp = [rule[1][0] for rule in self._rules] 
            terms= list(set(temp))
        else:
            # get rid of duplicates in terms to infer
            terms = list(set(terms))
            for t in terms:
                if t not in set([rule[1][0] for rule in self._rules]):
                    raise Exception("ERROR: Variable "+t+" does not appear in any consequent.")

        array_rules = array(self._rules, dtype=object)
        # cheking if the rule base is weighted, if not add dummy weights
        for n, cons in enumerate(array_rules.T[1]):
            if len(cons)<3:
                cons = cons + ("1.0",)
                array_rules.T[1][n] = cons
        
        if len(self._constants)==0:
            result = self.mediate_Mamdani(terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors, ignore_warnings=ignore_warnings, verbose=verbose, subdivisions=subdivisions, aggregation_function=aggregation_function)
        else:
            #remove constant variables from list of variables to infer
            ncost_terms = [t for t in terms if t not in self._constants]
            result = self.mediate_Mamdani(ncost_terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors, ignore_warnings=ignore_warnings, verbose=verbose, subdivisions=subdivisions, aggregation_function=aggregation_function)
            #add values of constant variables
            cost_terms = [t for t in terms if t in self._constants]
            for name in cost_terms:
                result[name] = self._variables[name]

        return result


    def probabilistic_inference(self, terms=None, ignore_errors=False, ignore_warnings=False, verbose=False):
        raise NotImplementedError()


    def inference(self, terms=None, subdivisions=1000, aggregation_function=max, ignore_errors=False, ignore_warnings=False, verbose=False):
        """
        Performs the fuzzy inference, trying to automatically choose the correct inference engine.

        Args:
            terms: list of the names of the variables on which inference must be performed. If empty, all variables appearing in the consequent of a fuzzy rule are inferred.
            subdivisions: the number of integration steps to be performed for calculating fuzzy set area (default: 1000).
            aggregation_function: pointer to function used to aggregate fuzzy sets during Mamdani inference, default is max. Use Python sum function, or simpful's probor function for sum and probabilistic OR, respectively.
            ignore_errors: True/False, toggles the raising of errors during the inference.
            ignore_warnings: True/False, toggles the raising of warnings during the inference.
            verbose: True/False, toggles verbose mode.

        Returns:
            a dictionary, containing as keys the variables' names and as values their numerical inferred values.
        """ 
        if self._detected_type == "Sugeno":
            return self.Sugeno_inference(terms=terms, ignore_errors=ignore_errors, ignore_warnings=ignore_warnings, verbose=verbose)
        elif self._detected_type == "probabilistic":
            return self.probabilistic_inference(terms=terms, ignore_errors=ignore_errors, ignore_warnings=ignore_warnings, verbose=verbose)
        elif self._detected_type is None: # default
            return self.Mamdani_inference(terms=terms, subdivisions=subdivisions, ignore_errors=ignore_errors, ignore_warnings=ignore_warnings, verbose=verbose, aggregation_function=aggregation_function)
        else:
            raise Exception("ERROR: simpful could not detect the model type, please use either Sugeno_inference() or Mamdani_inference() methods.")
            

    def plot_variable(self, var_name, outputfile="", TGT=None, element=None, highlight=None, ax=None, xscale="linear"):
        """
        Plots all fuzzy sets contained in a liguistic variable. Options for saving the figure and draw on a matplotlib ax are provided.

        Args:
            var_name: string containing the name of the linguistic variable to plot.
            outputfile: string containing path and filename where the plot must be saved.
            TGT: (deprecated) show the memberships of a specific element of discourse TGT in the figure.
            element: show the memberships of a specific element of discourse in the figure.
            highlight: string, indicating the linguistic term/fuzzy set to highlight in the plot. 
            ax: a matplotlib ax where the variable will be plotted.
            xscale: default "linear", supported scales "log". Changes the scale of the xaxis.
        """
        if matplotlib == False:
            raise Exception("ERROR: please, install matplotlib for plotting facilities")
        if TGT is not None:
            print("Warning: argument 'TGT' is deprecated and will be discontinued in future relseases. Please use 'element'")
            element = TGT

        if ax != None:
            ax = self._lvs[var_name].draw(ax=ax, element=element, highlight=highlight, xscale=xscale)
            return ax
        self._lvs[var_name].plot(outputfile=outputfile, element=element, highlight=highlight, xscale=xscale)


    def produce_figure(self, outputfile="", max_figures_per_row=4, element_dict=None):
        """
        Plots the membership functions of each linguistic variable contained in the fuzzy system.

        Args:
            outputfile: string containing path and filename where the plot must be saved.
            max_figures_per_row: maximum number of figures per row in the plot.
            element_dict: dictionary of elements of the universe of discourse whose membership must be plotted over the fuzzy sets.
        """
        if matplotlib == False:
            raise Exception("ERROR: please, install matplotlib for plotting facilities")

        # use a defaultdict in case no target was specified
        if element_dict is None: 
            element_dict = defaultdict(lambda: None)

        num_ling_variables = len(self._lvs)
        if num_ling_variables == 0:
            raise Exception("ERROR: no linguistic variables added. Please add at least one variable to plot")
        #print(" * Detected %d linguistic variables" % num_ling_variables)
        columns = min(num_ling_variables, max_figures_per_row)
        if num_ling_variables>max_figures_per_row:
            if num_ling_variables%max_figures_per_row==0:
                rows = num_ling_variables//max_figures_per_row 
            else:
                rows = num_ling_variables//max_figures_per_row + 1
        else:
            rows = 1

        fig, ax = subplots(rows, columns, figsize=(columns*5, rows*5))

        if rows==1: ax = [ax]
        if columns==1: ax= [ax]

        n = 0
        for k, v in self._lvs.items():
            r = n%max_figures_per_row
            c = n//max_figures_per_row
            v.draw(ax[c][r], element=element_dict[k])
            ax[c][r].set_ylim(0,1.05)
            n+=1

        for m in range(n, columns*rows):
            r = m%max_figures_per_row
            c = m//max_figures_per_row
            ax[c][r].axis('off')

        fig.tight_layout()

        if outputfile != "":
            fig.savefig(outputfile)
        else:
            show()


    def plot_surface(self, variables, output, detail=40, color_map="plasma", outputfile=""):
        """
        Plots the surface induced by the rules.

        Args:
            variables: a list containing a pair of linguistic variables' names (as strings) for the x and y axis.
            output: the name of the output variable (as string) to be computed.
            detail: number of subdivisions along each axis.
            color_map: the color map to be used for the plot.
            outputfile: string containing path and filename where the plot must be saved.

        Returns:
            a matplotlib figure object.
        """ 


        if len(variables)!=2: 
            print("ERROR: please specify the two variables for the surface plot")
            return None

        v1, v2 = variables

        min_v1, max_v1 = self._lvs[v1].get_universe_of_discourse()
        min_v2, max_v2 = self._lvs[v2].get_universe_of_discourse()

        A = linspace(min_v1, max_v1, detail)
        B = linspace(min_v2, max_v2, detail)
        C = []

        for a in A:
            temp = []
            for b in B:
                self.set_variable(v1, a)
                self.set_variable(v2, b)
                res = self.inference()[output]
                temp.append( res )
            C.append(temp)
        C = array(C)

        A,B = meshgrid(A,B)

        fig = figure(figsize=(8,6))
        ax = axes(projection='3d')

        v = ax.plot_surface(A,B,C, shade=True, cmap=color_map)
        ax.set_xlabel(self._lvs[v1]._concept)
        ax.set_ylabel(self._lvs[v2]._concept)
        ax.set_zlabel(output)
        colorbar(v, ax=ax)
        fig.tight_layout()

        if outputfile != "":
            fig.savefig(outputfile)
        else:
            show()
        return fig


    def aggregate(self, list_variables, function):
        """
        Performs a fuzzy aggregation of linguistic variables contained in a FuzzySystem object.

        Args:
            list_variables: list of linguistic variables names in the FuzzySystem object to aggregate.
            function: pointer to an aggregation function. The function must accept as an argument a list of membership values.

        Returns:
            the aggregated membership values.
        """ 
        memberships = []
        for variable, fuzzyset in list_variables.items():
            value = self._variables[variable]
            result = self._lvs[variable].get_values(value)[fuzzyset]
            memberships.append(result)
        return function(memberships)

# useful pre-implemented functions for aggregation or implication during fuzzy inference
def prod(m_list):
    """
        Performs aggregation of membership values using the product operation.

        Args:
            m_list: list of membership values to aggregate.

        Returns:
            the aggregated membership value.
    """ 
    return prod(m_list)

def probor(m_list):
    """
        Performs aggregation of membership values using the probabilistic OR operation.

        Args:
            m_list: list of membership values to aggregate.

        Returns:
            the aggregated membership value.
    """
    res = m_list[0]
    if len(m_list) == 1:
        return m_list[0]
    for i in range(1, len(m_list)):
        res = res + m_list[i] - res * m_list[i]
    return res

if __name__ == '__main__':
    pass
