from .fuzzy_sets import FuzzySet, MF_object, Sigmoid_MF, InvSigmoid_MF, Gaussian_MF, InvGaussian_MF, DoubleGaussian_MF, Triangular_MF, Trapezoidal_MF
from .rule_parsing import curparse, preparse, postparse
from numpy import array, linspace
from scipy.interpolate import interp1d
from copy import deepcopy
from collections import defaultdict, OrderedDict
import re
import string
try:
	import seaborn as sns
except ImportError:
	pass

# constant values
linestyles= ["-", "--", ":", "-."]

# for sanitization
valid_characters = string.ascii_letters + string.digits + "()_ "


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
			print("ERROR: please specify at least one fuzzy set")
			exit(-2)
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


	def draw(self, ax, TGT=None, highlight=None):
		"""
		This method returns a matplotlib ax, representing all fuzzy sets contained in the liguistic variable.

		Args:
			ax: the axis to plot to.
			TGT: show the memberships of a specific element of discourse TGT in the figure.
			highlight: string, indicating the linguistic term/fuzzy set to highlight in the plot.
		Returns:
			A matplotlib axis, representing all fuzzy sets contained in the liguistic variable.
		"""
		mi, ma = self.get_universe_of_discourse()
		x = linspace(mi, ma, 10000)

		
		if highlight is None:
			linestyles= ["-", "--", ":", "-."]
		else:
			linestyles= ["-"]*4


		for nn, fs in enumerate(self._FSlist):
			if fs._type == "function":
				y = [fs.get_value(xx) for xx in x]
				color = None
				lw = 1

				if highlight==fs._term: 
					color="red"
					lw =5 
				elif highlight is not None:
					color="lightgray"
				ax.plot(x,y, linestyles[nn%4], lw=lw, label=fs._term, color=color)
			else:
				sns.regplot(x=fs._points.T[0], y=fs._points.T[1], marker="d", color="red", fit_reg=False, ax=ax)
				f = interp1d(fs._points.T[0], fs._points.T[1], bounds_error=False, fill_value=(fs.boundary_values[0], fs.boundary_values[1]))
				ax.plot(x, f(x), linestyles[nn%4], label=fs._term,)
		if TGT is not None:
			ax.axvline(x=TGT, ymin=0.0, ymax=1.0, color="red", linestyle="--", linewidth=2.0)
		ax.set_xlabel(self._concept)
		ax.set_ylabel("Membership degree")
		if highlight is None: ax.legend(loc="best")
		return ax


	def plot(self, outputfile="", TGT=None, highlight=None):
		"""
		Shows a plot representing all fuzzy sets contained in the liguistic variable.

		Args:
			outputfile: path and filename where the plot must be saved.
			TGT: show the memberships of a specific element of discourse TGT in the figure.
			highlight: string, indicating the linguistic term/fuzzy set to highlight in the plot.
		"""
		try:
			from matplotlib.pyplot import plot, show, title, subplots, legend
			try:
				import seaborn as sns
			except ImportError:
				pass
		except ImportError:
			raise Exception("ERROR: please, install matplotlib for plotting facilities")

		fig, ax = subplots(1,1)
		self.draw(ax=ax, TGT=TGT, highlight=highlight)

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
		print(" Created by Marco S. Nobile (m.s.nobile@tue.nl)")
		print(" and Simone Spolaor (simone.spolaor@unimib.it)")
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
			if fs._term==fs_name:		return fs

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
			value = float(value)
			self._variables[name] = value
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

	def add_rules_from_file(self, path, verbose=False):
		"""
		Imports new fuzzy rules by reading the strings from a text file.
		"""
		if path[-3:].lower()!=".xls" and path[-4:].lower()!=".xlsx":
			with open(path) as fi:
				rules_strings = fi.readlines()
			self.add_rules(rules_strings, verbose=verbose)
		else:
			raise NotImplementedError("Excel support not available yet.")


	def _sanitize(self, rule):
		new_rule = "".join(ch for ch in rule if ch in valid_characters)
		return new_rule


	def add_rules(self, rules, verbose=False):
		"""
		Adds new fuzzy rules to the fuzzy system.

		Args:
			rules: list of fuzzy rules to be added. Rules must be specified as strings, respecting Simpful's syntax.
			sanitize: True/False, automatically removes non alphanumeric symbols from rules.
			verbose: True/False, toggles verbose mode.
		"""
		for rule in rules:
			
			# optional: remove invalid symbols
			if self._sanitize_input: rule = self._sanitize(rule)

			parsed_antecedent = curparse(preparse(rule), verbose=verbose, operators=self._operators)
			parsed_consequent = postparse(rule, verbose=verbose)
			self._rules.append( [parsed_antecedent, parsed_consequent] )
			if verbose:
				print(" * Added rule IF", parsed_antecedent, "THEN", parsed_consequent)
				print()
		if verbose: print(" * %d rules successfully added" % len(rules))


	def add_linguistic_variable(self, name, LV, verbose=False):
		"""
		Adds a new linguistic variable to the fuzzy system.

		Args:
			name: string containing the name of the linguistic variable.
			LV: linguistic variable object to be added to the fuzzy system.
			verbose: True/False, toggles verbose mode.
		"""
		if self._sanitize_input: name = self._sanitize(name)
		if LV._concept is None: 
			LV._concept = name
		self._lvs[name]=deepcopy(LV)
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
		if verbose: print(" * Output function for '%s' set to '%s'" % (name, function))
		self._set_model_type("Sugeno")

	def _set_model_type(self, model_type):
		if self._detected_type == "inconsistent": return
		if self._detected_type is  None:
			self._detected_type = model_type
			print (" * Detected %s model type" % model_type )
		elif self._detected_type != model_type:
			print("WARNING: model type is unclear (simpful detected %s, but I received a %s output)" % (self._detected_type, model_type))
			self._detected_type = 'inconsistent'

	def get_firing_strengths(self):
		"""
			Returns a list of the firing strengths of the the rules, 
			given the current state of input variables.

			Returns:
				a list containing rules' firing strengths.
		"""
		results = [float(antecedent[0].evaluate(self)) for antecedent in self._rules]
		return results



	def mediate(self, outputs, antecedent, results, ignore_errors=False):

		final_result = {}

		list_crisp_values = [x[0] for x in self._crispvalues.items()]
		list_output_funs  = [x[0] for x in self._outputfunctions.items()]

		for output in outputs:
			num = 0
			den = 0
			
			for (ant, res) in zip(antecedent, results):
				outname = res[0]
				outterm = res[1]
				crisp = True
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

					temp = value*crispvalue
					num += temp
					den += value

			try:
				if den == 0.0:
					final_result[output] = 0.0
					print("WARNING: the sum of rules' firing for variable '%s' is equal to 0. The result of the Sugeno inference was set to 0." % output)
				else:
					final_result[output] = num / den

			except ArithmeticError:
				if ignore_errors==True:
					print("WARNING: cannot perform Sugeno inference for variable '%s'. The variable appears only as antecedent in the rules or an arithmetic error occurred." % output)
				else:
					raise Exception("ERROR: cannot perform Sugeno inference for variable '%s'. The variable appears only as antecedent in the rules or an arithmetic error occurred." % output)
		
		return final_result


	def mediate_Mamdani(self, outputs, antecedent, results, ignore_errors=False, verbose=False, subdivisions=1000):

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

				if verbose:	
					print(" ** Rule composition:", ant, "->", res, ", output variable: '%s'" % outname, "with term: '%s'" % outterm)			

				if outname==output:

					try:
						value = ant.evaluate(self) 
					except RuntimeError: 
						raise Exception("ERROR: one rule could not be evaluated\n"
						+ " --- PROBLEMATIC RULE:\n"
						+ "IF " + str(ant) + " THEN " + str(res) + "\n")

					cuts_list[outterm].append(value)

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
				keep = max(comp_values)
				values.append(keep)
				weightedvalues.append(keep*u)

			sumwv = sum(weightedvalues)
			sumv = sum(values)
			
			if sumv == 0.0:
				CoG = 0
				print("WARNING: the sum of rules' firing for variable '%s' is equal to 0. The result of the Mamdani inference was set to 0." % output)
			else:
				CoG = sumwv/sumv
			
			if verbose: print (" * Weighted values: %.2f\tValues: %.2f\tCoG: %.2f"% (sumwv, sumv, CoG))
			
			final_result[output] = CoG 

		return final_result


	def Sugeno_inference(self, terms=None, ignore_errors=False, verbose=False):
		"""
		Performs Sugeno fuzzy inference.

		Args:
			terms: list of the names of the variables on which inference must be performed. If empty, all variables appearing in the consequent of a fuzzy rule are inferred.
			ignore_errors: True/False, toggles the raising of errors during the inference.
			verbose: True/False, toggles verbose mode.

		Returns:
			a dictionary, containing as keys the variables' names and as values their numerical inferred values.
		"""
		if self._sanitize and terms is not None: 
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
		if len(self._constants)==0:
			result = self.mediate(terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors)
		else:
			#remove constant variables from list of variables to infer
			ncost_terms = [t for t in terms if t not in self._constants]
			result = self.mediate(ncost_terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors)
			#add values of constant variables
			cost_terms = [t for t in terms if t in self._constants]
			for name in cost_terms:
				result[name] = self._variables[name]
		
		return result


	def Mamdani_inference(self, terms=None, ignore_errors=False, verbose=False, subdivisions=1000):
		"""
		Performs Mamdani fuzzy inference.

		Args:
			terms: list of the names of the variables on which inference must be performed. If empty, all variables appearing in the consequent of a fuzzy rule are inferred.
			subdivisions: the number of integration steps to be performed (default: 1000).
			ignore_errors: True/False, toggles the raising of errors during the inference.
			verbose: True/False, toggles verbose mode.

		Returns:
			a dictionary, containing as keys the variables' names and as values their numerical inferred values.
		"""
		if self._sanitize and terms is not None: 
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
		if len(self._constants)==0:
			result = self.mediate_Mamdani(terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors, verbose=verbose , subdivisions=subdivisions)
		else:
			#remove constant variables from list of variables to infer
			ncost_terms = [t for t in terms if t not in self._constants]
			result = self.mediate_Mamdani(ncost_terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors, verbose=verbose , subdivisions=subdivisions)
			#add values of constant variables
			cost_terms = [t for t in terms if t in self._constants]
			for name in cost_terms:
				result[name] = self._variables[name]

		return result


	def probabilistic_inference(self, terms=None, ignore_errors=False, verbose=False):
		raise NotImplementedError()


	def inference(self, terms=None, ignore_errors=False, verbose=False, subdivisions=1000):
		"""
		Performs the fuzzy inference, trying to automatically choose the correct inference engine.

		Args:
			terms: list of the names of the variables on which inference must be performed. If empty, all variables appearing in the consequent of a fuzzy rule are inferred.
			ignore_errors: True/False, toggles the raising of errors during the inference.
			verbose: True/False, toggles verbose mode.
			subdivisions: set the number of integration steps to be performed by Mamdani inference (default: 1000).

		Returns:
			a dictionary, containing as keys the variables' names and as values their numerical inferred values.
		""" 
		if self._detected_type == "Sugeno":
			return self.Sugeno_inference(terms=terms, ignore_errors=ignore_errors, verbose=verbose)
		elif self._detected_type == "probabilistic":
			return self.probabilistic_inference(terms=terms, ignore_errors=ignore_errors, verbose=verbose)
		elif self._detected_type is None: # default
			return self.Mamdani_inference(terms=terms, ignore_errors=ignore_errors, verbose=verbose, subdivisions=subdivisions)
		else:
			raise Exception("ERROR: simpful could not detect the model type, please use either Sugeno_inference() or Mamdani_inference() methods.")
			

	def plot_variable(self, var_name, outputfile="", TGT=None, highlight=None):
		"""
		Plots all fuzzy sets contained in a liguistic variable. An option for saving the figure is provided.

		Args:
			var_name: string containing the name of the linguistic variable to plot.
			outputfile: path and filename where the plot must be saved.
			TGT: a specific element of the universe of discourse to be highlighted in the figure. 
			highlight: string, indicating the linguistic term/fuzzy set to highlight in the plot. 
		"""
		self._lvs[var_name].plot(outputfile=outputfile, TGT=TGT, highlight=highlight)


	def produce_figure(self, outputfile='output.pdf', max_figures_per_row=4):
		"""
		Plots the membership functions of each linguistic variable contained in the fuzzy system.

		Args:
			outputfile: path and filename where the plot must be saved.
		"""

		from matplotlib.pyplot import subplots

		num_ling_variables = len(self._lvs)
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
			v.draw(ax[c][r])
			ax[c][r].set_ylim(0,1.05)
			n+=1

		for m in range(n, columns*rows):
			r = m%max_figures_per_row
			c = m//max_figures_per_row
			ax[c][r].axis('off')

		fig.tight_layout()
		fig.savefig(outputfile)


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

if __name__ == '__main__':
	pass