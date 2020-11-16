import operator
from .fuzzy_sets import FuzzySet, MF_object, Sigmoid_MF, InvSigmoid_MF, Gaussian_MF, InvGaussian_MF, DoubleGaussian_MF, Triangular_MF, Trapezoidal_MF
from .rule_parsing import curparse, preparse, postparse
from .rules import RuleGen
from numpy import array, linspace
from scipy.interpolate import interp1d
from copy import deepcopy
from collections import defaultdict
import numpy as np
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
			the two extreme values of the universe of discourse
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


	def draw(self, ax, TGT=None):
		"""
		This method returns a matplotlib ax, representing all fuzzy sets contained in the liguistic variable.

		Args:
			ax: the axis to plot to.
			TGT: show the memberships of a specific element of discourse TGT in the figure. 
		Returns:
			A matplotlib axis, representing all fuzzy sets contained in the liguistic variable.
		"""
		mi, ma = self.get_universe_of_discourse()
		x = linspace(mi, ma, 10000)

		linestyles= ["-", "--", ":", "-."]

		for nn, fs in enumerate(self._FSlist):
			if fs._type == "function":
				y = [fs.get_value(xx) for xx in x]
				ax.plot(x,y, linestyles[nn%4], label=fs._term, )
			else:
				sns.regplot(fs._points.T[0], fs._points.T[1], marker="d", color="red", fit_reg=False, ax=ax)
				f = interp1d(fs._points.T[0], fs._points.T[1], bounds_error=False, fill_value=(0,0))
				ax.plot(x, f(x), linestyles[nn%4], label=fs._term,)
				if TGT is not None:
					ax.plot(TGT, f(TGT), "*", ms=10, label="x")
		ax.set_xlabel(self._concept)
		ax.set_ylabel("Membership degree")
		ax.legend(loc="best")
		return ax


	def plot(self, TGT=None):
		"""
		Shows a plot representing all fuzzy sets contained in the liguistic variable.

		Args:
			TGT: show the memberships of a specific element of discourse TGT in the figure. 
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
		self.draw(ax=ax, TGT=TGT)
		show()
		
		

	def __repr__(self):
		if self._concept is None:
			text = "N/A"
		else:
			text = self._concept
		return "L.V.: "+text


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
			operators: a list of strings, specifying fuzzy operators to be used instead of defaults.
			Currently supported operators: 'AND_PRODUCT'.
			show_banner: True/False, toggles display of banner.
			verbose: True/False, toggles verbose mode.
	"""

	def __init__(self,  operators=None, show_banner=True, sanitize_input=False, verbose=False):

		self._rules = []
		self._lvs = {}
		self._variables = {}
		self._crispvalues = {}
		self._outputfunctions = {}
		self._outputfuzzysets = {}
		if show_banner: self._banner()
		self._operators = operators
		self._sanitize_input = sanitize_input
		if sanitize_input and verbose:
			print (" * Warning: Simpful rules sanitization is enabled, please pay attention to possible collisions of symbols.")

	def _banner(self):
		import pkg_resources
		vrs = pkg_resources.get_distribution('simpful').version 
		print("  ____  __  _  _  ____  ____  _  _  __   ")
		print(" / ___)(  )( \\/ )(  _ \\(  __)/ )( \\(  ) v%s " % vrs)
		print(" \\___ \\ )( / \\/ \\ ) __/ ) _) ) \\/ (/ (_/\\ ")
		print(" (____/(__)\\_)(_/(__)  (__)  \\____/\\____/")
		print()
		print(" Created by Marco S. Nobile (m.s.nobile@tue.nl)")
		print(" and Simone Spolaor (simone.spolaor@disco.unimib.it)")
		print()


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
			sanitize: True/False, automatically removes non alphanumeric symbols from rules
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
			This method returns a list of the firing strengths of the the rules, 
			given the current state of input variables.

			Returns:
				a list containing rules' firing strengths
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
							string_to_evaluate = string_to_evaluate.replace(k,str(v))
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
			cuts_list = defaultdict()

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

					cuts_list[outterm] = value

			values = []
			weightedvalues = []
			integration_points = linspace(x0, x1, subdivisions)

			convenience_dict = {}
			for k in cuts_list.keys():
				convenience_dict[k] = self._lvs[output].get_index(k)
			if verbose: print ( " * Indices:", convenience_dict)

			for u in integration_points:
				#print ("x=%.1f" % u)
				comp_values = []
				for k,v in cuts_list.items():
					# result = float(self._outputfuzzysets[k].get_value_cut(u, cut=v))
					n = convenience_dict[k]					
					fs_term = self._lvs[output]._FSlist[n]
					result = float(fs_term.get_value_cut(u, cut=v))
					comp_values.append(result)
				keep = max(comp_values)
				values.append(keep)
				weightedvalues.append(keep*u)

			sumwv = sum(weightedvalues); sumv = sum(values)
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
		# default: inference on ALL rules/terms
		if self._sanitize and terms is not None: 
			terms = [self._sanitize(term) for term in terms]
		if terms == None:
			temp = [rule[1][0] for rule in self._rules] 
			terms= list(set(temp))

		array_rules = array(self._rules, dtype='object')
		result = self.mediate( terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors )
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

		# default: inference on ALL rules/terms
		if self._sanitize and terms is not None: 
			terms = [self._sanitize(term) for term in terms]
		if terms == None:
			temp = [rule[1][0] for rule in self._rules] 
			terms= list(set(temp))

		array_rules = array(self._rules, dtype=object)
		result = self.mediate_Mamdani( terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors, verbose=verbose , subdivisions=subdivisions)
		return result


	def inference(self, terms=None, ignore_errors=False, verbose=False, subdivisions=1000, return_class = False):
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
			return ProbaFuzzySystem.probabilistic_inference(ignore_errors=ignore_errors, verbose=verbose, return_class = return_class)
		elif self._detected_type is None: # default
			return self.Mamdani_inference(terms=terms, ignore_errors=ignore_errors, verbose=verbose, subdivisions=subdivisions)
		else:
			raise Exception("ERROR: simpful could not detect the model type, please use either Sugeno_inference() or Mamdani_inference() methods.")
			
	def produce_figure(self, outputfile='output.pdf'):
		"""
		Plots the membership functions of each linguistic variable contained in the fuzzy system.

		Args:
			outputfile: path and filename where the plot must be saved.
		"""

		from matplotlib.pyplot import subplots

		num_ling_variables = len(self._lvs)
		#print(" * Detected %d linguistic variables" % num_ling_variables)
		columns = min(num_ling_variables, 4)
		if num_ling_variables>4:
			rows = num_ling_variables//4 + 1
		else:
			rows = 1

		fig, ax = subplots(rows, columns, figsize=(columns*5, rows*5))

		if rows==1: ax = [ax]
		if columns==1: ax= [ax]

		n = 0
		for k, v in self._lvs.items():
			r = n%4
			c = n//4
			v.draw(ax[c][r])
			ax[c][r].set_ylim(0,1)
			n+=1

		for m in range(n, columns*rows):
			r = m%4
			c = m//4
			ax[c][r].axis('off')

		fig.tight_layout()
		fig.savefig(outputfile)


class ProbaFuzzySystem(FuzzySystem, RuleGen):

	def __init__(self, consequents=None, var_names=None, centers=None, widths=None,
              X=None,  y=None, probas=None, threshold=None, generateprobas=False,
              operators=['AND_p', 'OR', 'AND', 'NOT'], ops=['AND_p', 'OR', 'AND'],
              all_var_names=None):

		FuzzySystem.__init__(self,  operators=None, show_banner=True,
		                     sanitize_input=False, verbose=False)
		RuleGen.__init__(self, cluster_centers=centers, var_names=var_names, n_consequents=consequents, threshold=threshold,
                   probas=probas, generateprobas=generateprobas, operators=operators, ops=ops, all_var_names=all_var_names)

		self.raw_rules=None
		self._detected_type = None
		self._X = X
		self.y = y
		self.var_names = var_names
		self.centers = centers
		self.widths = widths
		self.A = []
		self.just_beta = None
		self.probas_ = None
		self.__estimate = False
#		self._probas = self.estimate_probas() if probas is None else probas
	
	def router(self):
		if self._rules[0][1][0] > 0 and self._rules[0][1][1]==True:
			self.__estimate = True

	def add_proba_rules(self, rules, verbose=False):
		""" Works in a similarly to the normal add_rules method. Will take a list of rules and extract its Clauses.
		In addition to this it will also extract the probabilities of each rule.

		Args:
			rules (list): Need to respect probabilistic Syntax. E.g. sum of probabilities should be close to 1. 
			For an example please refer to the readme file.
			verbose (bool, optional): Will print out the parsed antecedent and consequent. Defaults to False.
		"""
		self.raw_rules = rules

		for rule in rules:
			parsed_antecedent = curparse(
				preparse(rule), verbose=verbose, operators=self._operators)
			consequent = postparse(rule)
			parsed_consequent = np.array(consequent)
			self._rules.append([parsed_antecedent, parsed_consequent])
		
		self.router()

		self._set_model_type('probabilistic')
		if verbose:
			print(" * Added rule IF", parsed_antecedent,
			      "THEN", parsed_consequent, '\n')
		if verbose:
			print(" * %d rules successfully added" % len(rules))

	def add_linguistic_variables(self):
		#Setup fuzzysets
		var_names = self.var_names
		for i, ling_var in enumerate(var_names):
			#Construct fuzzy sets
			fuzzysets = []
			for rulenr in range(len(self._rules)):
				fuzzyset = FuzzySet(
					function=Gaussian_MF(
						self.centers[rulenr, i],
						self.widths[rulenr, i]
					),
					term='cluster{}'.format(rulenr))
				fuzzysets.append(fuzzyset)

			#Add linguistic variable to fuzzy system
			MF_ling_var = LinguisticVariable(
				fuzzysets, concept=ling_var, universe_of_discourse=[-10, 10])
			self.add_linguistic_variable(ling_var, MF_ling_var)

	def proba_zero_order(self):
		for i in range(len(self._probas)):
			self.set_crisp_output_value('fun{}'.format(i), self._probas[i])


	def mediate_probabilistic(self, probs):
		""" Performs probabilistic inference. This method gets the firing strengths of each rule 
		and normalizes these outputs. This way we can see how much
		more an instance triggers each rule. It will return the probabilities for each class. 

		Args:
			probs: probabilities parsed from the given rules.

		Returns:
			<class 'numpy.ndarray'>: An ndarray containing the probabilties for each class.
		"""
		rule_outputs = np.array(self.get_firing_strengths())
		normalized_activation_rule = np.divide(rule_outputs, np.sum(rule_outputs))
		return np.matmul(normalized_activation_rule, probs)

	def prepare_a(self):
		""" Performs probabilistic inference. This method gets the firing strengths of each rule 
		and normalizes these outputs. This way we can see how much
		more an instance triggers each rule. It will return the probabilities for each class. 

		Args:
			probs: probabilities parsed from the given rules.

		Returns:
			<class 'numpy.ndarray'>: An ndarray containing the probabilties for each class.
		"""

		for instance in self._X:
			for var_name, feat_val in zip(self.var_names, instance):
				rule_outputs = np.array(self.get_firing_strengths())
				normalized_activation_rule = np.divide(rule_outputs, np.sum(rule_outputs))
				# save rule outputs for estimating probas later
				self.A.append(normalized_activation_rule)

	def preprocess_a(self):

		n_rules = len(self._rules)
		longform_betas = np.asarray(self.A)
		longform_betas = longform_betas.T.ravel()
		just_betas = np.split(longform_betas, n_rules)
		self.just_beta = np.asarray(just_betas)
		return just_betas

	def estimate_probas(self):
		self.prepare_a()
		A = self.preprocess_a()
		A = np.transpose(A)
		probas = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, self.y))
		if len(np.unique(self.y)) == 2:
			probas = np.hstack((probas, 1-probas))
		return probas

	def get_probas(self):
		""" Will get the probabilities from a probabilistic rule base.

		Returns:
			<class 'numpy.ndarray'>: The probabilities of a probabilistic fuzzy rulebase.
		"""
		probas = []
		for proba in self._rules:
			probas.append(proba[1])
		return np.vstack(probas)

	def set_proba_to_none(self):
		self._probas = None

	def probabilistic_inference(self, ignore_errors=False, verbose=False, return_class=False):
		""" A zero-order TS fuzzy system can produce the same output as the expected output of 
		a probabilistic fuzzy system provided that its consequent parameters are selected as the 
		conditional expectation of the defuzzified output membership functions. This approach
		gets the activations of rules given a instance (a sample of data), their corresponding 
		probability and will return either the corresponding probabilities for every class or 
		the class corresponding to the highest probability when return_class is set to True. See 
		the readme file for an example. Exact details are described in the paper by Fialho 
		et al. (2016) in the Applied Soft Computing journal.


		Args:
			ignore_errors (bool, optional): Not implemented. Defaults to False.
			verbose (bool, optional): Not implemented. Defaults to False.
			return_class (bool, optional): Choose depending on needs. Defaults to False. 
			When return_class is set to False a list of probabilities for each class is 
			returned, otherwise (True) the class itself is returned.
			

		Returns:
			<class 'numpy.int64'>: The class as a numpy integer
			<class 'numpy.ndarray'>: The probabilities for a given system. Shape: (n_samples, n_classes)

		"""
		if self.__estimate == False:
			probs = self.get_probas()
			result = self.mediate_probabilistic(probs)
			if return_class == True:
				return np.argmax(result)
			return result
		else:
			probs = self.estimate_probas()
			self.probabilistic_inference()


	def predict_pfs(self):
		"""Given a list of variables and a numpy matrix with (n_samples, n_variables) return predictions.

		Returns:
			[ndarray]: a list of predictions.
		"""
		preds_ = []
		for instance in self._X:
			for var_name, feat_val in zip(self.var_names, instance):
				self.set_variable(var_name, feat_val)
			preds_.append(self.probabilistic_inference())
		return preds_

if __name__ == '__main__':
	pass
