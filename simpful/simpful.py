from pylab import *
from .fuzzy_sets import FuzzySet, MF_object, Sigmoid_MF, InvSigmoid_MF, Gaussian_MF, InvGaussian_MF, DoubleGaussian_MF, Triangular_MF, Trapezoidal_MF
from .rule_parsing import curparse, preparse, postparse
from numpy import array, linspace
from scipy.interpolate import interp1d
from collections import defaultdict
try:
	import seaborn as sns
except ImportError:
	pass


linestyles= ["-", "--", ":", "-."]


class UndefinedUniverseOfDiscourseError(Exception):

	def __init__(self, message):
		self.message = message


class LinguisticVariable(object):

	def __init__(self, FS_list=[], concept=None, universe_of_discourse=None):
		"""
		Creates a new linguistic variable.
		Args:
			FS_list: a list of FuzzySet instances.
			concept: a brief description of the concept represented by the linguistic variable.
			universe_of_discourse: a list of two elements, specifying min and max of universe of discourse.
			It must be specified to exploit plotting facilities.
		"""

		if FS_list==[]:
			print("ERROR: please specify at least one fuzzy set")
			exit(-2)
		#if concept is None:
		#	print("ERROR: please specify a concept connected to the linguistic variable")
		#	exit(-3)
		self._universe_of_discourse = universe_of_discourse

		self._FSlist = FS_list
		self._concept = concept


	def get_values(self, v):
		result = {}
		for fs in self._FSlist:
			result[fs._term] = fs.get_value(v)
		return result


	def get_universe_of_discourse(self):
		"""
		Returns the universe of discourse of the linguistic variable.
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
		mi, ma = self.get_universe_of_discourse()
		x = linspace(mi, ma, 10000)

		#pal = sns.color_palette("husl", len(self._FSlist))
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
		return "L.V.: "+self._concept


class FuzzySystem(object):

	def __init__(self, operators=None, show_banner=True, verbose=True):
		"""
		Creates a new fuzzy system.
		Args:
			operators: a list of strings, specifying fuzzy operators to be used instead of defaults.
			Currently supported operators: 'AND_PRODUCT'.
			show_banner: True/False, toggles display of banner.
			verbose: True/False, toggles verbose mode.
		"""
		self._rules = []
		self._lvs = {}
		self._variables = {}
		self._crispvalues = {}
		self._outputfunctions = {}
		self._outputfuzzysets = {}
		if show_banner: self._banner()
		self._operators = operators

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
		try: 
			value = float(value)
			self._variables[name] = value
			if verbose: print(" * Variable %s set to %f" % (name, value))
		except ValueError:
			raise Exception("ERROR: specified value for "+name+" is not an integer or float: "+value)

	def add_rules(self, rules, verbose=False):
		"""
		Adds new fuzzy rules to the fuzzy system.
		Args:
			rules: list of fuzzy rules to be added. Rules must be specified as strings, respecting Simpful's syntax.
			verbose: True/False, toggles verbose mode.
		"""
		for rule in rules:
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
		if LV._concept is None: 
			LV._concept = name
		self._lvs[name]=LV
		if verbose: print(" * Linguistic variable '%s' successfully added" % name)

	def set_crisp_output_value(self, name, value, verbose=False):
		"""
		Adds a new crisp output value to the fuzzy system.
		Args:
			name: string containing the identifying name of the crisp output value.
			value: numerical value of the crisp output value to be added to the fuzzy system.
			verbose: True/False, toggles verbose mode.
		"""
		self._crispvalues[name]=value
		if verbose: print(" * Crisp output value for '%s' set to %f" % (name, value))


	def set_output_function(self, name, function, verbose=False):
		"""
		Adds a new output function to the fuzzy system.
		Args:
			name: string containing the identifying name of the output function.
			function: string containing the output function to be added to the fuzzy system.
			The function specified in the string must use the names of linguistic variables contained in the fuzzy system object.
			verbose: True/False, toggles verbose mode.
		"""
		self._outputfunctions[name]=function
		if verbose: print(" * Output function for '%s' set to '%s'" % (name, function))

	def set_output_FS(self, fuzzyset, verbose=False):
		"""
		Adds a new output function as a Fuzzy Set object.
		Args: 
			name: string containing the identifying name of the output function.
			fuzzyset: FuzzySet object used as output (in a Mamdani FIS).
			verbose: True/False, toggles verbose mode.
		"""
		self._outputfuzzysets[fuzzyset.get_term()]=fuzzyset
		if verbose: print(" * Output fuzzy set for '%s' set" % (name))


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

		list_crisp_values = [x[0] for x in self._crispvalues.items()]
		list_output_funs  = [x[0] for x in self._outputfunctions.items()]

		#print (self._variables)

		for output in outputs:

			if verbose:
				print (" * Processing output for variable %s" %  output)
				print ("   whose universe of discourse is:", self._lvs[output].get_universe_of_discourse())
			cuts_list = defaultdict()

			x0, x1 = self._lvs[output].get_universe_of_discourse()

			for (ant, res) in zip(antecedent, results):

				outname = res[0]
				outterm = res[1]

				if verbose:	print(ant,res,outname,outterm)			

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

			for u in integration_points:
				#print ("x=%.1f" % u)
				comp_values = []
				for k,v in cuts_list.items():
					result = float(self._outputfuzzysets[k].get_value_cut(u, cut=v))
					comp_values.append(result)
				keep = max(comp_values)
				values.append(keep)
				weightedvalues.append(keep*u)

			CoG = sum(weightedvalues)/sum(values)
			if verbose: print (" * CoG:", CoG)
			
			final_result[output] = CoG 

		return final_result


	def Sugeno_inference(self, terms=None, ignore_errors=False, verbose=False):
		"""
		Performs Sugeno fuzzy inference.
		Args:
			terms: list of the names of the variables on which inference must be performed.
			If empty, all variables appearing in the consequent of a fuzzy rule are inferred.
			ignore_errors: True/False, toggles the raising of errors during the inference.
			verbose: True/False, toggles verbose mode.

		Returns:
			a dictionary, containing as keys the variables' names and as values their numerical inferred values.
		"""
		# default: inference on ALL rules/terms
		if terms == None:
			temp = [rule[1][0] for rule in self._rules] 
			terms= list(set(temp))

		array_rules = array(self._rules)
		result = self.mediate( terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors )
		return result


	def Mamdani_inference(self, terms=None, ignore_errors=False, verbose=False, subdivisions=1000):
		# raise Exception("Mamdani inference is under development")
		"""
		Performs Mamdani fuzzy inference.
		Args:
			terms: list of the names of the variables on which inference must be performed.
			If empty, all variables appearing in the consequent of a fuzzy rule are inferred.
			subdivisions: the number of integration steps to be performed (default: 1000).
			ignore_errors: True/False, toggles the raising of errors during the inference.
			verbose: True/False, toggles verbose mode.

		Returns:
			a dictionary, containing as keys the variables' names and as values their numerical inferred values.
		"""
		# default: inference on ALL rules/terms
		if terms == None:
			temp = [rule[1][0] for rule in self._rules] 
			terms= list(set(temp))

		array_rules = array(self._rules)
		result = self.mediate_Mamdani( terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors, verbose=verbose , subdivisions=subdivisions)
		return result


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

		#print(" * Plotting figure %dx%d" % (columns, rows))
		#print(self._lvs)
		fig, ax = subplots(rows, columns, figsize=(columns*5, rows*5))

		if rows==1: ax = [ax]
		if columns==1: ax= [ax]

		n = 0
		for k, v in self._lvs.items():
			#print(k, v)
			r = n%4
			c = n//4
			#print(r,c)
			v.draw(ax[c][r])
			ax[c][r].set_ylim(0,1)
			n+=1

		for m in range(n, columns*rows):
			r = m%4
			c = m//4
			ax[c][r].axis('off')

		fig.tight_layout()
		fig.savefig(outputfile)

if __name__ == '__main__':
	
	pass