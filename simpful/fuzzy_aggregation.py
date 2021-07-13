from simpful import FuzzySet
from numpy import prod

class FuzzyAggregator(object):
	"""
		Creates a new fuzzy aggregation object.

		Args:
			verbose: True/False, toggles verbose mode.
	"""

	def __init__(self, verbose=False):
		self._variables = {}
		self._values = {}
		self.verbose = verbose

	def add_variables(self, *args):
		"""
		Adds variables and their fuzzy sets to perform fuzzy aggregation.

		Args:
			*args: 'FuzzySet' objects, whose 'term' argument is the name of the variable.
		"""
		for v in args:
			if isinstance(v, FuzzySet):
				self._variables[v._term] = v
			else:
				raise Exception("ERROR: please provide only fuzzy set objects as arguments")

	def set_variable(self, name, value):
		"""
		Sets the numerical value of a variable to be aggregated.

		Args:
			name: name of the variables to be set.
			value: numerical value to be set.
		"""
		try: 
			value = float(value)
			self._values[name] = value
			if self.verbose: print(" * Variable %s set to %f" % (name, value))
		except ValueError:
			raise Exception("ERROR: specified value for "+name+" is not an integer or float: "+value)

	def aggregate(self, variables=None, aggregation_fun="product"):
		"""
		Performs fuzzy aggregation.

		Args:
			variables: list of variables names to be aggregated.  If empty, all added variables are aggregated.
			aggregation_fun: pointer to a fuzzy aggregation function or string name of an implemented aggregation method. Default method is "product".
				Currently implemented methods: product, min, max, arit_mean
		Returns:
			Numerical result of the aggregation, as provided by the aggregation function.
		"""
		# In development
		
		# default: aggregate ALL variables
		if variables == None:
			variables = list(set(self._variables.keys()))

		if len(variables) > len(set(variables)):
			raise Exception("ERROR: provided list of variables to aggregate contains two or more repetitions, terminating.")

		memberships = []
		for v in variables:
			try:
				value = self._values[v]
				result = self._variables[v].get_value(value)
				memberships.append(result)
			except KeyError:
				raise Exception("ERROR: term "+v+" not defined.")
		
		if self.verbose:
			print(" * Aggregating the following values:", memberships)
			print(" * Using aggregation function:", aggregation_fun)
		
		if callable(aggregation_fun):
			return aggregation_fun(memberships)
		elif aggregation_fun == "product":
			return prod(memberships)
		elif aggregation_fun == "min":
			return min(memberships)
		elif aggregation_fun == "max":
			return max(memberships)
		elif aggregation_fun == "arit_mean":
			return sum(memberships)/len(memberships)
		else:
			raise Exception("ERROR: Please provide pointer to callable function or the name of an implemented aggregation function")