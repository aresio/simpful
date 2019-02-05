from numpy import array, argmin, argmax
from scipy.interpolate import interp1d
from collections import defaultdict




class MembershipFunction(object):

	def __init__(self, FS_list=[], concept=""):
		if FS_list==[]:
			print "ERROR: please specify at least one fuzzy set"
			exit(-2)
		if concept=="":
			print "ERROR: please specify a concept connected to the MF"
			exit(-3)

		self._FSlist = FS_list
		self._concept = concept


	def get_values(self, v):
		#print "Getting MF(%f).." % (v)
		result = {}
		for fs in self._FSlist:
			result[fs._term] = fs.get_value(v)
		#print "Results in get_values:", result
		return result

	def get_universe_of_discourse(self):
		mins = []
		maxs = []
		for fs in self._FSlist:
			mins.append(min(fs._points.T[0]))
			maxs.append(max(fs._points.T[0]))
		return min(mins), max(maxs)

	def draw(self, TGT):	
		import seaborn as sns
		mi, ma = self.get_universe_of_discourse()
		x = linspace(mi, ma, 1e4)
		for fs in self._FSlist:
			sns.regplot(fs._points.T[0], fs._points.T[1], marker="d", fit_reg=False)
			f = interp1d(fs._points.T[0], fs._points.T[1], bounds_error=False, fill_value=(0,0))
			plot(x, f(x), "--", label=fs._term)
			plot(TGT, f(TGT), "*", ms=10)
		title(self._concept)
		legend(loc="best")
		show()

	def __repr__(self):
		return self._concept


class FuzzySet(object):

	def __init__(self, points=None, term="", verbose=False):
		if len(points)<2: 
			print "ERROR: more than one point required"
			exit(-1)
		if term=="":
			print "ERROR: please specify a linguistic term"
			exit(-3)

		if verbose:
			if len(points)==2: # singleton
				pass
			elif len(points)==3: # triangle
				print "Triangle fuzzy set required:", points
				self._type = "TRIANGLE"
			elif len(points)==4: # trapezoid
				print "Trapezoid fuzzy set required:", points
				self._type = "TRAPEZOID"
			else:
				print "Polygon set required:", points
				self._type = "POLYGON"

		self._points = array(points)
		self._term = term

	def get_value(self, v):		
		f = interp1d(self._points.T[0], self._points.T[1], 
			bounds_error=False, fill_value=(self._points.T[1][0], self._points.T[1][1]))
		result = f(v)
		return(result)


class FuzzyReasoner(object):

	def __init__(self, show_banner=True):
		self._rules = []
		self._mfs = {}
		self._variables = {}
		self._crispvalues = {}
		if show_banner: self._banner()

	def _banner(self):
		print "  ____  __  _  _  ____  ____  _  _  __   "
		print " / ___)(  )( \\/ )(  _ \\(  __)/ )( \\(  ) v1.0.8 "
		print " \\___ \\ )( / \\/ \\ ) __/ ) _) ) \\/ (/ (_/\\ "
		print " (____/(__)\\_)(_/(__)  (__)  \\____/\\____/"
		print 
		print " Created by Marco S. Nobile (nobile@disco.unimib.it)"
		print " and Simone Spolaor (simone.spolaor@disco.unimib.it)"
		print 

	def set_variable(self, name, value, verbose=False):
		self._variables[name] = value
		if verbose: print  " * Variable %s set to %f" % (name, value)

	def add_rules(self, rules):
		for rule in rules:
			parsed_antecedent = curparse(preparse(rule))
			parsed_consequent = postparse(rule)
			self._rules.append( [parsed_antecedent, parsed_consequent] )
			print " * Added rule IF", parsed_antecedent, "THEN", parsed_consequent
		print " * %d rules successfully added" % len(rules)


	def add_membership_function(self, name, MF):
		self._mfs[name]=MF
		print " * Membership function for '%s' successfully added" % name

	def set_crisp_output_value(self, name, value):
		self._crispvalues[name]=value
		print " * Crisp output value for '%s' set to %f" % (name, value)

	def mediate(self, outputs, antecedent, results, ignore_errors=False):

		final_result = {}

		for output in outputs:
			num = 0
			den = 0
			
			for (ant, res) in zip(antecedent, results):
				outname = res[0]
				outterm = res[1]
				if outname==output:
					try:
						crispvalue = self._crispvalues[outterm]
					except KeyError:
						print "ERROR: one rule calculates an output named '%s', but I cannot find it among the output crisp terms. Aborting." % outterm
						print " --- PROBLEMATIC RULE:"
						print "IF", ant, "THEN", res
						print " --- CRISP OUTPUTS:"
						for k,v in self._crispvalues.items():
							print k, v
						print
						raise Exception("Mistake in crisp output names")

					try:
						value = ant.evaluate(self) 
					except: 
						print "ERROR: one rule cannot be evaluated properly because of a problematic clause"
						print " --- PROBLEMATIC RULE:"
						print "IF", ant, "THEN", res, "\n"
						exit()
						#raise Exception("Mistake in fuzzy rule")

					temp = value*crispvalue
					num += temp
					den += value

			try:
				final_result[output] = num / den
			except:
				if ignore_errors==True:
					print "WARNING: cannot perform Sugeno inference for variable '%s', it does only appear as antecedent in the fuzzy rules" % output
				else:
					print "ERROR: cannot perform Sugeno inference for variable '%s', it does only appear as antecedent in the fuzzy rules" % output
					exit()
		return final_result


	def Sugeno_inference(self, terms, ignore_errors=False):
		array_rules = array(self._rules)
		result = self.mediate( terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors )
		return result

	"""
	def plot_surface(self, variables, output, ax, steps=100):

		from mpl_toolkits.mplot3d import Axes3D
		import matplotlib.pyplot as plt
		from matplotlib import cm
		from matplotlib.ticker import LinearLocator, FormatStrFormatter
	
		if len(variables)>2: 
			print "Unable to plot more than 3 dimensions, aborting."
			exit(-10)
		
		if len(variables)==2:

			ud1 = variables[0].get_universe_of_discourse()
			ud2 = variables[1].get_universe_of_discourse()
			inter1 = linspace(ud1[0], ud1[1], steps) 
			inter2 = linspace(ud2[0], ud2[1], steps) 

			X, Y = meshgrid(inter1, inter2)

			def wrapper(x,y):
				print x,y
				self.set_variable(variables[0]._concept, x)
				self.set_variable(variables[1]._concept, y)
				res = self.evaluate_rules()
				print res
				return res[output]
			
			zs = array([wrapper(x,y) for x,y in zip(ravel(X), ravel(Y))])
			Z = zs.reshape(X.shape)

			ax.plot_surface(array(X),array(Y),array(Z), cmap= "CMRmap")
			ax.set_xlabel(variables[0]._concept)
			ax.set_ylabel(variables[1]._concept)
			ax.set_zlabel(output)
			ax.view_init(-90, 0)  # vertical, horizontal
	"""
		

class Clause(object):

	def __init__(self, variable, term):
		self._variable = variable
		self._term = term

	def evaluate(self, FuzzySystem, verbose=False):
		ans = FuzzySystem._mfs[self._variable].get_values(FuzzySystem._variables[self._variable])
		if verbose: 
			print "Checking if", self._variable, 
			print "whose value is", FuzzySystem._variables[self._variable],
			print "is actually", self._term
			print "answer:", ans[self._term]
		try:
			return ans[self._term]
		except KeyError:
			print "ERROR: cannot find term '%s' in fuzzy rules, aborting." % self._term
			print " ---- PROBLEMATIC CLAUSE:"
			print self
			raise Exception("Name error in some clause of some rule")

	def __repr__(self):
		return "c.(%s IS %s)" % (self._variable, self._term)


class Functional(object):

	def __init__(self, fun, A, B):
		self._A = A
		self._B = B
		self._fun = fun

	def evaluate(self, FuzzySystem):
		A = self._A.evaluate(FuzzySystem)
		B = self._B.evaluate(FuzzySystem)
		return array(eval(self._fun+"(%s, %s)" % (A,B)))
		
	def __repr__(self):
		return "f.(" + str(self._A) + " " + self._fun + " " + str(self._B) + ")"


def OR(x,y): return max(x, y)
def AND(x,y): return min(x, y)
def NOT(x): return 1.-x


def preparse(STRINGA):
	# extract the antecedent
	return STRINGA[STRINGA.find("IF")+2:STRINGA.find("THEN")].strip()

def postparse(STRINGA):
	# extract the consequent
	stripped = STRINGA[STRINGA.find("THEN")+4:].strip("() ")
	return stripped[:stripped.find("IS")].strip(), stripped[stripped.find("IS")+2:].strip()

def find_index_operator(string):
	#print string
	pos = 0
	par = 1
	while(par>0):
		pos+=1
		if pos>=len(string):
			raise Exception("badly formatted rule, terminating")
		if string[pos]==")": par-=1
		if string[pos]=="(": par+=1
	pos2 = pos
	while(string[pos2]!="("):
		pos2+=1
	return pos+1, pos2


def curparse(STRINGA):
	import re
	#regex = re.compile("^\([a-z,_,A-Z]* IS [a-z,_,A-Z]*\)$")
	regex = re.compile("^\([a-z,_,A-Z,0-9]*\s*IS\s*[a-z,_,A-Z,0-9]*\)$")
	if regex.match(STRINGA):
		
		# base case
		variable = STRINGA[1:STRINGA.find("IS")].strip()
		term     = STRINGA[STRINGA.find("IS")+3:-1].strip()
		return Clause(variable, term)

	else:

		# recursion
		removed_parentheses = STRINGA[STRINGA.find("(")+1:STRINGA.rfind(")")].strip()
		#beginindop, endindop = find_index_operator(removed_parentheses)
		try:
			beginindop, endindop = find_index_operator(removed_parentheses)
		except:
			print "ERROR: badly formatted rule (wrong capitalization?). Aborting."
			print " ---- PROBLEMATIC RULE:"
			print STRINGA
			exit()

		firsthalf = removed_parentheses[:beginindop].strip()
		secondhalf = removed_parentheses[endindop:].strip()
		operator = removed_parentheses[beginindop:endindop].strip()
		return Functional(operator, curparse(firsthalf), curparse(secondhalf))


if __name__ == '__main__':

	pass