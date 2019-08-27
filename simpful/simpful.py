from __future__ import print_function
from numpy import array, argmin, argmax
from scipy.interpolate import interp1d
from collections import defaultdict
from math import *
import re

class MembershipFunction(object):

	def __init__(self, FS_list=[], concept=""):
		if FS_list==[]:
			print ("ERROR: please specify at least one fuzzy set")
			exit(-2)
		if concept=="":
			print ("ERROR: please specify a concept connected to the MF")
			exit(-3)

		self._FSlist = FS_list
		self._concept = concept


	def get_values(self, v):
		result = {}
		for fs in self._FSlist:
			result[fs._term] = fs.get_value(v)
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
			print ("ERROR: more than one point required")
			exit(-1)
		if term=="":
			print ("ERROR: please specify a linguistic term")
			exit(-3)

		if verbose:
			if len(points)==2: # singleton
				pass
			elif len(points)==3: # triangle
				print ("Triangle fuzzy set required:", points)
				self._type = "TRIANGLE"
			elif len(points)==4: # trapezoid
				print ("Trapezoid fuzzy set required:", points)
				self._type = "TRAPEZOID"
			else:
				print ("Polygon set required:", points)
				self._type = "POLYGON"

		self._points = array(points)
		self._term = term

	def get_value(self, v):		
		f = interp1d(self._points.T[0], self._points.T[1], 
			bounds_error=False, fill_value=(self._points.T[1][0], self._points.T[1][-1]))
		result = f(v)
		return(result)


class FuzzyReasoner(object):

	def __init__(self, show_banner=True):
		self._rules = []
		self._mfs = {}
		self._variables = {}
		self._crispvalues = {}
		self._outputfunctions = {}
		if show_banner: self._banner()

	def _banner(self):
		print ("  ____  __  _  _  ____  ____  _  _  __   ")
		print (" / ___)(  )( \\/ )(  _ \\(  __)/ )( \\(  ) v1.2.0 ")
		print (" \\___ \\ )( / \\/ \\ ) __/ ) _) ) \\/ (/ (_/\\ ")
		print (" (____/(__)\\_)(_/(__)  (__)  \\____/\\____/")
		print ()
		print (" Created by Marco S. Nobile (m.s.nobile@tue.nl)")
		print (" and Simone Spolaor (simone.spolaor@disco.unimib.it)")
		print ()

	def set_variable(self, name, value, verbose=False):
		try: 
			value = float(value)
			self._variables[name] = value
			if verbose: print  (" * Variable %s set to %f" % (name, value))
		except:
			print ("ERROR: specified value for", name, "is not an integer or float:", value)
			exit()

	def add_rules(self, rules, verbose=False):
		for rule in rules:
			parsed_antecedent = curparse(preparse(rule), verbose=verbose)
			parsed_consequent = postparse(rule, verbose=verbose)
			self._rules.append( [parsed_antecedent, parsed_consequent] )
			print (" * Added rule IF", parsed_antecedent, "THEN", parsed_consequent)
		print (" * %d rules successfully added" % len(rules))


	def add_membership_function(self, name, MF):
		self._mfs[name]=MF
		print (" * Membership function for '%s' successfully added" % name)

	def set_crisp_output_value(self, name, value):
		self._crispvalues[name]=value
		print (" * Crisp output value for '%s' set to %f" % (name, value))


	def set_output_function(self, name, function):
		self._outputfunctions[name]=function
		print (" * Output function for '%s' set to '%s'" % (name, function))


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
							print ("ERROR: one rule calculates an output named '%s', but I cannot find it among the output crisp terms or funtions. Aborting." % outterm)
							print (" --- PROBLEMATIC RULE:")
							print ("IF", ant, "THEN", res)
							print (" --- CRISP OUTPUTS:")
							for k,v in self._crispvalues.items():
								print (k, v)
							print
							for k,v in self._outputfunctions.items():
								print (k,v)
							raise Exception("Mistake in output names")
							exit()
					if crisp:
						crispvalue = self._crispvalues[outterm]
					else:
						string_to_evaluate = self._outputfunctions[outterm]
						for k,v in self._variables.items():
							string_to_evaluate = string_to_evaluate.replace(k,str(v))
						#print (" * About to evaluate: %s" % string_to_evaluate)
						crispvalue = eval(string_to_evaluate)						

					try:
						value = ant.evaluate(self) 
					except: 
						print ("ERROR: one rule cannot be evaluated properly because of a problematic clause")
						print (" --- PROBLEMATIC RULE:")
						print ("IF", ant, "THEN", res, "\n")
						exit()
						#raise Exception("Mistake in fuzzy rule")

					temp = value*crispvalue
					num += temp
					den += value

			try:
				final_result[output] = num / den
			except:
				if ignore_errors==True:
					print ("WARNING: cannot perform Sugeno inference for variable '%s', it does only appear as antecedent in the fuzzy rules" % output)
				else:
					print ("ERROR: cannot perform Sugeno inference for variable '%s', it does only appear as antecedent in the fuzzy rules" % output)
					exit()
		return final_result


	def Sugeno_inference(self, terms=None, ignore_errors=False):

		# default: inference on ALL rules/terms
		if terms == None:
			temp = [rule[1][0] for rule in self._rules] 
			terms= list(set(temp))

		array_rules = array(self._rules)
		result = self.mediate( terms, array_rules.T[0], array_rules.T[1], ignore_errors=ignore_errors )
		return result


class Clause(object):

	def __init__(self, variable, term, verbose=False):
		self._variable = variable
		self._term = term

	def evaluate(self, FuzzySystem, verbose=False):
		ans = FuzzySystem._mfs[self._variable].get_values(FuzzySystem._variables[self._variable])
		if verbose: 
			print ("Checking if", self._variable,)
			print ("whose value is", FuzzySystem._variables[self._variable],)
			print ("is actually", self._term)
			print ("answer:", ans[self._term])
		try:
			return ans[self._term]
		except KeyError:
			print ("ERROR: cannot find term '%s' in fuzzy rules, aborting." % self._term)
			print (" ---- PROBLEMATIC CLAUSE:")
			print (self)
			raise Exception("Name error in some clause of some rule")

	def __repr__(self):
		return "c.(%s IS %s)" % (self._variable, self._term)


class Functional(object):

	def __init__(self, fun, A, B):
		self._A = A
		self._B = B
		self._fun = fun

	def evaluate(self, FuzzySystem):
		if self._A=="":
			# support for unary operators
			# print ("Unary detected")
			B = self._B.evaluate(FuzzySystem)
			return array(eval(self._fun+"(%s)" % B))
		else:
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

def postparse(STRINGA, verbose=False):
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


def curparse(STRINGA, verbose=False):

	# base case
	if STRINGA=="": return "" 
	
	#regex = re.compile("^\([a-z,_,A-Z]* IS [a-z,_,A-Z]*\)$")
	regex = re.compile("^\([a-z,_,A-Z,0-9]*\s*IS\s*[a-z,_,A-Z,0-9]*\)$")
	if regex.match(STRINGA):
		
		# base case
		variable = STRINGA[1:STRINGA.find("IS")].strip()
		term     = STRINGA[STRINGA.find("IS")+3:-1].strip()
		ret_clause = Clause(variable, term, verbose=verbose)
		if verbose: 
			print (ret_clause)
		return ret_clause

	else:

		# recursion
		removed_parentheses = STRINGA[STRINGA.find("(")+1:STRINGA.rfind(")")].strip()
		if removed_parentheses[:3]=="NOT":
			beginindop = 0
			endindop = 3
		else:
			try:
				beginindop, endindop = find_index_operator(removed_parentheses)
			except:
				print ("ERROR: badly formatted rule (wrong capitalization perhaps?). Aborting.")
				print (" ---- PROBLEMATIC RULE:")
				print (STRINGA)
				exit()

		firsthalf = removed_parentheses[:beginindop].strip()
		secondhalf = removed_parentheses[endindop:].strip()
		operator = removed_parentheses[beginindop:endindop].strip()

		
		return Functional(operator, curparse(firsthalf), curparse(secondhalf))


if __name__ == '__main__':
	
	pass