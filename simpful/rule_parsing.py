import re
from numpy import array
import numpy as np

class Clause(object):
	"""  A clause is a part of the sentence that contains a verb ('IS', in this context). E.g. OXI IS low_flow.

	Args:
		object (<class 'simpful.rule_parsing.Clause'>): A variable (e.g. OXI) and a term (e.g. low_flow) is needed to form a Clause.
	"""
	def __init__(self, variable, term, verbose=False):
		self._variable = variable
		self._term = term

	def evaluate(self, FuzzySystem, verbose=False, operators=None):
		try:
			ans = FuzzySystem._lvs[self._variable].get_values(FuzzySystem._variables[self._variable])
		except KeyError:
			raise Exception("ERROR: variable '" + self._variable + "' not defined.\n"
				+ " ---- PROBLEMATIC CLAUSE:\n"
				+ str(self))
		if verbose: 
			print("Checking if", self._variable,)
			print("whose value is", FuzzySystem._variables[self._variable],)
			print("is actually", self._term)
			print("answer:", ans[self._term])
		try:
#			FuzzySystem.A.append(ans[self._term])	# i need to remove this (@nikhil)
			return ans[self._term]
		except KeyError:
			raise Exception("ERROR: term '" + self._term + "'' not defined.\n"
				+ " ---- PROBLEMATIC CLAUSE:\n"
				+ str(self))

	def __repr__(self):
		return "c.(%s IS %s)" % (self._variable, self._term)


class Functional(object):
	""" Represents a set of Clauses; see Clause description for more information.

	Args:
		object (<class 'simpful.rule_parsing.Functional'>): Contains a minimum of 2 Clauses
		aliased as A and B. These can be joined by a simpful supported operator.
	"""    
	def __init__(self, fun, A, B, operators=None):
		self._A = A
		self._B = B

		if operators is None:
			self._fun = fun
		else:
			if "AND_PRODUCT" in operators: 
				if fun=="AND":
					self._fun = "AND_p"
				else:
					self._fun = fun
			else:
				self._fun = fun

	def evaluate(self, FuzzySystem):
		if self._A=="":
			# support for unary operators
			# print("Unary detected")
			B = self._B.evaluate(FuzzySystem)
			return array(eval(self._fun+"(%s)" % B))
		else:
			A = self._A.evaluate(FuzzySystem)
			B = self._B.evaluate(FuzzySystem)
			# bugfix for not @ nikhil
			if re.match(r'[)]\s', self._fun) is not None:
				self._fun = re.sub('[)]\s', '', self._fun)
			return array(eval(self._fun+"(%s, %s)" % (A,B)))
		
	def __repr__(self):
		return "f.(" + str(self._A) + " " + self._fun + " " + str(self._B) + ")"


# basic definitions of operators
def OR(x,y): return max(x, y)
def AND(x,y): return min(x, y)
def AND_p(x,y): return x*y
def NOT(x): return 1.-x


def preparse(STRINGA):
	"""Extracts the antecedent of a defined rule.

	Args:
		STRINGA (<class 'str'>): Rule to be parsed.

	Returns:
		<class 'str'>: Antecendent of parsed rule.

	Examples:
		>>> normal_rule = "IF (OXI IS low_flow) THEN (POWER IS LOW_POWER)"
		'(OXI IS low_flow)'
	"""
	return STRINGA[STRINGA.find("IF")+2:STRINGA.find(" THEN")].strip()

def postparse(STRINGA, verbose=False):
	"""Handles the consequent of a rule. Extracts the output and its corresponding term.

	Args:
		STRINGA (<class 'str'>): Rule to be parsed.
		verbose (bool, optional): [Not implemented]. Defaults to False.

	Raises:
		Exception: Raises exceptions regarding capitalizaion and syntax.
		Exception: Raises exceptions regarding the sum of probabilities. The probabilities must sum upto 1.

	Returns:
		[<class 'tuple'>]: Will return a tuple containing the consequent of a  normal rule. 
		In the case of probabilistic rules a tuple of size 2 (of type list) is returned 
		with element 1 containing the probabilities and element 2 the output and probabilities.

	Examples:
		>>> normal_rule = "IF (OXI IS low_flow) THEN (POWER IS LOW_POWER)"
		>>> print((postparse(rule2)))
		('POWER', 'LOW_POWER')

		>>> proba_rule = "IF (OXI IS low_flow) THEN P(POWER IS LOW_POWER)=0.33, P(POWER IS MEDIUM_POWER)=0.33, P(POWER IS HIGH_FUN)=0.34"
		>>> print(postparse(proba_rule))
		([0.33, 0.33, 0.34], ['POWER', 'LOW_POWER', '0.33', 'POWER', 'MEDIUM_POWER', '0.33', 'POWER', 'HIGH_FUN', '0.34'])

	"""
	stripped = STRINGA[STRINGA.find(" THEN")+5:].strip("() ")
	if STRINGA.find("THEN") == -1:
		raise Exception("ERROR: badly formatted rule, please check capitalization and syntax.\n"
						+ " ---- PROBLEMATIC RULE:\n"
						+ STRINGA)
	if re.match(r"P\(", stripped) is not None:
		probas = [float(i) for i in (re.findall(r"\d\.?\d*", stripped))]
		if not probas:
			class_info = re.findall(r"\w+(?=\sIS)|(?<=IS\s)\w+|\d\.\d\d|None", stripped)
			number_of_probas = class_info.count('None')
			if number_of_probas is False:
				raise Exception("For probability estimation probabilities must be intialized as None.")
			trigger = True
			return [number_of_probas, trigger]

		# if not np.isclose(sum(probas), 1):
		# 	raise Exception ("ERROR: badly formatted rule, sum of probabilities needs to be equal to 1.\n"
		# 					+ " ---- PROBLEMATIC RULE:\n"
		# 					+ STRINGA)
		
		return probas
	else:
		return tuple(re.findall(r"\w+(?=\sIS)|(?<=IS\s)\w+", stripped))

def find_index_operator(string, verbose=False):
	"""Will try to find an operator (e.g. AND, OR, NOT etc)

	Args:
		string (<class 'str'>): is usually passed from curparse without outer parenthesis.
		verbose (bool, optional): [description]. Defaults to False.

	Returns:
		tuple: indexes including whitespaces of operators (AND, AND_p, OR, NOT).
	
	Example:
		>>> removed_parentheses = 'OXI IS low_flow) AND (OXI IS medium_flow'
		>>> (16, 21)
	"""    
	if verbose: print(" * Looking for an operator in", string)
	pos = 0
	par = 1
	while(par>0):
		pos+=1
		# if pos>=len(string):
			#print(pos, pos2)
			# raise Exception("badly formatted rule, terminating")
		if string[pos]==")": par-=1
		if string[pos]=="(": par+=1
	pos2 = pos
	while(string[pos2]!="("):
		pos2+=1
	return pos+1, pos2

def curparse(STRINGA, verbose=False, operators=None):
	"""Given a rule the Clauses/Functional Objects are extracted.

	Args:
		STRINGA (<class 'str'>): A pre-parsed rule.
		verbose (bool, optional): Will tell user whether a single automic 
		clause is detected or not. Defaults to False.
		operators (None): Defaults to None. Is meant for initialization.

	Raises:
		Exception: An IndexError will be raised when find_index_operator(removed_parentheses, verbose=verbose) 
		is not possible.

	Returns:
		Clause OR Functional Object(possibly containing Clauses): After having detected 
		either a Clause or a Functional Object recursion is used to find new Clauses.
	
	Example:
		>>> STRINGA = '(OXI IS low_flow) AND (OXI IS medium_flow)'
		>>> curparse(STRINGA)
		f.(c.(OXI IS low_flow) AND c.(OXI IS medium_flow))
	"""
	# base case
	if STRINGA=="": return "" 

	STRINGA = STRINGA.strip()
	if STRINGA[0]!="(": STRINGA="("+STRINGA
	if STRINGA[-1]!=")": STRINGA=STRINGA+")"
	
	regex = re.compile(r"^\([a-z,_,A-Z,0-9]*\s*IS\s*[a-z,_,A-Z,0-9]*\)$")
	if regex.match(STRINGA):
		
		if verbose:	print(" * Regular expression is matching with single atomic clause:", STRINGA)

		# base case
		variable = STRINGA[1:STRINGA.find(" IS")].strip()
		term	 = STRINGA[STRINGA.find(" IS")+3:-1].strip()
		ret_clause = Clause(variable, term, verbose=verbose)
		if verbose:	print(" * Rule:", ret_clause)
		return ret_clause

	else:

		# there can be two explanations: missing parentheses, or sub-expression
		if verbose:	print(" * Regular expression is not matching with single atomic clause:", STRINGA)
		
		# try recursion
		removed_parentheses = STRINGA[STRINGA.find("(")+1:STRINGA.rfind(")")].strip()
		
		# if (removed_parentheses.find("(")==-1): return curparse("("+removed_parentheses+")")

		if verbose: print("  - After parentheses removal:", removed_parentheses)    #Get start and end index of operator

		if removed_parentheses[:3]=="NOT":
			beginindop = 0
			endindop = 3
			#if verbose: print( " * Detected unary operator NOT")
		else:
			try:
				beginindop, endindop = find_index_operator(removed_parentheses, verbose=verbose)
			except IndexError:
				raise Exception("ERROR: badly formatted rule, please check capitalization and syntax.\n"
					+ " ---- PROBLEMATIC RULE:\n"
					+ STRINGA)
		# remove firsthalf by selecting everything before operator, secondhalf by everything after
		firsthalf = removed_parentheses[:beginindop].strip()
		secondhalf = removed_parentheses[endindop:].strip()
		operator = removed_parentheses[beginindop:endindop].strip()
		if verbose:	print("  -- Found %s *%s* %s" % (firsthalf, operator, secondhalf))
		
		return Functional(operator, curparse(firsthalf, verbose=verbose, operators=operators), curparse(secondhalf, verbose=verbose, operators=operators), operators=operators)
