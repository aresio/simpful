import re
from numpy import array

class Clause(object):

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
            return ans[self._term]
        except KeyError:
            raise Exception("ERROR: term '" + self._term + "'' not defined.\n"
                + " ---- PROBLEMATIC CLAUSE:\n"
                + str(self))

    def __repr__(self):
        return "c.(%s IS %s)" % (self._variable, self._term)


class Functional(object):

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
            return array(eval(self._fun+"(%s, %s)" % (A,B)))
        
    def __repr__(self):
        return "f.(" + str(self._A) + " " + self._fun + " " + str(self._B) + ")"


# basic definitions of 
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

    Example:
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
        probas = [float(i) for i in (re.findall(r"\d\.\d\d", stripped))]
        if sum(probas)!=1:
            raise Exception ("ERROR: badly formatted rule, sum of probabilities needs to be equal to 1.\n"
                            + " ---- PROBLEMATIC RULE:\n"
                            + STRINGA)
        class_info = re.findall(r"\w+(?=\sIS)|(?<=IS\s)\w+|\d\.\d\d", stripped)
        out = (probas, class_info)
        return out
    else:
        return tuple(re.findall(r"\w+(?=\sIS)|(?<=IS\s)\w+", stripped))

def find_index_operator(string, verbose=False):
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

        if verbose: print("  - After parentheses removal:", removed_parentheses)

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

        firsthalf = removed_parentheses[:beginindop].strip()
        secondhalf = removed_parentheses[endindop:].strip()
        operator = removed_parentheses[beginindop:endindop].strip()
        if verbose:	print("  -- Found %s *%s* %s" % (firsthalf, operator, secondhalf))
        
        return Functional(operator, curparse(firsthalf, verbose=verbose, operators=operators), curparse(secondhalf, verbose=verbose, operators=operators), operators=operators)
