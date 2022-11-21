import re
from numpy import array

regex_clause_with_parentheses = re.compile(r"^\(\w*\s*IS\s*\w*\)$")
regex_clause = re.compile(r"^\w*\s*IS\s*\w*$")

class Clause(object):

    def __init__(self, variable, term, verbose=False):
        self._variable = variable
        self._term = term

    def evaluate(self, FuzzySystem, verbose=False, operators=None):
        try:
            ans = FuzzySystem._lvs[self._variable].get_values(FuzzySystem._variables[self._variable])
        except KeyError:
            raise Exception("ERROR: variable '" + self._variable + "' not defined, or input value not given.\n"
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

        if fun=="NOT":
            if B == "": raise Exception("Second operand missing")
        elif A == "": raise Exception("First operand missing")

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
def OR_p(x,y): return x+y-(x*y)
def AND(x,y): return min(x, y)
def AND_p(x,y): return x*y
def NOT(x): return 1.-x


def preparse(STRINGA):
    # extract the antecedent
    return STRINGA[STRINGA.find("IF")+2:STRINGA.find(" THEN")].strip()

def postparse(STRINGA, verbose=False):
    stripped = STRINGA[STRINGA.find(" THEN")+5:].strip(" ")
    if STRINGA.find("THEN") == -1:
        raise Exception("ERROR: badly formatted rule, please check capitalization and syntax.\n"
                        + " ---- PROBLEMATIC RULE:\n"
                        + STRINGA)
    # Probabilistic fuzzy rule
    if re.match(r"P\(", stripped) is not None:
        return tuple(re.findall(r"\w+(?=\sis)|(?<=is\s)\w+|\d\.\d\d", stripped))
    # Weighted fuzzy rule
    elif re.search(r"WEIGHT\s*(\d|\.)", stripped) is not None:
        return tuple(re.findall(r"\w+(?=\sIS\s)|(?<=\sIS\s)\w+|\.?\d\.?\d*", stripped))
    # Normal fuzzy rule
    else:
        return tuple(re.findall(r"\w+(?=\sIS\s)|(?<=\sIS\s)\w+", stripped))

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

def recursive_parse(text, verbose=False, operators=None, allow_empty=True): 
    # remove useless spaces around text
    text = text.strip()

    # case 0: empty string
    if text=="" or text=="()": 
        if verbose: print("WARNING: empty clause detected")
        if not allow_empty:
            raise Exception("ERROR: emtpy clauses not allowed") 
        else:
            return ""
    
    # case 1: simple clause ("this IS that")
    if regex_clause.match(text):
        if verbose: 
            print(" * Simple clause matched")

        variable = text[:text.find(" IS")].strip()
        term     = text[text.find(" IS")+3:].strip()
        ret_clause = Clause(variable, term, verbose=verbose)
        if verbose: 
            print(" * Rule:", ret_clause)
        return ret_clause
    
    elif regex_clause_with_parentheses.match(text):
        if verbose:
            print(" * Simple clause with parentheses matched")

        variable = text[1:text.find(" IS")].strip()
        term     = text[text.find(" IS")+3:-1].strip()
        ret_clause = Clause(variable, term, verbose=verbose)
        if verbose: 
            print(" * Rule:", ret_clause)
        return ret_clause

    else:
        if verbose:
            print(" * Regular expression is not matching with single atomic clause")

        # possible valid cases: 
        # 1) atomic clause
        # 2) atomic clause OPERATOR atomic clause
        # 3) NOT atomic clause
        # 4) (clause OPERATOR clause)
        # 5) ((...)) # experimental

        if text[:3]=="NOT":
            beginindop = 0
            endindop = 3
        elif text[:4]=="(NOT":
            text = text[1:-1]
            beginindop = 0
            endindop = 3
        else:
            try:
                beginindop, endindop = find_index_operator(text, verbose=verbose)

            except IndexError:
                # last attempt: remove parentheses (if any!)
                try: 
                    if text[0] == "(" and text[-1] == ")": 
                        text = text[1:-1]
                        return recursive_parse(text, operators=operators, verbose=verbose, allow_empty=allow_empty)
                    else: 
                        raise Exception("ERROR: badly formatted rule, please check capitalization and syntax.\n"
                        + " ---- PROBLEMATIC RULE:\n"
                        + text)

                except: 
                    raise Exception("ERROR: badly formatted rule, please check capitalization and syntax.\n"
                        + " ---- PROBLEMATIC RULE:\n"
                        + text)

        firsthalf = text[:beginindop].strip()
        secondhalf = text[endindop:].strip()
        operator = text[beginindop:endindop].strip()
        if operator.find(" ")>-1: 
            if verbose: 
                print("WARNING: space in operator '%s' detected" % operator)
                print(" "*(28+operator.find(" "))+"^")
            raise Exception("ERROR: operator %s invalid: cannot use spaces in operators" % operator)

        if verbose: print("  -- Found %s *%s* %s" % (firsthalf, operator, secondhalf))

        try:
            novel_fun = Functional(operator, 
            recursive_parse(firsthalf, verbose=verbose, operators=operators, allow_empty=allow_empty), 
            recursive_parse(secondhalf, verbose=verbose, operators=operators, allow_empty=allow_empty), 
        operators=operators)
        except:
            raise Exception("ERROR: badly formatted rule, please check capitalization and syntax.\n"
                    + " ---- PROBLEMATIC RULE:\n"
                    + text)
        return novel_fun