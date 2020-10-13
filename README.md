![Python package](https://github.com/aresio/simpful/workflows/Python%20package/badge.svg?branch=master)

# simpful
A Python library for fuzzy logic reasoning, designed to provide a simple and lightweight API, as close as possible to natural language. Simpful supports Mamdani and Sugeno reasoning of any order, parsing any complex fuzzy rules involving AND, OR, and NOT operators, using arbitrarily shaped fuzzy sets.

## Usage

This example shows how to specify the information about the linguistic variables, fuzzy sets, fuzzy rules, and input values to Simpful. The last line of code prints the result of the fuzzy reasoning.

### Example 1: Modelling the heating power of a gas burner using oxygen supply (Takagi Sugeno)

A simple fuzzy model (Takagi Sugeno) describing how the heating power of a gas burner depends on the oxygen supply. We use a point-based approach for defining the fuzzy sets. The consequents can either be crisp or functional. 

```
import simpful as sf

FS = sf.FuzzySystem()

# Define a linguistic variable.
S_1 = sf.FuzzySet( points=[[0, 1.],  [1., 1.],  [1.5, 0]],          term="low_flow" )
S_2 = sf.FuzzySet( points=[[0.5, 0], [1.5, 1.], [2.5, 1], [3., 0]], term="medium_flow" )
S_3 = sf.FuzzySet( points=[[2., 0],  [2.5, 1.], [3., 1.]],          term="high_flow" )
FS.add_linguistic_variable("OXI", sf.LinguisticVariable( [S_1, S_2, S_3] ))

# Define consequents.
FS.set_crisp_output_value("LOW_POWER", 0)
FS.set_crisp_output_value("MEDIUM_POWER", 25)
FS.set_output_function("HIGH_FUN", "OXI**2")

# Define fuzzy rules.
RULE1 = "IF (OXI IS low_flow) THEN (POWER IS LOW_POWER)"
RULE2 = "IF (OXI IS medium_flow) THEN (POWER IS MEDIUM_POWER)"
RULE3 = "IF (NOT (OXI IS low_flow)) THEN (POWER IS HIGH_FUN)"
FS.add_rules([RULE1, RULE2, RULE3])

# Set antecedents values, perform Sugeno inference and print output values.
FS.set_variable("OXI", .51)
print(FS.inference(['POWER']))
```


### Example 2: The Tipping Problem with Mamdani.

This second example shows how to model a FIS using Mamdani inference. It also shows some facilities 
that make modeling more concise and clear: automatic Triangles (i.e., pre-baked linguistic variables 
with equally spaced triangular fuzzy sets) and the automatic detection of the inference method.

```
from simpful import *

FS = FuzzySystem()

TLV = AutoTriangle(3, terms=['poor', 'average', 'good'], universe_of_discourse=[0,10])
FS.add_linguistic_variable("service", TLV)
FS.add_linguistic_variable("quality", TLV)

O1 = TriangleFuzzySet(0,0,13,   term="low")
O2 = TriangleFuzzySet(0,13,25,  term="medium")
O3 = TriangleFuzzySet(13,25,25, term="high")
FS.add_linguistic_variable("tip", LinguisticVariable([O1, O2, O3], universe_of_discourse=[0,25]))

FS.add_rules([
	"IF (quality IS poor) OR (service IS poor) THEN (tip IS low)",
	"IF (service IS average) THEN (tip IS medium)",
	"IF (quality IS good) OR (quality IS good) THEN (tip IS high)"
	])

FS.set_variable("quality", 6.5) 
FS.set_variable("service", 9.8) 

tip = FS.inference()
print(tip)
```

### Example 3: Probabilistic Classification based on a Takagi Sugeno approach.

Simpful now supports classification using conditional probabilities. Please keep in mind that these probabilities can not be estimated automatically yet and have to be fed to the system when defining the rules. The implementation is based on the following paper: [Function approximation using probabilistic fuzzy systems](https://research.tue.nl/en/publications/function-approximation-using-probabilistic-fuzzy-systems).

```

import simpful as sf

# A simple fuzzy model describing how the heating power of a gas burner depends on the oxygen supply.

# Initialize class
FS = sf.FuzzySystem()

# Define a linguistic variable.
S_1 = sf.FuzzySet( points=[[0, 1.],  [1., 1.],  [1.5, 0]],          term="low_flow" )
S_2 = sf.FuzzySet( points=[[0.5, 0], [1.5, 1.], [2.5, 1], [3., 0]], term="medium_flow" )
S_3 = sf.FuzzySet( points=[[2., 0],  [2.5, 1.], [3., 1.]],          term="high_flow" )
FS.add_linguistic_variable("OXI", sf.LinguisticVariable( [S_1, S_2, S_3] ))

# Define fuzzy rules.
RULE1 = "IF (OXI IS low_flow) THEN P(POWER IS LOW_POWER)=0.33, P(POWER IS MEDIUM_POWER)=0.33, P(POWER IS HIGH_POWER)=0.34"
RULE2 = "IF (OXI IS medium_flow) THEN P(POWER IS LOW_POWER)=0.33, P(POWER IS MEDIUM_POWER)=0.33, P(POWER IS HIGH_POWER)=0.34"
RULE3 = "IF (NOT (OXI IS low_flow)) THEN P(POWER IS LOW_POWER)=0.33, P(POWER IS MEDIUM_POWER)=0.33, P(POWER IS HIGH_POWER)=0.34"
FS.add_proba_rules([RULE1, RULE2, RULE3])

# Set Variable, perform probabilistic inference and print output values.
FS.set_variable("OXI", .51)
print(FS.inference())

```

## Installation

`pip install simpful`

## Citing Simpful

If you find Simpful useful for your research, please cite our work as follows:

Spolaor S., Fuchs C., Cazzaniga P., Kaymak U., Besozzi D., Nobile M.S.: Simpful: a user-friendly Python library for fuzzy logic, International Journal of Computational Intelligence Systems, 2020 (accepted)

## Further info
Created by Marco S. Nobile at the Eindhoven University of Technology and Simone Spolaor at the University of Milano-Bicocca. Usage information can be found on the [wiki page](https://github.com/aresio/simpful/wiki).
If you need further information, please write an e-mail at: m.s.nobile@tue.nl.