# simpful
A Python library for fuzzy logic reasoning, designed to provide a simple and lightweight API, as close as possible to natural language.
Simpful supports Mamdani and Sugeno reasoning of any order, parsing any complex fuzzy rules involving AND, OR, and NOT operators, using arbitrarily shaped fuzzy sets.

## Usage

```
import simpful as sf

# A simple fuzzy model describing how the heating power of a gas burner depends on the oxygen supply.

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
print (FS.Sugeno_inference(['POWER']))
```

## Installation

`pip install simpful`

## Further info
Created by Marco S. Nobile at the Eindhoven University of Technology and Simone Spolaor at the University of Milano-Bicocca. 

If you need further information, please write an e-mail at: m.s.nobile@tue.nl.
