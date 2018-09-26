# simpful
A simple python library for fuzzy logic reasoning

## Usage

```
from simpful import *

# A simple fuzzy model describing how the heating power of a gas burner depends on the oxygen supply.

FR = FuzzyReasoner()

RULE1 = " IF (OXI IS low) THEN (POWER IS LOW_POWER)"
RULE2 = " IF (OXI IS medium) THEN (POWER IS MEDIUM_POWER)"
RULE3 = " IF (OXI IS high) THEN (POWER IS HIGH_POWER)"

FR._crispvalues["LOW_POWER"] 	  = 0
FR._crispvalues["MEDIUM_POWER"] = 25
FR._crispvalues["HIGH_POWER"] 	= 100

FS_1 = FuzzySet( points=[[0, 1.],  [1., 1.],  [1.5, 0]],          term="low_flow" )
FS_2 = FuzzySet( points=[[0.5, 0], [1.5, 1.], [2.5, 1], [3., 0]], term="medium_flow" )
FS_3 = FuzzySet( points=[[2., 0],  [2.5, 1.], [3., 1.]],          term="high_flow" )
FR._mfs["OXI"] = MembershipFunction( [FS_1, FS_2, FS_3], concept="OXI" )

FR.add_rules([RULE1, RULE2, RULE3])

# set antecedents values, perform Sugeno inference and print output values
FR.set_variable("OXI", .4)
print FR.Sugeno_inference()
```

## Installation

`pip install simpful`

## Further info
Created by Marco S. Nobile and Simone Spolaor at the University of Milano-Bicocca, Italy. 

If you need further information, please drop a line at: nobile@disco.unimib.it. 
