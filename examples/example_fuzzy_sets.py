import simpful as sf

# A showcase of available fuzzy sets.

# Crisp
C_1 = sf.CrispSet(a=0, b=5, term="low")
C_2 = sf.CrispSet(a=5, b=10, term="high")
sf.LinguisticVariable([C_1, C_2], universe_of_discourse=[0, 10]).plot()

# Point-based polygon
P_1 = sf.FuzzySet(points=[[2.0, 1.0], [4.0, 0.25], [6.0, 0.0]], term="low")
P_2 = sf.FuzzySet(points=[[2.0, 0.0], [4.0, 0.25], [6.0, 1.0]], term="high")
sf.LinguisticVariable([P_1, P_2], universe_of_discourse=[0, 10]).plot()

# Triangle
Tri_1 = sf.TriangleFuzzySet(a=0, b=0, c=5, term="low")
Tri_2 = sf.TriangleFuzzySet(a=0, b=5, c=10, term="medium")
Tri_3 = sf.TriangleFuzzySet(a=5, b=10, c=10, term="high")
sf.LinguisticVariable([Tri_1, Tri_2, Tri_3], universe_of_discourse=[0, 10]).plot()

# Trapezoid
Tra_1 = sf.TrapezoidFuzzySet(a=0, b=0, c=2, d=4, term="low")
Tra_2 = sf.TrapezoidFuzzySet(a=2, b=4, c=6, d=8, term="medium")
Tra_3 = sf.TrapezoidFuzzySet(a=6, b=8, c=10, d=10, term="high")
sf.LinguisticVariable([Tra_1, Tra_2, Tra_3], universe_of_discourse=[0, 10]).plot()

# Gaussian
G_1 = sf.GaussianFuzzySet(mu=5, sigma=2, term="medium")
G_2 = sf.InvGaussianFuzzySet(mu=5, sigma=2, term="not medium")
sf.LinguisticVariable([G_1, G_2], universe_of_discourse=[0, 10]).plot()

# Double Gaussian
DG_1 = sf.DoubleGaussianFuzzySet(mu1=1, sigma1=0.1, mu2=1, sigma2=1, term="low")
DG_2 = sf.DoubleGaussianFuzzySet(mu1=3.5, sigma1=1, mu2=6, sigma2=5, term="high")
sf.LinguisticVariable([DG_1, DG_2], universe_of_discourse=[0, 10]).plot()

# Sigmoid
S_1 = sf.InvSigmoidFuzzySet(c=5, a=2, term="low")
S_2 = sf.SigmoidFuzzySet(c=5, a=2, term="high")
sf.LinguisticVariable([S_1, S_2], universe_of_discourse=[0, 10]).plot()

# Function-based fuzzy set
import numpy as np
def fun1(x):
	return 0.5*np.cos(0.314*x)+0.5
def fun2(x):
	return 0.5*np.sin(0.314*x-1.5)+0.5

F_1 = sf.FuzzySet(function=fun1, term="low")
F_2 = sf.FuzzySet(function=fun2, term="high")
sf.LinguisticVariable([F_1, F_2], universe_of_discourse=[0, 10]).plot()

# Singletons set
Ss_1 = sf.SingletonsSet(pairs=[[1.0, 0.2], [2.0, 0.8], [3.0, 0.4]], term="low")
Ss_2 = sf.SingletonsSet(pairs=[[3.0, 0.3], [5.0, 0.9], [6.0, 0.1]], term="high")
sf.LinguisticVariable([Ss_1, Ss_2], universe_of_discourse=[0, 10]).plot()
