from simpful import *
import matplotlib.pylab as plt
from numpy import linspace, array

FS = FuzzySystem()

S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="poor")
S_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="good")
S_3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="excellent")
FS.add_linguistic_variable("Service", LinguisticVariable([S_1, S_2, S_3], concept="Service quality", universe_of_discourse=[0,10]))

F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="rancid")
F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="delicious")
FS.add_linguistic_variable("Food", LinguisticVariable([F_1, F_2], concept="Food quality", universe_of_discourse=[0,10]))

T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small")
T_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="average")
T_3 = FuzzySet(function=Trapezoidal_MF(a=10, b=20, c=25, d=25), term="generous")
FS.add_linguistic_variable("Tip", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0,25]))

R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)"
R2 = "IF (Service IS good) THEN (Tip IS average)"
R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
FS.add_rules([R1, R2, R3])

# Plotting surface
xs = []
ys = []
zs = []
DIVs = 20
for x in linspace(0,10,DIVs):
    for y in linspace(0,10,DIVs):
        FS.set_variable("Food", x)
        FS.set_variable("Service", y)
        tip = FS.inference()['Tip']
        xs.append(x)
        ys.append(y)
        zs.append(tip)
xs = array(xs)
ys = array(ys)
zs = array(zs)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xx, yy = plt.meshgrid(xs,ys)

ax.plot_trisurf(xs,ys,zs, vmin=0, vmax=25, cmap='gnuplot2')
ax.set_xlabel("Food")
ax.set_ylabel("Service")
ax.set_zlabel("Tip")
ax.set_title("Simpful", pad=20)
ax.set_zlim(0, 25)
plt.tight_layout()
plt.show()