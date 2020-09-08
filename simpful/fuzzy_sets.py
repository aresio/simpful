from numpy import array, linspace
import numpy as np
from scipy.interpolate import interp1d

class MF_object(object):

	def __init__(self):
		pass

	def __call__(self, x):
		ret = self._execute(x)
		return min(1, max(0, ret))
		

###############################
# USEFUL PRE-BAKED FUZZY SETS #
###############################

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class Triangular_MF(MF_object):

	def __init__(self, a=0, b=0.5, c=1):
		"""
		Creates a triangular membership function.
		Requires a <= b <= c and the semantics is the following:

		1	|   .
			|  / \
			| /   \
		0	|/     \
			---------
	         a  b  c

		"""
		self._a = a
		self._b = b
		self._c = c
		if (a>b):
			raise("Error in triangular fuzzy set: a=%.2f should be <= b=%.2f" % (a,b))
		elif (b>c):
			raise("Error in triangular fuzzy set: b=%.2f should be <= c=%.2f" % (b,c))
		
	def _execute(self, x):
		if x < self._b:
			if self._a != self._b:
				return (x-self._a) * (1/(self._b-self._a))
			else:
				return 1
		else:
			if self._b != self._c:
				return 1 + (x-self._b) * (-1/(self._c-self._b))
			else:
				return 1

class Trapezoidal_MF(MF_object):

	def __init__(self, a=0, b=0.25, c=0.75, d=1):
		"""
		Creates a trapezoidal membership function.
		Requires a <= b <= c <= d.
		"""
		self._a = a
		self._b = b
		self._c = c
		self._d = d
		
	def _execute(self, x):
		if x < self._b:
			if self._a != self._b:
				return (x-self._a) * (1/(self._b-self._a))
			else:
				return 1
		elif x >= self._b and x <= self._c:
			return 1
		else:
			if self._c != self._d:
				return 1 + (x-self._c) * (-1/(self._d-self._c))
			else:
				return 1

class Sigmoid_MF(MF_object):

	def __init__(self, c=0, a=1):
		"""
		Creates a sigmoidal membership function.
		"""
		self._c = c
		self._a = a
		
	def _execute(self, x):
		return 1.0/(1.0 + np.exp(-self._a*(x-self._c))) 

class InvSigmoid_MF(MF_object):

	def __init__(self, c=0, a=1):
		"""
		Creates an inversed sigmoid membership function.
		"""
		self._c = c
		self._a = a
		
	def _execute(self, x):
		return 1.0 - 1.0/(1.0 + np.exp(-self._a*(x-self._c))) 

class Gaussian_MF(MF_object):

	def __init__(self, mu, sigma):
		"""
		Creates a gaussian membership function.
		"""
		self._mu = mu
		self._sigma = sigma

	def _execute(self, x):
		return gaussian(x, self._mu, self._sigma)

class InvGaussian_MF(MF_object):

	def __init__(self, mu, sigma):
		"""
		Creates an inversed gaussian membership function.
		"""
		self._mu = mu
		self._sigma = sigma

	def _execute(self, x):
		return 1.-gaussian(x, self._mu, self._sigma)

class DoubleGaussian_MF(MF_object):

	def __init__(self, mu1, sigma1, mu2, sigma2):
		"""
		Creates a double gaussian membership function.
		"""
		self._mu1 = mu1
		self._sigma1 = sigma1
		self._mu2 = mu2
		self._sigma2 = sigma2

	def _execute(self, x):
		first = gaussian(x, self._mu1, self._sigma1)
		second = gaussian(x, self._mu2, self._sigma2)
		return first*second


class FuzzySet(object):

	def __init__(self, points=None, function=None, term="", high_quality_interpolate=True, verbose=False):
		"""
		Creates a a new fuzzy set.
		Args:
			points: list of points to define a polygonal fuzzy sets.
			Each point is defined  as a list of two coordinates in the universe of discourse/membership degree space.
			function: function to define a non-polygonal fuzzy set.
			Supports pre-implemented membership functions Sigmoid_MF, InvSigmoid_MF, Gaussian_MF, InvGaussian_MF,
			DoubleGaussian_MF or user-defined functions.
			term: string representing the linguistic term to be associated to the fuzzy set.
			high_quality_interpolate: True/False, toggles high quality interpolation.
			verbose: True/False, toggles verbose mode.
		"""

		self._term = term

		if points is None and function is not None:
			self._type = "function"
			self._funpointer = function
			#self._funargs	= function['args']
			return


		if len(points)<2: 
			print("ERROR: more than one point required")
			exit(-1)
		if term=="":
			print("ERROR: please specify a linguistic term")
			exit(-3)
		self._type = "pointbased"
		self._high_quality_interpolate = high_quality_interpolate
		self._points = array(points)
		

	def get_value(self, v):
		""" Return the membership value of v to this Fuzzy Set. """

		if self._type == "function":
			return self._funpointer(v)

		if self._high_quality_interpolate:
			return self.get_value_slow(v)
		else:
			return self.get_value_fast(v)


	def get_term(self):
		return self._term


	def get_value_cut(self, v, cut):
		""" Return the membership value of v to this Fuzzy Set, capped to the cut value. """

		return min(cut, self.get_value(v))
		

	def get_value_slow(self, v):		
		f = interp1d(self._points.T[0], self._points.T[1], 
			bounds_error=False, fill_value=(self._points.T[1][0], self._points.T[1][-1]))
		result = f(v)
		return(result)

	def get_value_fast(self, v):
		x = self._points.T[0]
		y = self._points.T[1]
		N = len(x)
		if v<x[0]: return self._points.T[1][0] 
		for i in range(N-1):
			if (x[i]<= v) and (v <= x[i+1]):
				return self._fast_interpolate(x[i], y[i], x[i+1], y[i+1], v)
		return self._points.T[1][-1] # fallback for values outside the Universe of the discourse

	def _fast_interpolate(self, x0, y0, x1, y1, x):
		#print(x0, y0, x1, y1, x); exit()
		return y0 + (x-x0) * ((y1-y0)/(x1-x0))


	def integrate(self, x0, x1, cut=1):
		import scipy.integrate as integrate
		result = integrate.quad(self.get_value_cut, x0, x1, args=(cut))
		return result[0]