from numpy import array, linspace
import numpy as np
from scipy.interpolate import interp1d

class MF_object(object):

	def __init__(self):
		pass

	def __call__(self, x):
		ret = self._execute(x)
		return min(1, max(0, ret))
		

#########################################
# USEFUL PRE-BAKED MEMBERSHIP FUNCTIONS #
#########################################

def _gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class Triangular_MF(MF_object):
	"""
		Creates a normalized triangular membership function.
		Requires a <= b <= c and the semantics is the following:
		::

			1	|   .
				|  / \\
				| /   \\
			0	|/     \\
				---------
				 a  b  c

	    Args:
			a: universe of discourse coordinate of the leftmost vertex.
			b: universe of discourse coordinate of the upper vertex.
			c: universe of discourse coordinate of the rightmost vertex.
	"""

	def __init__(self, a=0, b=0.5, c=1):
		self._a = a
		self._b = b
		self._c = c
		if (a>b):
			raise Exception("Error in triangular fuzzy set: a=%.2f should be <= b=%.2f" % (a,b))
		elif (b>c):
			raise Exception("Error in triangular fuzzy set: b=%.2f should be <= c=%.2f" % (b,c))
		
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

	def __repr__(self):
		return "<Triangular MF (%f, %f, %f)>"% (self._a, self._b, self._c)

class Trapezoidal_MF(MF_object):
	"""
		Creates a normalized trapezoidal membership function.
		Requires a <= b <= c <= d.

		Args:
			a: universe of discourse coordinate of the leftmost vertex.
			b: universe of discourse coordinate of the upper left vertex.
			c: universe of discourse coordinate of the upper right vertex.
			d: universe of discourse coordinate of the rightmost vertex.
	"""

	def __init__(self, a=0, b=0.25, c=0.75, d=1):
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
	"""
		Creates a sigmoidal membership function.

		Args:
			c: universe of discourse coordinate of the inflection point.
			a: steepness of the curve.
	"""

	def __init__(self, c=0, a=1):
		self._c = c
		self._a = a
		
	def _execute(self, x):
		return 1.0/(1.0 + np.exp(-self._a*(x-self._c))) 

class InvSigmoid_MF(MF_object):
	"""
		Creates an inversed sigmoid membership function.

		Args:
			c: universe of discourse coordinate of inflection point.
			a: steepness of the curve.
	"""
	
	def __init__(self, c=0, a=1):
		self._c = c
		self._a = a
		
	def _execute(self, x):
		return 1.0 - 1.0/(1.0 + np.exp(-self._a*(x-self._c))) 

class Gaussian_MF(MF_object):
	"""
		Creates a Gaussian membership function.

		Args:
			mu: mean of the distribution.
			sigma: standard deviation of the distribution.
	"""
	
	def __init__(self, mu, sigma):
		self._mu = mu
		self._sigma = sigma
		if sigma<=0: print("WARNING: sigma should be strictly positive, Simpful received sigma=%f" % sigma)

	def _execute(self, x):
		return _gaussian(x, self._mu, self._sigma)

class InvGaussian_MF(MF_object):
	"""
		Creates an inversed Gaussian membership function.

		Args:
			mu: mean of the distribution.
			sigma: standard deviation of the distribution.
	"""

	def __init__(self, mu, sigma):
		self._mu = mu
		self._sigma = sigma

	def _execute(self, x):
		return 1.-_gaussian(x, self._mu, self._sigma)

class DoubleGaussian_MF(MF_object):
	"""
		Creates a double Gaussian membership function.

		Args:
			mu1: mean of the first distribution.
			sigma1: standard deviation of the first distribution.
			mu2: mean of the second distribution.
			sigma2: standard deviation of the second distribution.
	"""

	def __init__(self, mu1, sigma1, mu2, sigma2):
		self._mu1 = mu1
		self._sigma1 = sigma1
		if sigma1<=0: print("WARNING: sigma should be strictly positive, Simpful received sigma1=%f" % sigma1)
		self._mu2 = mu2
		self._sigma2 = sigma2
		if sigma2<=0: print("WARNING: sigma should be strictly positive, Simpful received sigma2=%f" % sigma2)


	def _execute(self, x):
		first = _gaussian(x, self._mu1, self._sigma1)
		second = _gaussian(x, self._mu2, self._sigma2)

		if x <= self._mu1:
			return first
		elif x>= self._mu2:
			return second
		else:
			return 1.0


class Crisp_MF(MF_object):
	"""
		Creates a crisp membership function.

		Args:
			a: left extreme value of the set.
			b: right extreme value of the set.
	"""
	
	def __init__(self, a, b):
		self._left = a
		self._right = b

	def _execute(self, x):
		if x<self._left: return 0
		if x>self._right: return 0
		return 1

class FuzzySet(object):
	"""
		Creates a new fuzzy set.

		Args:
			points: list of points to define a polygonal fuzzy sets. Each point is defined as a list of two coordinates in the universe of discourse/membership degree space.
			function: function to define a non-polygonal fuzzy set. Supports pre-implemented membership functions Sigmoid_MF, InvSigmoid_MF, Gaussian_MF, InvGaussian_MF, DoubleGaussian_MF, Triangle_MF, Trapezoidal_MF or user-defined functions.
			term: string representing the linguistic term to be associated to the fuzzy set.
			high_quality_interpolate: True/False, toggles high quality interpolation for point-based fuzzy sets. Default value is set to False.
			boundary_values: list of two membership values for point-based fuzzy sets. The first and second value are used to fill in values at the left-side and right-side of the fuzzy set, respectively. If None (default value), fuzzy sets will be considered as shouldered.
			verbose: True/False, toggles verbose mode.
	"""

	def __init__(self, points=None, function=None, term="", high_quality_interpolate=False, boundary_values=None, verbose=False):
		self._term = term

		if points is None and function is not None:
			self._type = "function"
			self._funpointer = function
			#self._funargs	= function['args']
			return


		if len(points)<2: 
			raise Exception("ERROR: more than one point required")
		if term=="":
			raise Exception("ERROR: please specify a linguistic term")
		for p in points:
			if len(p)>2: raise Exception("ERROR: one fuzzy set named \""+self._term+"\" has more than two coordinates.")
		self._type = "pointbased"
		self._high_quality_interpolate = high_quality_interpolate
		self._points = array(points)

		if boundary_values == None:
			self.boundary_values = [self._points.T[1][0], self._points.T[1][-1]]
		else:
			if len(boundary_values) == 2 and all(isinstance(x, (int, float)) for x in boundary_values): 
				self.boundary_values = boundary_values
			else: 
				raise Exception("ERROR: boundary_values must be a list of two numbers")
		

	def __repr__(self):
		return "<Fuzzy set (%s), term='%s'>" % (self._type, self._term)


	def get_value(self, v):
		""" Return the membership value of v to this Fuzzy Set.

			Args:
				v: element of the universe of discourse.

			Returns: 
				The membership value of v to this Fuzzy Set.
		"""
		if self._type == "function":
			return self._funpointer(v)

		if self._high_quality_interpolate:
			return self.get_value_slow(v)
		else:
			return self.get_value_fast(v)


	def get_term(self):
		""" Return the linguistic term associated to this fuzzy set.
		"""
		return self._term


	def get_value_cut(self, v, cut):
		""" Return the membership value of v to this Fuzzy Set, capped to the cut value.

			Args:
				v: element of the universe of discourse.
				cut: alpha cut of the fuzzy set.
		"""

		return min(cut, self.get_value(v))
		

	def get_value_slow(self, v):
		f = interp1d(self._points.T[0], self._points.T[1], 
			bounds_error=False, fill_value=(self.boundary_values[0], self.boundary_values[1]))
		result = f(v)
		return(result)

	def get_value_fast(self, v):
		x = self._points.T[0]
		y = self._points.T[1]
		N = len(x)
		if v<x[0]: return self.boundary_values[0] # fallback for values outside the Universe of the discourse
		for i in range(N-1):
			if (x[i]<= v) and (v <= x[i+1]):
				return self._fast_interpolate(x[i], y[i], x[i+1], y[i+1], v)
		return self.boundary_values[1] # fallback for values outside the Universe of the discourse

	def _fast_interpolate(self, x0, y0, x1, y1, x):
		return y0 + (x-x0) * ((y1-y0)/(x1-x0))


	def integrate(self, x0, x1, cut=1):
		import scipy.integrate as integrate
		result = integrate.quad(self.get_value_cut, x0, x1, args=(cut))
		return result[0]

	def set_params():
		print("Attention: this is a virtual method for setting parameters of pre-baked fuzzy sets.")

	def set_points(self, points):
		"""
		Changes points of the point-based fuzzy set.

		Args:
			points: a list of points to define a polygonal fuzzy sets. Each point is defined as a list of two coordinates in the universe of discourse/membership degree space.
		"""
		if len(points)<2: 
			raise Exception("ERROR: more than one point required")
		if self._type == "function":
			print("WARNING: the fuzzy set named \""+self._term+"\" was converted from function-based to point-based.")
		for p in points:
			if len(p)>2: raise Exception("ERROR: one point in \""+self._term+"\" has more than two coordinates.")
		self._type = "pointbased"
		self._high_quality_interpolate = False
		self._points = array(points)
		self.boundary_values = [self._points.T[1][0], self._points.T[1][-1]]

###############################
# USEFUL PRE-BAKED FUZZY SETS #
###############################

class TriangleFuzzySet(FuzzySet):
	"""
		Creates a new triangular fuzzy set.

		Args:
			a: universe of discourse coordinate of the leftmost vertex.
			b: universe of discourse coordinate of the upper vertex.
			c: universe of discourse coordinate of the rightmost vertex.
			term: string representing the linguistic term to be associated to the fuzzy set.
	"""

	def __init__(self, a, b, c, term):
		triangle_MF = Triangular_MF(a,b,c)
		super().__init__(function=triangle_MF, term=term)

	def set_params(self, a=None, b=None, c=None):
		"""
		Changes parameters of the triangular fuzzy set.

		Args:
			a: universe of discourse coordinate of the leftmost vertex.
			b: universe of discourse coordinate of the upper vertex.
			c: universe of discourse coordinate of the rightmost vertex.
		"""
		if a is not None: self._funpointer._a = a
		if b is not None: self._funpointer._b = b
		if c is not None: self._funpointer._c = c

class TrapezoidFuzzySet(FuzzySet):
	"""
		Creates a new trapezoidal fuzzy set.

		Args:
			a: universe of discourse coordinate of the leftmost vertex.
			b: universe of discourse coordinate of the upper left vertex.
			c: universe of discourse coordinate of the upper right vertex.
			d: universe of discourse coordinate of the rightmost vertex.
			term: string representing the linguistic term to be associated to the fuzzy set.
	"""

	def __init__(self, a, b, c, d, term):
		trapezoid_MF = Trapezoidal_MF(a,b,c,d)
		super().__init__(function=trapezoid_MF, term=term)

	def set_params(self, a=None, b=None, c=None, d=None):
		"""
		Changes parameters of the trapezoidal fuzzy set.

		Args:
			a: universe of discourse coordinate of the leftmost vertex.
			b: universe of discourse coordinate of the upper left vertex.
			c: universe of discourse coordinate of the upper right vertex.
			d: universe of discourse coordinate of the rightmost vertex.
		"""
		if a is not None: self._funpointer._a = a
		if b is not None: self._funpointer._b = b
		if c is not None: self._funpointer._c = c
		if d is not None: self._funpointer._d = d

class SigmoidFuzzySet(FuzzySet):
	"""
		Creates a new sigmoidal fuzzy set.

		Args:
			c: universe of discourse coordinate of inflection point.
			a: steepness of the curve.
			term: string representing the linguistic term to be associated to the fuzzy set.
	"""

	def __init__(self, c, a, term):
		sigmoid_MF = Sigmoid_MF(c,a)
		super().__init__(function=sigmoid_MF, term=term)

	def set_params(self, c=None, a=None):
		"""
		Changes parameters of the sigmoidal fuzzy set.

		Args:
			c: universe of discourse coordinate of inflection point.
			a: steepness of the curve.
		"""
		if c is not None: self._funpointer._c = c
		if a is not None: self._funpointer._a = a

class InvSigmoidFuzzySet(FuzzySet):
	"""
		Creates a new inversed sigmoidal fuzzy set.

		Args:
			c: universe of discourse coordinate of inflection point.
			a: steepness of the curve.
			term: string representing the linguistic term to be associated to the fuzzy set.
	"""

	def __init__(self, c, a, term):
		invsigmoid_MF = InvSigmoid_MF(c,a)
		super().__init__(function=invsigmoid_MF, term=term)

	def set_params(self, c=None, a=None):
		"""
		Changes parameters of the inversed sigmoidal fuzzy set.

		Args:
			c: universe of discourse coordinate of inflection point.
			a: steepness of the curve.
		"""
		if c is not None: self._funpointer._c = c
		if a is not None: self._funpointer._a = a

class GaussianFuzzySet(FuzzySet):
	"""
		Creates a new Gaussian fuzzy set.

		Args:
			mu: mean of the distribution.
			sigma: standard deviation of the distribution.
			term: string representing the linguistic term to be associated to the fuzzy set.
	"""
	
	def __init__(self, mu, sigma, term):
		gaussian_MF = Gaussian_MF(mu,sigma)
		super().__init__(function=gaussian_MF, term=term)

	def set_params(self, mu=None, sigma=None):
		"""
		Changes parameters of the Gaussian fuzzy set.

		Args:
			mu: mean of the distribution.
			sigma: standard deviation of the distribution.
		"""
		if mu is not None: self._funpointer._mu = mu
		if sigma is not None: self._funpointer._sigma = sigma

class InvGaussianFuzzySet(FuzzySet):
	"""
		Creates a new inversed Gaussian fuzzy set.

		Args:
			mu: mean of the distribution.
			sigma: standard deviation of the distribution.
			term: string representing the linguistic term to be associated to the fuzzy set.
	"""
	
	def __init__(self, mu, sigma, term):
		invgaussian_MF = InvGaussian_MF(mu,sigma)
		super().__init__(function=invgaussian_MF, term=term)

	def set_params(self, mu=None, sigma=None):
		"""
		Changes parameters of the inversed Gaussian fuzzy set.

		Args:
			mu: mean of the distribution.
			sigma: standard deviation of the distribution.
		"""
		if mu is not None: self._funpointer._mu = mu
		if sigma is not None: self._funpointer._sigma = sigma

class DoubleGaussianFuzzySet(FuzzySet):
	"""
		Creates a new double Gaussian fuzzy set.
		
		Args:
			mu1: mean of the first distribution.
			sigma1: standard deviation of the first distribution.
			mu2: mean of the second distribution.
			sigma2: standard deviation of the second distribution.
			term: string representing the linguistic term to be associated to the fuzzy set.
	"""

	def __init__(self, mu1, sigma1, mu2, sigma2,  term):
		doublegaussian_MF = DoubleGaussian_MF(mu1, sigma1, mu2, sigma2)
		super().__init__(function=doublegaussian_MF, term=term)

	def set_params(self, mu1=None, sigma1=None, mu2=None, sigma2=None):
		"""
		Changes parameters of the double Gaussian fuzzy set.

		Args:
			mu1: mean of the first distribution.
			sigma1: standard deviation of the first distribution.
			mu2: mean of the second distribution.
			sigma2: standard deviation of the second distribution.
		"""
		if mu1 is not None: self._funpointer._mu1 = mu1
		if sigma1 is not None: self._funpointer._sigma1 = sigma1
		if mu2 is not None: self._funpointer._mu2 = mu2
		if sigma2 is not None: self._funpointer._sigma2 = sigma2

class CrispSet(FuzzySet):
	"""
		Creates a new crisp set.
		
		Args:
			a: left extreme value of the set.
			b: right extreme value of the set.
			term: string representing the linguistic term to be associated to the crisp set.
	"""

	def __init__(self, a, b, term):
		crisp_MF = Crisp_MF(a, b)
		super().__init__(function=crisp_MF, term=term)

	def set_params(self, a=None, b=None):
		"""
		Changes parameters of the crisp set.

		Args:
			a: left extreme value of the set.
			b: right extreme value of the set.
		"""
		if a is not None: self._funpointer._left = a
		if b is not None: self._funpointer._right = b