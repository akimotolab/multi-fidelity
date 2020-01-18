# This code based on the following python code and is modified by 
# Youhei Akimoto August 2019.
# ===================================================================
# A 200 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
# Updated by Niels Aage February 2016
# 
# Code is described in the following reference:
# Erik Andreassen, Anders Clausen, Mattias Schevenels, Boyan S. Lazarov, Ole Sigmund.
# Efficient topology optimization in MATLAB using 88 lines of code
# ===================================================================
from __future__ import division
import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.cholmod


class ComplianceMinimization:
	"""Compliance Mnimization by Density-based Topology Optimization
	
	minimize   : c(x)
	subject to : v(x) = volfrac
	             0 <= x <= 1,
	where
		x : real vector whose elements represent the densities of elements in the design domain
		c : compliance
		v : volume fraction over the volume of the design domain, [0, 1]
	"""
	def __init__(self, nelx=180, nely=60, volfrac=0.4, penal=3.0):
		"""Density-based Compliance Minimization Problem

		Parameters
		----------
		nelx, nely : int
			number of elements in each coordinate
		volfrac : float, in [0, 1]
			total amount of material, V(x), is constrained to volfrac * V(1)
		penal : float, default = 3
			penalization factor to ensure black-and-white solutions
		"""
		self.nelx = nelx
		self.nely = nely
		self.volfrac = volfrac
		self.penal = penal
		# Max and min stiffness
		self.Emin = 1e-9
		self.Emax = 1.0
		# dofs:
		self.ndof = 2 * (nelx + 1) * (nely + 1)
		# FE: Build the index vectors for the for coo matrix format.
		self.KE = self._lk()
		self.edofMat = np.zeros((nelx*nely,8),dtype=int)
		for elx in range(nelx):
			for ely in range(nely):
				el = ely+elx*nely
				n1=(nely+1)*elx+ely
				n2=(nely+1)*(elx+1)+ely
				self.edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
		# Construct the index pointers for the coo format
		self.iK = np.kron(self.edofMat,np.ones((8,1))).flatten()
		self.jK = np.kron(self.edofMat,np.ones((1,8))).flatten()    
		# BC's and support
		self.dofs=np.arange(2*(nelx+1)*(nely+1))
		self.fixed=np.union1d(self.dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
		self.free=np.setdiff1d(self.dofs,self.fixed)
		# Solution and RHS vectors
		self.f=np.zeros((self.ndof,1))
		self.u=np.zeros((self.ndof,1))
		# Set load
		self.f[1,0]=-1
		# Memory Allocation
		self.dv = np.ones(nely*nelx)
		self.dc = np.ones(nely*nelx)
		self.ce = np.ones(nely*nelx)

	def __call__(self, xPhys):
		"""Compute objective c(x) and volume faction v(x) and their gradient

		Parameters
		----------
		xPhys : 1d array of float
			density vector
		
		Returns
		-------
		obj (float)   : c(x)
		vol (float)   : v(x)
		dc (1d array) : dc / dx
		dv (1d array) : dv / dx
		"""
		obj, self.dc = self.compliance(xPhys)
		vol, self.dv = self.volume_fraction(xPhys)
		return obj, vol, self.dc, self.dv

	def compliance(self, x):
		"""Compute the compliance

		Parameters
		----------
		x : 1d array of float
			density vector
		
		Returns
		-------
		obj (float)   : c(x)
		dc (1d array) : dc / dx
		"""
		# Setup and solve FE problem
		sK=((self.KE.flatten()[np.newaxis]).T*(self.Emin+(x)**self.penal*(self.Emax-self.Emin))).flatten(order='F')
		K = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()
		# Remove constrained dofs from matrix and convert to coo
		K = self._deleterowcol(K,self.fixed,self.fixed).tocoo()
		# Solve system 
		K = cvxopt.spmatrix(K.data,K.row.astype(np.int),K.col.astype(np.int))
		B = cvxopt.matrix(self.f[self.free,0])
		cvxopt.cholmod.linsolve(K,B)
		self.u[self.free,0]=np.array(B)[:,0] 
		# Objective and sensitivity
		self.ce[:] = (np.dot(self.u[self.edofMat].reshape(self.nelx*self.nely,8),self.KE) * self.u[self.edofMat].reshape(self.nelx*self.nely,8) ).sum(1)
		obj=( (self.Emin+x**self.penal*(self.Emax-self.Emin))*self.ce ).sum()
		self.dc[:] = (-self.penal * x**(self.penal - 1) * (self.Emax - self.Emin)) * self.ce
		return obj, self.dc

	def volume_fraction(self, x):
		"""Compute the volume fraction

		Parameters
		----------
		x : 1d array of float
			density vector
		
		Returns
		-------
		vol (float)   : v(x)
		dv (1d array) : dv / dx
		"""
		vol = np.mean(x)
		self.dv[:] = 1.0 / (self.nelx * self.nely)
		return vol, self.dv

	def plot(self, xPhys, name):
		"""Produce the structure image and save it
		
		Parameters
		----------
		xPhys : 1d array
			density vector
		name : str
			output file name with extension (e.g., .pdf)
		"""
		nelx, nely = self.nelx, self.nely
		fig,ax = plt.subplots()
		im = ax.imshow(-xPhys.reshape((nelx,nely)).T, 
					   cmap='gray',
					   interpolation='none',
					   norm=colors.Normalize(vmin=-1, vmax=0))
		plt.savefig(name)
		plt.close()

	def _lk(self):
		#element stiffness matrix
		E=1
		nu=0.3
		k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
		KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
		[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
		[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
		[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
		[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
		[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
		[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
		[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
		return (KE)

	def _deleterowcol(self, A, delrow, delcol):
		# Assumes that matrix is in symmetric csc form !
		m = A.shape[0]
		keep = np.delete (np.arange(0, m), delrow)
		A = A[keep, :]
		keep = np.delete (np.arange(0, m), delcol)
		A = A[:, keep]
		return A  
	
class SIMP:
	"""Modified SIMP Approach for Density-based Compliance Minimization"""

	def __init__(self, nelx=180, nely=60, volfrac=0.4, rmin=5.4, ft=1):
		"""
		Parameters
		----------
		nelx, nely : int
			number of elements in each coordinate
		volfrac : float, in [0, 1]
			total amount of material, V(x), is constrained to volfrac * V(1)
		rmin : float
			filter radius
		ft : int
			0 : Sensitivity Filter
			1 : Density Filter
		"""
		self.nelx = nelx
		self.nely = nely
		self.rmin = rmin
		self.volfrac = volfrac
		self.ft = ft

		# Filter: Build (and assemble) the index+data vectors for the coo matrix format
		nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
		iH = np.zeros(nfilter)
		jH = np.zeros(nfilter)
		sH = np.zeros(nfilter)
		cc=0
		for i in range(nelx):
			for j in range(nely):
				row=i*nely+j
				kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
				kk2=int(np.minimum(i+np.ceil(rmin),nelx))
				ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
				ll2=int(np.minimum(j+np.ceil(rmin),nely))
				for k in range(kk1,kk2):
					for l in range(ll1,ll2):
						col=k*nely+l
						fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
						iH[cc]=row
						jH[cc]=col
						sH[cc]=np.maximum(0.0,fac)
						cc=cc+1
		# Finalize assembly and convert to csc format
		self.H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()	
		self.Hs=self.H.sum(1)
		# Allocate design variables (as array), initialize and allocate sens.
		self.x=volfrac * np.ones(nely*nelx,dtype=float)
		self.xold=self.x.copy()
		self.xPhys=self.x.copy()
		self.dc = np.ones(nely*nelx)
		self.dv = np.ones(nely*nelx)
		self.g=0 # must be initialized to use the NGuyen/Paulino OC approach

	def __call__(self, dc, dv):
		"""Update design variables
		
		Parameters
		----------
		dc, dv : 1d array
			dc / dx and dv / dx

		Return
		------
		xPhys (1d array) : updated variable
		"""
		# Sensitivity filtering:
		if self.ft==0:
			self.dc[:] = np.asarray((self.H*(self.x*dc))[np.newaxis].T/self.Hs)[:,0] / np.maximum(0.001,self.x)
		elif self.ft==1:
			self.dc[:] = np.asarray(self.H*(dc[np.newaxis].T/self.Hs))[:,0]
			self.dv[:] = np.asarray(self.H*(dv[np.newaxis].T/self.Hs))[:,0]

		# Optimality criteria
		self.xold[:]=self.x
		(self.x[:], self.g) = self._oc(self.nelx,self.nely,self.x,self.volfrac,self.dc,self.dv,self.g)

		# Filter design variables
		if self.ft==0:   self.xPhys[:]=self.x
		elif self.ft==1:	self.xPhys[:]=np.asarray(self.H*self.x[np.newaxis].T/self.Hs)[:,0]
		
		return self.xPhys

	def solve(self, co):
		"""Optimize Density-based Compliance Minimization Problem

		Parameter
		---------
		co : instance of `ComplianceMinimization`
			problem to be solved
		"""
		nelx, nely = self.nelx, self.nely
		change = 0.0
		for t in range(2000):
			obj, vol, dc, dv = co(self.xPhys)
			# Write iteration history to screen
			print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(t, obj, vol, change))
			co.plot(self.xPhys, "fig/itr{}.pdf".format(t))
			# Update design variables
			self(dc, dv)
			# Compute the change by the inf. norm
			change=np.linalg.norm(self.x.reshape(nelx*nely,1)-self.xold.reshape(nelx*nely,1),np.inf)
			# Termination Check
			if change < 0.01:	break 

	def _oc(self, nelx,nely,x,volfrac,dc,dv,g):
		l1=0
		l2=1e9
		move=0.2
		# reshape to perform vector operations
		xnew=np.zeros(nelx*nely)

		while (l2-l1)/(l1+l2)>1e-3:
			lmid=0.5*(l2+l1)
			xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
			gt=g+np.mean((dv*(xnew-x)))
			if gt>0 :
				l1=lmid
			else:
				l2=lmid
		return (xnew,gt)

class NGNet:
	"""NGNet : height vectors -> density vectors"""

	def __init__(self, n_basis_x, n_basis_y, nelx, nely, scale):
		"""NGNet Constructor

		Parameters
		----------
		n_basis_x, n_basis_y : int
			number of bases on each coordinate, n_basis_x * n_basis_y bases in total
		nelx, nely : int
			number of elements in each coordinate
		scale : float
			scale parameter of the Gaussian kernel
		"""
		dx = 1.0 / n_basis_x
		dy = 1.0 / n_basis_y
		position = np.mgrid[0.5*dx:1.0:dx, 0.5*dy:1.0:dy]
		self.xpos = position[0].reshape((1, -1))
		self.ypos = position[1].reshape((1, -1))
		self.xrad = scale * dx
		self.yrad = scale * dy
		dx = 1.0 / nelx
		dy = 1.0 / nely
		xx = np.mgrid[0.5*dx:1.0:dx, 0.5*dy:1.0:dy]
		xmesh = xx[0].reshape((-1, 1))
		ymesh = xx[1].reshape((-1, 1))
		xdist = xmesh - self.xpos
		ydist = ymesh - self.ypos
		self.kernel = np.exp(-0.5 * (xdist**2 / self.xrad**2 + ydist**2 / self.yrad**2))

	def __call__(self, height):
		"""NGNet : height vectors -> density vectors
		
		Parameters
		----------
		height : 1d array
			height parameter for each kernel, [-1, 1]

		Return
		------
		density (1d array) :  density of each element, 0 or 1
		"""
		gray = np.dot(self.kernel, height) / np.sum(self.kernel, axis=1)
		binary = np.asarray(gray > 0., dtype=float)
		return binary

  
if __name__ == "__main__":
	CASE = 1
	# Default input parameters
	nelx=180
	nely=60
	volfrac=0.4
	rmin=5.4
	penal=3.0
	ft=1 # ft==0 -> sens, ft==1 -> dens

	import sys
	if len(sys.argv)>1: nelx   =int(sys.argv[1])
	if len(sys.argv)>2: nely   =int(sys.argv[2])
	if len(sys.argv)>3: volfrac=float(sys.argv[3])
	if len(sys.argv)>4: rmin   =float(sys.argv[4])
	if len(sys.argv)>5: penal  =float(sys.argv[5])
	if len(sys.argv)>6: ft     =int(sys.argv[6])

	if CASE == 1:
		co = ComplianceMinimization(nelx, nely, volfrac, penal)
		simp = SIMP(nelx, nely, volfrac, rmin, ft)
		simp.solve(co)
	elif CASE == 2:
		ngnet = NGNet(18*3, 6*3, nelx, nely, 1.)
		x = ngnet(0.5 + np.random.randn(18*6*3*3))
		co.plot(x, 'ngnet.pdf')
	else:
		raise ValueError("CASE <= 2")