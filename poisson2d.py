import numpy as np
import sympy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg
from sympy import Mul
x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(ue, x, 2)+sp.diff(ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xij, self.yij ...
        self.N = N
        x = np.linspace(0, self.L, self.N+1)
        y = np.linspace(0, self.L, self.N+1)
        xij, yij = np.meshgrid(x, y, indexing='ij')
        self.xij = xij  
        self.yij = yij
        return self.xij, self.yij

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1,self.N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        self.D = D
        return self.D
        
    def laplace(self):
        """Return vectorized Laplace operator"""
        self.h = self.L/self.N
        self.D2()
        D2x = (1./self.h**2)*self.D
        D2y = (1./self.h**2)*self.D
        self.laplace_op = (sparse.kron(D2x, sparse.eye(self.N+1)) +  sparse.kron(sparse.eye(self.N+1), D2y))
        return self.laplace_op
       

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.N+1, self.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        self.bnds = np.where(B.ravel() == 1)[0]
        return self.bnds

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        # return A, b
        self.laplace()
        self.get_boundary_indices()
        A = self.laplace_op
        self.f = self.ue.diff(x, 2) + self.ue.diff(y, 2)
        F = sp.lambdify((x, y), self.f)(self.xij, self.yij)
        self.b = F.ravel()
        A = A.tolil()
        for j in self.bnds:
            A[j] = 0
            A[j, j] = 1
            index_y = int((j- (j%(self.N+1)))/(self.N+1))
            index_x = j%(self.N+1)
            self.b[j] = sp.exp(sp.cos(4*sp.pi*self.yij[index_y][-index_y-1])*sp.sin(2*sp.pi*self.xij[index_x][index_x])) #Boundary conditions
        self.A = A.tocsr()
        return self.A, self.b

    def l2_error(self, u):
        """Return l2-error norm"""
        h = self.L/self.N
        self.error = h*np.linalg.norm(((u - sp.lambdify((x, y), self.ue)(self.xij, self.yij))))
        return self.error

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.N = N
        self.create_mesh(self.N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((self.N+1, self.N+1))
        return self.U

    def convergence_rates(self, m=6):
        E = []
        h = []
        N0 = 8
        dx = self.L/N0
        for m in range(m):
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(dx)
            dx = dx/2
            N0 *= 2
            
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        
        return r, np.array(E), np.array(h)
    def Lagrangebasis(self, xj, x=x):
        n = len(xj)
        ell = []
        numert = Mul(*[x - xj[i] for i in range(n)])

        for i in range(n):
            numer = numert/(x - xj[i])
            denom = Mul(*[(xj[i] - xj[j]) for j in range(n) if i != j])
            ell.append(numer/denom)
        return ell
    def Lagrangefunction2D(self,u, basisx, basisy):
        N, M = u.shape
        f = 0
        for i in range(N):
            for j in range(M):
                f += basisx[i]*basisy[j]*u[i, j]
        return f
    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        self.create_mesh(self.N)
        #Find index "i = index_counter_x" s.t. xij[i,0] <= x <= xij[i+1,0]
        ##Find index "j = index_counter_y" s.t. yij[0,j] <= y  <= yij[0,j+1]
        index_counter_x = 0
        for K in range(self.N):
            if self.xij[index_counter_x,0] <= x <= self.xij[index_counter_x+1,0]:
                break
            else: 
                index_counter_x += 1
        index_counter_y = 0
        for L in range(self.N):
            if self.yij[0,index_counter_y] <= y <= self.yij[0,index_counter_y+1]:
                break
            else: 
                index_counter_y += 1
        x_val = x 
        y_val = y
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        #Lagrange interpolation'
        lx = self.Lagrangebasis(self.xij[index_counter_x:index_counter_x+2, 0], x=x)
        ly = self.Lagrangebasis(self.yij[0, index_counter_y:index_counter_y+2], x=y)
        f = self.Lagrangefunction2D(self.U[index_counter_x:index_counter_x+2, index_counter_y:index_counter_y+2], lx, ly)
        val = f.subs({x: x_val, y: y_val})
        return val

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2
test_convergence_poisson2d()
def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h, y: 1-sol.h/2}).n()) < 1e-3
test_interpolation()
