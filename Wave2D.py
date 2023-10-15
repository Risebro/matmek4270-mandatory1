import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self,N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xji, self.yij = ...
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xij, self.yij ...
        L = 1
        x = np.linspace(0, L, N+1)
        y = np.linspace(0, L, N+1)
        xij, yij = np.meshgrid(x, y, indexing='ij')
        self.xij = xij  
        self.yij = yij
        return self.xij, self.yij
    def D2(self,N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1,N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        self.D = D
        return self.D
    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c*np.pi*np.linalg.norm(np.array([self.mx, self.my]))

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.dx = 1/N
        self.create_mesh(N)
        self.D2(N)
        Un, Unm1 = np.zeros((2, N+1, N+1))
        Unm1[:] = sp.lambdify((x, y,t), self.ue(mx,my))(self.xij, self.yij,0) #initial condition
        D1 = self.D/(self.dx**2)
        Un[:] = Unm1[:] + 0.5*(self.c*self.dt)**2*(D1 @ Unm1 + Unm1 @ D1.T)
        self.Un = Un
        self.Unm1 = Unm1
        return self.Un, self.Unm1
    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.dx /self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        u_exact_n = sp.lambdify((x, y, t), self.ue(self.mx, self.my))(self.xij, self.yij, t0)
        # Compute the L2 norm of the error
        error = u_exact_n - u
        h = self.dx
        error_norm =  h* np.linalg.norm(error)   
        self.error_norm = error_norm
        return self.error_norm
    def apply_bcs(self):
        self.Unp1[0] = 0 #top 
        self.Unp1[-1] = 0 #bottom
        self.Unp1[:, -1] = 0 #right
        self.Unp1[:, 0] = 0 #left
        return None
    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.c = c
        self.mx = mx
        self.my = my 
        self.cfl = cfl
        self.initialize(N,mx,my)
        self.D2(N)
        D1 = self.D/(self.dx**2)
        Unp1 = np.zeros((N+1, N+1))
        self.Unp1 = Unp1
        data = {0: self.Unm1.copy()}
        for n in range(1, Nt):
            self.Unp1[:] = 2*self.Un - self.Unm1 + (self.c*self.dt)**2*(D1 @ self.Un + self.Un @ D1.T)
            # Set boundary conditions
            self.apply_bcs()
            # Swap solutions
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
            data[n] = self.Un.copy()
        if store_data > 0: 
            return data
        elif store_data == -1:
            return (self.dx, self.l2_error(self.Un, (n+1)*self.dt))
    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err= self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)
class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1,N+1), 'lil')
        D[0, :2] = -2, 2
        D[-1, -2:] = 2,-2
        self.D = D
        return self.D
    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        return None
    def plot_animation(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3):
        # Initialize the solution
        #Not optimal, but it works.
        self.c = c
        self.mx = mx
        self.my = my
        self.cfl = cfl
        self.initialize(N, mx, my)
        self.D2(N)
        D1 = self.D / (self.dx ** 2)
        Unp1 = np.zeros((N + 1, N + 1))
        self.Unp1 = Unp1

        # Create a figure and axis for the animation
        fig, ax = plt.subplots()
        im = ax.imshow(self.Unm1, cmap=cm.coolwarm, extent=(0, 1, 0, 1))
        plt.colorbar(im)

        def animate(n):
            self.Unp1[:] = 2 * self.Un - self.Unm1 + (self.c * self.dt) ** 2 * (D1 @ self.Un + self.Un @ D1.T)
            self.apply_bcs()
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
            im.set_array(self.Un)
            ax.set_title(f'Time step {n}/{Nt}')
            return im,

        ani = FuncAnimation(fig, animate, frames=Nt, repeat=False)
        plt.show()
def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2
test_convergence_wave2d()

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05
test_convergence_wave2d_neumann()

def test_exact_wave2d():
    m = 3;Nt= 10; C= 1/np.sqrt(2); N = 888
    sol = Wave2D()
    solN = Wave2D_Neumann()
    err_D = sol(N,Nt,cfl=C, c=1, mx = m, my=m,store_data=-1)[1]
    err_N = solN(N,Nt,cfl=C, c=1, mx = m, my=m,store_data=-1)[1]
    assert abs(err_D) < 1e-12 and abs(err_N) < 1e-12
test_exact_wave2d()

# Create an instance of the Wave2D_Neumann class
#solN = Wave2D_Neumann()

# Call the plot_animation method to generate and display the animation
#N = 64  # Adjust the grid size as needed
#Nt = 100  # Adjust the number of time steps as needed
#solN.plot_animation(N, Nt, cfl=0.5, c=1.0, mx=3, my=3)