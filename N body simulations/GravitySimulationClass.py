import numpy,IPython,time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class GravitySimulation:
    "A class used for N-body simulations"

    def __init__(self, G=1.0,Softening=0.05, RandomSeed = 643,OutputPrefix = '512/'):
        "We here set G and the random seed from which the ICs are generated. OutputPrefix sets the prefix of output plots."
        self.G = G
        self.Softening = Softening
        self.RandomSeed = RandomSeed
        self.OutputPrefix = OutputPrefix
        numpy.random.seed(RandomSeed)
        self.AccelerationTime = []


    def InitializeHernquistHalo(self, TotalMass = 1.0, ScaleRadius = 1.0, Nparticles = 512):
        "We generate initial conditions with a density profile following a Hernquist profile. Initial velocities are 0-10 per cent of escape velocity (this is quite arbitrary here)."
        rho0 = TotalMass/2/numpy.pi/ScaleRadius**3

        def Rho(r,rho0, ScaleRadius):
            return rho0/(r/ScaleRadius) * 1.0/(1.0 + r/ScaleRadius)**3

        def Mass(r,TotalMass, ScaleRadius):
            return TotalMass * (r/ScaleRadius)**2 / (1.0+r/ScaleRadius)**2

        #The radius of each particle is determined by first sampling uniformly distributed numbers between 0 and Mtotal, and based on this r is calculated
        M_particles = numpy.random.random(Nparticles)*TotalMass
        r_particles = ScaleRadius/(numpy.sqrt(TotalMass/M_particles)-1.0)

        #set theta
        Theta_particles = numpy.arccos( 2.0 *  numpy.random.random(Nparticles) - 1.0)
        Phi_particles = numpy.random.uniform(0.0,numpy.pi*2,Nparticles)
        #set x,y,z
        x = r_particles * numpy.sin(Theta_particles) * numpy.cos(Phi_particles)
        y = r_particles * numpy.sin(Theta_particles) * numpy.sin(Phi_particles)
        z = r_particles * numpy.cos(Theta_particles)

        #set velocities:
        EscapeVel_particles = numpy.sqrt(2*self.G*M_particles/r_particles)
        v_particles = numpy.random.uniform(0.0,0.1,Nparticles) * EscapeVel_particles
        #v_particles = numpy.sqrt(self.G*M_particles/r_particles)
        Theta_particles = numpy.arccos( 2.0 *  numpy.random.random(Nparticles) - 1.0)
        Phi_particles = numpy.random.uniform(0.0,numpy.pi*2,Nparticles)

        vx = v_particles * numpy.sin(Theta_particles) * numpy.cos(Phi_particles)
        vy = v_particles * numpy.sin(Theta_particles) * numpy.sin(Phi_particles)
        vz = v_particles * numpy.cos(Theta_particles)

        #save stuff in class
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz

        self.m = r_particles*0.0+TotalMass/Nparticles

        self.CalcAcceleration()#this determines self.V, the potential.


    def DoPlot(self,N=0):
        "Create a plot. N is number contained in filename."
        plt.figure(N,figsize=(14,6.0))
        plt.subplots_adjust(left=0.06,right=0.95, top=0.95,wspace=0.18, hspace=0.2,bottom=0.09)

        plt.subplot(1,2,1)
        plt.plot(self.x,self.y,'o',ms=2)
        plt.xlim((-10,10))
        plt.ylim((-10,10))

        ax = plt.subplot(1,2,2)
        plt.scatter(self.x,self.y,c=self.V,s=2,vmin=-1.6,vmax=-0.1)
        #colorbar = plt.colorbar()
        #colorbar.set_label('Potential', rotation=270)

        plt.axis('equal')
        plt.xlim((-10.0,10.0))
        plt.ylim((-10.0,10.0))

        cbaxes = inset_axes(ax, width="50%", height="5%", loc=9)
        color_bar = plt.colorbar(cax=cbaxes,orientation='horizontal')
        color_bar.set_label( r'$V$', color='black', fontsize=15)
        color_bar.ax.tick_params(labelsize=12)

        plt.title("t = %.2f" %(t), loc= "center")

        plt.savefig('%s_%.3d.png'%(self.OutputPrefix,N))
        plt.clf()
        plt.close()


    def CalcAcceleration(self):
        "We calculate the acceleration of all particles. We save the time it took to calculate the accelerations to the list AccelerationTime."
        t_start = time.time()
        Nparticles = self.x.size
        dx = numpy.repeat(self.x,Nparticles) - numpy.tile(self.x,Nparticles)
        dy = numpy.repeat(self.y,Nparticles) - numpy.tile(self.y,Nparticles)
        dz = numpy.repeat(self.z,Nparticles) - numpy.tile(self.z,Nparticles)
        dx.shape = (Nparticles,Nparticles)
        dy.shape = (Nparticles,Nparticles)
        dz.shape = (Nparticles,Nparticles)

        r2_ij = dx**2+dy**2+dz**2+self.Softening**2
        r_ij = numpy.sqrt(r2_ij)
        m2_ij = numpy.outer(self.m,self.m)

        F_ij = self.G*m2_ij / r2_ij / r_ij

        self.ax = numpy.sum(F_ij * dx,axis=0) / self.m
        self.ay = numpy.sum(F_ij * dy,axis=0) / self.m
        self.az = numpy.sum(F_ij * dz,axis=0) / self.m

        self.V = - numpy.sum( self.G*m2_ij / r_ij ,axis=0) / self.m

        t_end = time.time()
        self.AccelerationTime.append(t_end-t_start)


    def RunSimulation(self,dt = 0.01,tmax = 2.0,TimebetweenPlots = 0.5):
        "Main loop of the simulation. dt is timestep, tmax is end time, we generate an output plot with self.DoPlot at TimebetweenPlots intervals."
        IntegerTimestepsBetweenPlots = int(TimebetweenPlots/dt)

        #Leapfrog offset
        self.CalcAcceleration()
        self.vx += self.ax*dt/2.0
        self.vy += self.ay*dt/2.0
        self.vz += self.az*dt/2.0

        #main loop
        global t
        t = 0
        timeslist = []
        IntegerTimestep = 0
        while t<tmax:
            time0 = time.time()
            #if IntegerTimestep % IntegerTimestepsBetweenPlots == 0:
                #self.DoPlot(IntegerTimestep/IntegerTimestepsBetweenPlots)
                #print( 'time=',t,', tmax =',tmax )

            self.x += self.vx*dt
            self.y += self.vy*dt
            self.z += self.vz*dt

            self.CalcAcceleration()

            self.vx += self.ax*dt
            self.vy += self.ay*dt
            self.vz += self.az*dt

            t += dt
            IntegerTimestep += 1

            timeslist.append(time.time()- time0)

        print(numpy.mean(timeslist))
        meantimes.append(numpy.mean(timeslist))



obj1 = GravitySimulation()

N = [16, 32, 64, 128, 256, 512, 1024, 2048]
meantimes = []
for i in N:   
    obj1.InitializeHernquistHalo(Nparticles= i)
    obj1.RunSimulation()

plt.plot(N[1:],meantimes[1:])
plt.xlabel("Number of particles")
plt.ylabel("Mean time it took for one calculation")
plt.loglog()
plt.savefig('N_vs_meantimes.png')
plt.show()	



