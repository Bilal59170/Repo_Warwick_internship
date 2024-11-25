#CODE FOR THE BENNEY EQUATION


###IMPORT
##Imports
import numpy as np
import scipy.optimize 
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from sklearn.linear_model import LinearRegression

import solver_BDF 




######## SYSTEM SETTINGS ######

##VARIABLES: Physics & Mathematics (Values From Oscar's code)
h_N =  0.000018989145744046399526063052252081 #Nusselt solution from Oscar's code. NEED VERIFICATION
L_x = 30    # Dimensionless;  (horizontal length)/h_N;    epsilon=1/L_x;
epsilon = 1/L_x
L_y = 10    # Dimensionless: vertical length/h_N
T = 35     # Dimensionless: end time of the simulation
theta = 1.047197551  #Slope angle: in rad

mu_l = 1.0e-3   #fluid viscosity
rho_l = 10000   #fluid volumic mass
gamma = 0.00015705930063439097934322589390558 #
g = 9.81        #gravity acceleration
U_N = rho_l*g*(h_N**2)*np.sin(theta)/(2*mu_l)     #Speed of the Nusselt solution
Ca = mu_l*U_N/gamma #O(epsilon^2)
Re = rho_l*U_N*h_N/mu_l  #O(1)

print("\n\n#### BEGINING OF THE PRINT ###")
print("\n Nusselt velocity: ", U_N )
print("Value of Re (check if O(1)):", Re)
print("Value of Ca and Ca/(eps**2) (Check if O(eps**2)):", Ca, Ca/(epsilon**2))


##VARIABLEs: steps & domain
#Steps: 2 ways
#Space, then time
N_x = 128 #To choose little at first and then, increase 
dx = L_x/N_x #and not  dx = L_x/(N_x-1) as it periodic along x-axis so we don't count the last point so Lx-dx = (Nx-1)dx
dx_2, dx_3, dx_4 = dx**2, dx**3, dx**4
CFL_factor = 100 #take it big 
dt = dx/U_N/CFL_factor #CFL conditions
N_t = int(T/dt+1) #bcs (N_t-1)dt = T
print("Nb of (space, time) points: ", (N_x, N_t) )

#Time, then space
# N_t = 256 #To choose little at first and then, increase 
# dt = T/(N_t-1)
# CFL_factor = 100 #take it big 
# dx = dt*U_N*CFL_factor #My own CFL conditions 
# N_x = int(L_x/dx) #as it's periodic. no +1 
# dx_2, dx_3, dx_4 = dx**2, dx**3, dx**4

print("Nb of (space, time) points: ", (N_x, N_t) )

#Time & space domains
domain_x = np.linspace(0, L_x, N_x, endpoint=False) #Periodic domain: we don't need the last point (endpoint=False)
domain_t = np.linspace(0, T, N_t, endpoint=True) #Not periodic: we need last point
#test the boundaries of the domain (0 and L_x -dx if well initialised)
print("First and last point of the domain:", domain_x[0], domain_x[-1]) 
print("Lx-dx:", L_x-dx)


### Initial conditions 
#Somes functions to use for Initial Condition
def sincos(x, _h_mean, _ampl_c, _ampl_s, _freq_c, _freq_s):
    return _h_mean + _ampl_c*np.cos((2*np.pi/L_x)*_freq_c*x) + _ampl_s*np.sin((2*np.pi/L_x)*_freq_s*x) #Initial condition. (sinus, periodic on D)

#initial condition
# h_mat = np.zeros((N_t, N_x)) #Matrix of the normalised height. Each line is for a given time from 0 to (N_t-1)*dt
h_mean = 1
ampl_c, ampl_s, freq_c, freq_s = 0.02, 0.01, 3, 1  #frequencies need to be a relative to have periodicity
Initial_Conditions = sincos(domain_x, h_mean, ampl_c, ampl_s, freq_c, freq_s)

if True:#Plot of initial condition
    plt.plot(domain_x, Initial_Conditions)
    plt.title("Initial conditions for the normalized height h ")
    plt.show()




######## SOLVING #######

###### Finite Difference & BDF Scheme ######
### Solving
order_BDF_scheme = 2
if True:
    #computation times :  ((N_x, N_t), t computation): [(128, 229),14s), ((256, 458), 60s), ((512, 915),T= 710s)]
    h_mat = solver_BDF.solver_Benney_BDF_FD(N_x=N_x, N_t= N_t, dx=dx, dt=dt, IC=Initial_Conditions,
                                             theta=theta, order_BDF_scheme=order_BDF_scheme, Ca=Ca, Re=Re)

    ##Saving the solution
    bool_save = False
    if bool_save: 
        np.savetxt('Benney_equation_code\\Benney_numerical_solution_Nx_{N_x}.txt'.format(N_x=N_x), h_mat)


##Loading the solution
bool_load_solution = False
if bool_load_solution:
    h_mat= np.loadtxt('Benney_equation_code\\Benney_numerical_solution_Nx_128.txt')
    assert ((h_mat.shape[0]==N_t)and(h_mat.shape[1]==N_x)), "Solution loading: Problem of shape "



### VISUALISATION 
##animation function

if True:#Animation of benney numerical solution
    animation_Benney = solver_BDF.func_anim(_time_series=np.array([h_mat]), _anim_space_array = domain_x,
                                        _anim_time_array = domain_t,
                                        title="Benney height for (N_x, N_t, L_x, T, Re, Ca) =({N_x}, {N_t}, {L_x}, {T}, {Re}, {Ca})".format(N_x=N_x, N_t=N_t, L_x=L_x, T=T, Re=solver_BDF.round_fct(Re,3), Ca=solver_BDF.round_fct(Ca, 3)), 
                                        title_x_axis=r"x axis: horizontal inclined by $\theta$",
                                        title_y_axis= r"y-axis (inclined by $\theta$)",
                                        _legend_list = ["height h(x,t)"])
    plt.show()

    if False:
        animation_Benney.save('Benney_equation_code\\animation_Benney_Nx_{N_x}.mp4'.format(N_x=N_x)) #needs the program ffmpeg installed and in the PATH

N_t_begin = 128
space_steps_list = 2**np.arange(7,10)
print((space_steps_list))






### VERIFICATION OF THE METHOD
if True: #Difference in loglog graph
    ##Make loglog graph of difference

    #Load the arrays
    print("List of the tested space steps:", space_steps_list)
    arr_solutions = []
    for space_step in space_steps_list:
        print(space_step)
        arr_solutions.append(np.loadtxt('Benney_equation_code\\Benney_numerical_solution_Nx_{element}.txt'.format(element=space_step)))
    
    #Compute the differences 
    arr_L2_diff, arr_Linf_diff = np.zeros(len(space_steps_list)-1), np.zeros(len(space_steps_list)-1)
    for i in range(len(space_steps_list)-1):
        arr_L2_diff[i] = np.linalg.norm(arr_solutions[i+1][-1,0::2]-arr_solutions[i][-1])
        arr_Linf_diff[i] = np.max(np.absolute((arr_solutions[i+1][-1,0::2]-arr_solutions[i][-1])))


    # #Linear regression of the differences
    # N_1, N_2 = 0, N_t-1
    # x_lin_reg_array = domain_t[N_1:N_2].reshape(-1,1) #Necessary reshape for sklearn

    # #L2 Case
    # y_lin_reg_array = np.log(diff_L2[N_1:N_2]).reshape(-1, 1)
    # reg = LinearRegression(fit_intercept=True).fit(x_lin_reg_array, y_lin_reg_array) #sklearn function
    # Reg_lin_coef_a_L2 , Reg_lin_coef_b_L2= reg.coef_[0][0], reg.intercept_[0]
    # print("Slope coefficient(s) ax+b, Determination coefficient R²:", Reg_lin_coef_a_L2, Reg_lin_coef_b_L2, reg.score(x_lin_reg_array, y_lin_reg_array))

    # #Linf case
    # y_lin_reg_array = np.log(diff_Linf[N_1:N_2]).reshape(-1, 1)
    # reg = LinearRegression(fit_intercept=True).fit(x_lin_reg_array, y_lin_reg_array) #sklearn function
    # Reg_lin_coef_a_Linf , Reg_lin_coef_b_Linf= reg.coef_[0][0], reg.intercept_[0]
    # print("Slope coefficient(s) ax+b, Determination coefficient R²:", Reg_lin_coef_a_Linf, Reg_lin_coef_b_Linf, reg.score(x_lin_reg_array, y_lin_reg_array))


    ##Plot
    fig, axs = plt.subplots(1,2, figsize=(10, 5))
    axs[0].plot(space_steps_list[:-1], arr_L2_diff, label=r"$||h(t+dt)-h(t)||_{L^2}$")
    # axs[0].plot(domain_t[1:], np.exp(Reg_lin_coef_b_L2)*(domain_t[1:]**Reg_lin_coef_a_L2), label="Linear Regression")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel("time t (begins at dt)")
    axs[0].legend()
    axs[0].set_title(r"diff in $L^2$ norm of h for increasing dx")

    axs[1].plot(space_steps_list[:-1], arr_Linf_diff, label=r"$||h(t+dt)-h(t)||_{L^{\infty}}$")
    # axs[1].plot(domain_t[1:], Reg_lin_coef_b_Linf*(domain_t[1:]**Reg_lin_coef_a_Linf), label="Linear Regression")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel("time t (begins at dt)")
    axs[1].legend()
    axs[1].set_title(r"Diff in $L^{\infty}$ norm of h for increasing dx")

    plt.show()

    


###### SPECTRAL METHOD #########

