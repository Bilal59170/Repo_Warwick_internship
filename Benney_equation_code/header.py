### Sort of header. Regroups all the common variables and functions that are used in several .py files in the same time.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML



#### System VARIABLES: Physics & Mathematics (Values From Oscar's code)
L_x = 30    # Dimensionless;  (horizontal length)/h_N;    epsilon=1/L_x;
nu =  2*np.pi/L_x
T = 200   # Dimensionless: end time of the simulation
theta = np.pi/3 #Slope angle: in rad
print("Critical upper Reynolds Number:", 5/4*np.cos(theta)/np.sin(theta))

#Set the physical and dimensionless parameters and deduce h_n and U_n (cf 08/01)
mu_l = 1.0016e-3   #Water dynamical viscosity at 20°C (cf https://wiki.anton-paar.com/en/water/)
rho_l = 1000   #Water volumic mass
gamma = 71.97e-3 # surface tension of water at 20°C (cf https://srd.nist.gov/jpcrdreprint/1.555688.pdf)
g = 9.81        #gravity acceleration

Ca = 0.01 #Supposed O(epsilon^2)
Re =  5 #Supposed O(1)
h_N = (Re/Ca)*(mu_l**2/(rho_l*gamma))
U_N = rho_l*g*(h_N**2)*np.sin(theta)/(2*mu_l)  #Speed of the Nusselt solution

epsilon = h_N/L_x
delta = 1e-3

###Some Usefull fonctions

#Function for setting the time and space steps
def set_steps_and_domain(_N_x, _CFL_factor, _N_t=None, T=T):
    """
    Computes dt from dx (or the other way around if _N_t isn't None) using CFL conditions with
    U_N velocity. I take the same step if the CFL conditions give me a less precise step 
    (i.e dt= _dt = min(_dx, _dx/U_N/_CFL_factor)

    Input: 
    - The name are explicit and _CFL_factor is the factor in the CFL conditions (cf wiki)

    Output:(_N_x, _N_t, _dx, _dt, domain_x, domain_t) 
"""

    if _N_x is not None: #Space, then time
        _dx = L_x/_N_x #not dx = L_x/(N_x-1): x-periodic so we don't count the last point so Lx-dx = (Nx-1)dx
        _dt = min(_dx*T/L_x , _dx/U_N/_CFL_factor) #CFL conditions
        _N_t = 2*int(T/_dt+1) #bcs (N_t-1)dt = T

        
    elif _N_t is not None: #Time, then space
        _dt = T/(_N_t-1)
        _dx = min(_dt*L_x/T, _dt*U_N*_CFL_factor) #My own CFL conditions 
        _N_x = int(L_x/_dx) #as it's periodic. no +1 
    
    #Time & space domains
    domain_x = np.linspace(0, L_x, _N_x, endpoint=False) #Periodic domain: don't need the last point (endpoint=False)
    domain_t = np.linspace(0, T, _N_t, endpoint=True) #Not periodic: we need last point

    return _N_x, _N_t, _dx, _dt, domain_x, domain_t

#Function for initial conditions
def sincos(x, _h_mean, _ampl_c, _ampl_s, _freq_c, _freq_s):
    '''Function to compute sinusoidal periodic initial condition'''
    return _h_mean + _ampl_c*np.cos(_freq_c*x) + _ampl_s*np.sin(_freq_s*x) 

## Animation Functions 
def round_fct(r, nb_decimal):
    '''Detect the power of 10 and round the number nb_decimal further. 
    Coded to have titles of animation not to big.
    Expl: round_fct(0.000123456, 4) = 0.000123 (or 0.0001234 I don't remember)'''
    if r==0:
        return 0
    else:
        # print("NUMBER R", r)
        power_10 = int(np.log10(abs(r)))
        factor = 10**(power_10)
        return round(r/factor, nb_decimal)*factor

def func_anim(_time_series, _anim_space_array, _anim_time_array, 
              title, title_x_axis = None, title_y_axis= None, _legend_list = None):
 
    #(Nb_tab, N_t, N_x) tab
    Nb_time_series =  _time_series.shape[0]
    gap = (_time_series.max()- _time_series.min())/10

    #subplot initialisation
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))

    # Initialise the plot ligns
    # print("Nb_times_series", Nb_time_series)
    array_line_analytical = Nb_time_series*[0]
    for k in range(Nb_time_series):
        if not(_legend_list is None):
            array_line_analytical[k], = axs.plot([], [], label=_legend_list[k])
        else:
            array_line_analytical[k], = axs.plot([], [])

    axs.set_xlim([_anim_space_array.min(), _anim_space_array.max()])
    axs.set_ylim([_time_series.min()-gap, _time_series.max()+gap])

    # if bool_grid:
    #     if _x_ticks_major is None:
    #         axs.set_xticks(_anim_space_array, minor=True)
    #         axs.grid(which='both')  # Major ticks for x-axis, every 1 unit
    #     else:
    #         axs.set_xticks(_x_ticks_major) #Major xticks : the one which is showed on the x axis
    #         axs.set_xticks(_anim_space_array, minor=True) #minor xticks
    #         axs.grid(which='both') #grid for the minor tick

    # Update the function in the animation
    def update(frame):
        t_1 = _anim_time_array[frame]
        y = np.array([_time_series[k][frame] for k in range(Nb_time_series)])
        for k in range(_time_series.shape[0]):
            
            array_line_analytical[k].set_data(_anim_space_array, y[k])
            
        axs.set_title(title + ' at t= {}'.format(round_fct(t_1, 5)))
        axs.set_xlabel(title_x_axis)
        axs.set_ylabel(title_y_axis)
        if not(_legend_list is None):
            axs.legend()
        
        return array_line_analytical,

    # Create the animation
    return FuncAnimation(fig, update, frames=len(_anim_time_array)-1)

