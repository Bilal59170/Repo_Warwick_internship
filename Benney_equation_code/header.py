### Header file. Regroups all the global variables and functions that are used in several .py files in the same time.
print("\n\n****  header.py: Beginning of the print  *** \n")


###Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML



#### System VARIABLES: Physics & Mathematics (Values From Oscar's code)
L_x = 30    # Dimensionless;  (horizontal length)/h_N;    epsilon=1/L_x;
nu =  2*np.pi/L_x #Not the nu from the Ks equation
T = 300   # Dimensionless: end time of the simulation
theta = np.pi/3 #Slope angle: in rad
print("Time T: ", T)
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

epsilon = 1/L_x
delta = 1e-3



###Linear stability : Re_0 and k_0
Re_0 = 5/4/np.tan(theta)
k_0_sq = Ca*8/5*(Re-Re_0)/(nu**2) #/(nu**2) as k_0**2 not k_0

if k_0_sq < 0:
    print("Linear Stability:  no Critical wave number, k_0**2 <0")
else:
    k_0 = np.sqrt(k_0_sq)
    print("Linear Stability: Critical wave number k_0:", k_0)


# plt.rc('font', size=11) # Set the font size for all the plots 



###Some configurations and Usefull fonctions

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
        _N_t = int(T/_dt+1) #bcs (N_t-1)dt = T

        
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

#Function to round 
def round_fct(r, nb_decimal):
    '''Detect the power of 10 and round the number nb_decimal further. 
    Coded to have titles of animation not to big.
    Expl: round_fct(0.000123456, 4) = 0.000123 '''

    if r ==0:
        return 0
    elif r is None:
        return None

    r_str = format(abs(r), ".20f") #avoid to have a scientific writing of abs(r)


    bool_point = False #If the point in the string is passed
    bool_zero = True #if there is only 0
    pos_log10 = False # sign of log10(abs(r))
    idx_point, idx_last_zero = -1, -1

    for i in range(len(r_str)): # loop to 
        if r_str[i] == "." and not(bool_point):
            bool_point = True
            idx_point = i
        elif r_str[i] != "0" and not(bool_point):
            bool_zero = False
            pos_log10 = True
        elif r_str[i]!= '0' and bool_zero and bool_point:
            bool_zero = False
            if i-1 == idx_point:
                idx_last_zero = i-2
            else:
                idx_last_zero = i-1

    # print("idx_point", idx_point)
    # print("idx_last_zero", idx_last_zero)
        
    if pos_log10:
        if nb_decimal == 1:
            correct_str = r_str[:idx_point]
        else:
            correct_str = r_str[:idx_point+nb_decimal]
    else:
        correct_str = r_str[: idx_last_zero+nb_decimal+1]

    if r>0:
        result = float(correct_str)
    else:
        result = -float(correct_str)
    
    return result


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

def print_neat_matrix(matrix):
    """
    Prints a NumPy array (or matrix) neatly with aligned columns.

    Parameters:
        matrix (numpy.ndarray): The NumPy array to be printed.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy.ndarray")

    # Determine the maximum width of any element for alignment
    max_width = max(len(f"{item:.4g}") for item in matrix.flatten())

    # Generate the format string for alignment
    format_str = f"{{:>{max_width}.4g}}"

    # Print each row of the matrix
    for row in matrix:
        formatted_row = " ".join(format_str.format(item) for item in row)
        print(formatted_row, "\n")
