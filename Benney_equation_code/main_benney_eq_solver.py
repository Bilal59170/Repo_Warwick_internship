#CODE FOR THE BENNEY EQUATION


###IMPORT
import numpy as np
import scipy.optimize 
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from sklearn.linear_model import LinearRegression

import solver_BDF 

print("\n\n#### BEGINING OF THE PRINT ###")


######## SYSTEM SETTINGS ######

##VARIABLES: Physics & Mathematics (Values From Oscar's code)
h_N =  0.000018989145744046399526063052252081 #Nusselt solution from Oscar's code. 
L_x = 30    # Dimensionless;  (horizontal length)/h_N;    epsilon=1/L_x;
epsilon = 1/L_x
L_y = 10    # Dimensionless: vertical length/h_N
T = 200   # Dimensionless: end time of the simulation
theta = 1.047197551  #Slope angle: in rad
print("Critical upper Reynolds Number:", 5/4*np.cos(theta)/np.sin(theta))

mu_l = 1.0e-3   #fluid viscosity
rho_l = 1000   #fluid volumic mass
gamma = 0.00015705930063439097934322589390558 #
g = 9.81        #gravity acceleration
U_N = rho_l*g*(h_N**2)*np.sin(theta)/(2*mu_l)     #Speed of the Nusselt solution
Ca = mu_l*U_N/gamma #O(epsilon^2)
Re = rho_l*U_N*h_N/mu_l  #O(1)


print("\n Nusselt velocity: ", U_N )
print("Value of Re (check if O(1)):", Re)
print("Value of Ca and Ca/(eps**2) (Check if O(eps**2)):", Ca, Ca/(epsilon**2))




##VARIABLEs: steps & domain
#Steps: 2 ways

def set_steps_and_domain(_N_x, _CFL_factor, _N_t=None, T=T):
    """
    Input: The name are explicit
    Output:(_N_x, _N_t, _dx, _dt, domain_x, domain_t) 
    2 Things possible
    - If _N_x provided (usefull to control _N_x): (N_t, dx, dt (by using some CFL Conditions))
    - If _N_t provided: (N_x, dx, dt (by using some CFL Conditions))"""

    if _N_x is not None: #Space, then time
        _dx = L_x/_N_x #not dx = L_x/(N_x-1): x-periodic so we don't count the last point so Lx-dx = (Nx-1)dx
        _dt = _dx/U_N/_CFL_factor #CFL conditions
        _N_t = int(T/_dt+1) #bcs (N_t-1)dt = T

        
    elif _N_t is not None: #Time, then space
        _dt = T/(_N_t-1)
        _dx = _dt*U_N*_CFL_factor #My own CFL conditions 
        _N_x = int(L_x/_dx) #as it's periodic. no +1 
    
    #Time & space domains
    domain_x = np.linspace(0, L_x, _N_x, endpoint=False) #Periodic domain: don't need the last point (endpoint=False)
    domain_t = np.linspace(0, T, _N_t, endpoint=True) #Not periodic: we need last point

    return _N_x, _N_t, _dx, _dt, domain_x, domain_t

N_x = 128 #To choose little at first and then, increase 
CFL_factor = 150 #take it big 
N_x, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(_N_x=N_x, _CFL_factor = CFL_factor)
dx_2, dx_3, dx_4 = dx**2, dx**3, dx**4
print("Nb of (space, time) points: ", (N_x, N_t))

#test the boundaries of the domain (0 and L_x -dx if well initialised)
print("First and last point of the domain:", domain_x[0], domain_x[-1]) 
print("Lx-dx:", L_x-dx)


### Initial conditions 
def sincos(x, _h_mean, _ampl_c, _ampl_s, _freq_c, _freq_s):
    '''Function to compute sinusoidal periodic initial condition'''
    return _h_mean + _ampl_c*np.cos(_freq_c*x) + _ampl_s*np.sin(_freq_s*x) 

h_mean = 1
ampl_c, ampl_s, freq_c, freq_s = 0.001, 0.003, 3, 1  #frequencies need to be a relative to have periodicity
Initial_Conditions = sincos(
    domain_x, h_mean, ampl_c, ampl_s, (2*np.pi/L_x)*freq_c, (2*np.pi/L_x)*freq_s)

if False:#Plot of initial condition
    plt.plot(domain_x, Initial_Conditions)
    plt.title("IC (t=0) for the normalized height h (Ac, As, fc, fs)"+
    "=({Ac} {As}, {fc}, {fs})".format(Ac=ampl_c, As=ampl_s, fc=freq_c, fs=freq_s))
    plt.show()










######## SOLVING #######

order_BDF_scheme = 2
space_steps_array = np.array([128])
print("Space steps: ", space_steps_array)
#IC
h_mean, ampl_c, ampl_s, freq_c, freq_s = 1, 0, 0.5, 0, 1 
#Gaussian external pressure parameters and external normal pressure function
A_Ns, mu_Ns, sigma_Ns = -50, L_x/2, 0.01 
N_s_function = lambda x:solver_BDF.N_s_derivatives_gaussian(
    x, A_Ns=A_Ns, mu_Ns=mu_Ns, sigma_Ns=sigma_Ns, L=L_x)


#Boolean variables to control what action to do with FD method
bool_solve_FD, bool_save_FD, bool_load_FD = False, False, False
bool_anim_FD, bool_save_anim_FD = False, False

#Boolean variables to control what action to do with Spectral method
bool_solve_Spectral, bool_save_spectral, bool_load_spectral = True, False, False
bool_anim_spectral, bool_save_anim_spectral = True, False





###### Finite Difference & BDF Scheme ######

### Solving & animation
title_file = 'Benney_equation_code\\FD_method_BDF_order{BDF_order}_Nx_{N_x}.txt'.format(
                BDF_order=order_BDF_scheme, N_x=N_x)
title_anim = 'Benney_equation_code\\FD_method_animation_BDF_order{order_BDF}_Nx_{N_x}_A_Ns{A_Ns}.mp4'.format(
        order_BDF = order_BDF_scheme, N_x=N_x, A_Ns=A_Ns)

for i in range(len(space_steps_array)):
    N_x, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(
        _N_x=space_steps_array[i], _CFL_factor = CFL_factor)
    dx_2, dx_3, dx_4 = dx**2, dx**3, dx**4
    Initial_Conditions = sincos(
        domain_x, h_mean, ampl_c, ampl_s, (2*np.pi/L_x)*freq_c, (2*np.pi/L_x)*freq_s)

    if bool_solve_FD:
        #computation times : ((N_x, N_t), t computation): 
        # [(128, 229),14s), ((256, 458), 60s), ((512, 915),T= 710s)]
        h_mat_FD = solver_BDF.solver_Benney_BDF_FD(
            N_x=N_x, N_t= N_t, dx=dx, dt=dt, IC=Initial_Conditions,
            theta=theta, order_BDF_scheme=order_BDF_scheme, Ca=Ca, Re=Re, N_s_function=N_s_function,
            nb_percent=1)

        ##Saving the solution
        if bool_save_FD: 
            np.savetxt(title_file, h_mat_FD)


##Loading the solution
if bool_load_FD:
    h_mat_FD= np.loadtxt(title_file)
    assert ((h_mat_FD.shape[0]==N_t)and(h_mat_FD.shape[1]==N_x)), "Solution loading: Problem of shape"


### VISUALISATION 
##animation function

if bool_anim_FD:#Animation of benney numerical solution
    animation_Benney = solver_BDF.func_anim(_time_series=np.array([h_mat_FD]), 
        _anim_space_array = domain_x, _anim_time_array = domain_t,
        title="Benney height for (N_x, N_t, L_x, T, Re, Ca) =({N_x}, {N_t}, {L_x}, {T}, {Re}, {Ca})".format(
            N_x=N_x, N_t=N_t, L_x=L_x, T=T, Re=solver_BDF.round_fct(Re,3), Ca=solver_BDF.round_fct(Ca, 3)), 

        title_x_axis=r"x axis: horizontal inclined by $\theta$",
        title_y_axis= r"y-axis (inclined by $\theta$)",
        _legend_list = ["height h(x,t) with FD method and BDF order {}".format(order_BDF_scheme)])
    # plt.show()

if bool_save_anim_FD:
    animation_Benney.save(title_anim)  #needs the program ffmpeg installed and in the PATH


### VERIFICATION OF THE METHOD

#Load the arrays

def plot_difference_graph(list_h_mat, general_subplot_title, title_left_graph=None, 
                          title_right_graph=None, save_plot = False, 
                          file_name= None, regression_lin=False): #Difference in loglog graph
    '''
    Output: Make a loglog (in log10 scale) graph of the difference of the same method
        with different number of space points.The abscis is chosen to be the space bigger 
        step (i.e dx) between the two method compared. 

    Example: The comparison between (128, 256) would be at dx = L_x/128   
    Input: 
        - list_h_mat: list of the numerical solutions computed with INCREASING number of steps. 
        Usually [128, 256, 512, 1024]'''
    

    #Compute the differences : with increasing number of points
    arr_L2_diff, arr_Linf_diff = np.zeros(len(space_steps_array)-1), np.zeros(len(space_steps_array)-1)
    #remove some part of the array so that they can be substracted: 128=2**7, 1024=2**10 
    for i in range(len(space_steps_array)-1):
        arr_L2_diff[i] = np.linalg.norm(list_h_mat[-1][-1,0::(2**(3-i))]-list_h_mat[i][-1])
        arr_Linf_diff[i] = np.max(np.absolute((list_h_mat[-1][-1,0::(2**(3-i))]-list_h_mat[i][-1])))

    ###Plot
    fig, axs = plt.subplots(1,2, figsize=(15, 5))
    dx_array = L_x/space_steps_array[:-1]


    #Make a linear regression
    if regression_lin:
        #Linear regression of the differences
        x_lin_reg_array = np.log10(dx_array).reshape(-1,1) #Necessary reshape for sklearn

        #L2 Case
        y_lin_reg_array = np.log10(arr_L2_diff).reshape(-1, 1)
        reg = LinearRegression(fit_intercept=True).fit(x_lin_reg_array, y_lin_reg_array) #sklearn function
        #slope a, intercept b and  Determination coefficient R²
        Reg_lin_coef_a_L2 , Reg_lin_coef_b_L2= reg.coef_[0][0], reg.intercept_[0]
        Reg_lin_coef_r2_L2 = reg.score(x_lin_reg_array, y_lin_reg_array)

        #Linf case
        y_lin_reg_array = np.log10(arr_Linf_diff).reshape(-1, 1)
        reg = LinearRegression(fit_intercept=True).fit(x_lin_reg_array, y_lin_reg_array) #sklearn function
        #slope a, intercept b and  Determination coefficient R²
        Reg_lin_coef_a_Linf , Reg_lin_coef_b_Linf= reg.coef_[0][0], reg.intercept_[0]
        Reg_lin_coef_r2_Linf = reg.score(x_lin_reg_array, y_lin_reg_array) 




    axs[0].scatter(dx_array, arr_L2_diff, label="Scheme differences")
    rescaled_lin_reg_L2 = 10**(Reg_lin_coef_b_L2)*(dx_array**Reg_lin_coef_a_L2)
    axs[0].plot(dx_array,rescaled_lin_reg_L2, 
                label=r"Linear Regression, (a, b, R²) $\approx$ ({a}, {b}, {R2})".format(
                a=solver_BDF.round_fct(Reg_lin_coef_a_L2, 2), b=solver_BDF.round_fct(Reg_lin_coef_b_L2, 2)
                , R2=solver_BDF.round_fct(Reg_lin_coef_r2_L2, 4)),
                color='r')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    # Adding annotations
    text_steps = ['\"1024-128\"', '\"1024-256\"', '\"1024-512\"' ] #increasing in the number of points
    annotations = [{'text': text_steps[i], 'xy': (dx_array[i], arr_L2_diff[i])} for i in range(len(dx_array))]
    #annotate on a pixel from a certain pixel distance from the xy point
    for annotation in annotations:
        axs[0].annotate(
            text=annotation['text'],
            xy=annotation['xy'],
            xytext=(0, 5),
            textcoords= "offset pixels")
    axs[0].set_xlabel("dx, the biggest step between 2 numerical solutions")
    axs[0].set_title(r"diff in $L^2$ norm in loglog graph")
    axs[0].legend(loc='upper left', fontsize="small")
 
    axs[1].scatter(dx_array, arr_Linf_diff, label="Scheme differences")
    rescaled_lin_reg_Linf = 10**(Reg_lin_coef_b_Linf)*(dx_array**Reg_lin_coef_a_Linf)
    axs[1].plot(dx_array,rescaled_lin_reg_Linf, 
            label=r"Linear Regression, (a, b, R²) $\approx$ ({a}, {b}, {R2})".format(
            a=solver_BDF.round_fct(Reg_lin_coef_a_Linf, 2), b=solver_BDF.round_fct(Reg_lin_coef_b_Linf, 2)
            , R2=solver_BDF.round_fct(Reg_lin_coef_r2_Linf, 4)),
            color='r')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    annotations = [{'text': text_steps[i], 'xy': (dx_array[i], arr_Linf_diff[i])}
                    for i in range(len(text_steps))]
    for annotation in annotations:
        axs[1].annotate(
            annotation['text'], xy=annotation['xy'], xytext=(0, 5), 
            textcoords = "offset pixels")
    axs[1].set_xlabel("dx, the biggest step between 2 numerical solutions")
    axs[1].set_title(r"Diff in $L^{\infty}$ norm in loglog graph")
    axs[1].legend(loc='upper left', fontsize="small")

    fig.suptitle(general_subplot_title)

    if save_plot:
        assert not(file_name is None), "fct \"plot_difference_graph\": Problem in the save_plot"
        plt.savefig(file_name)
    plt.show()

if False:
    list_h_mat_FD = [] 
    for space_step in space_steps_array:
        list_h_mat_FD.append(
            np.loadtxt(
                'Benney_equation_code\\Saved_numerical_solutions\\FD_method_BDF_order{BDF_order}_Nx_{N_x}.txt'
                .format(BDF_order=order_BDF_scheme, N_x=space_step)))
        
if False: #Plot the difference Graph
    print("List of the tested space steps:", space_steps_array)

    #list of the FD & BDF numerical solution for different space steps
  
    
    #plot the difference graph
    plot_difference_graph(
        list_h_mat_FD, 
        general_subplot_title=
        "FD + order {} BDF Scheme: Difference of 2 computed h from 2 different dx at the final time T."
        .format(order_BDF_scheme),
        save_plot=True,
        regression_lin=True,
        file_name="Benney_equation_code\\BDF_order_{}_FD_difference_graph".format(order_BDF_scheme))

if False: #FD method Animation & Graph: Fixed step,  Compare the different BDF Orders
    #Plot
    N_x_plot = 128
    N_x, N_t, _, _, domain_x, domain_t = set_steps_and_domain(_N_x=N_x_plot, _CFL_factor=CFL_factor)
    list_result_BDF_FD_N_x, order_BDF_list = [], [1, 2, 3,6]
    index_eval_time = int(N_t)-1

    for i in range( len(order_BDF_list)):
        
        list_result_BDF_FD_N_x.append(
            np.loadtxt(
            'Benney_equation_code\\Saved_numerical_solutions'
            '\\FD_method_BDF_order{BDF_order}_Nx_{N_x}.txt'.format(
            BDF_order=order_BDF_list[i], N_x=N_x))
        )

        plt.plot(domain_x, list_result_BDF_FD_N_x[i][index_eval_time,:], 
                 label="BDF {}".format(order_BDF_list[i]))
        

    plt.xlabel("inclined plane"), plt.ylabel("height h(x, t)")
    plt.legend()
    plt.title("FD with BDF 1, 2, 3 at final time with N_x = {}".format(N_x_plot))
    plt.savefig("Benney_equation_code\\FD_BDF_comparison_order_1236_Nx_{}.png".format(N_x_plot))
    plt.show()

    # Animation


    animation_BDF_FD = solver_BDF.func_anim(
        _time_series=np.array(list_result_BDF_FD_N_x), _anim_space_array = domain_x,
        _anim_time_array = domain_t,
        title="Benney height with FD method and BDF for (N_x, CFL, Re, Ca)"
          +"=({N_x}, {CFL}, {Re}, {Ca})".format(
        N_x=1024, CFL=30, Re=solver_BDF.round_fct(Re,3), Ca=solver_BDF.round_fct(Ca, 3)), 

        title_x_axis=r"x axis: horizontal inclined by $\theta$",
        title_y_axis= r"y-axis (inclined by $\theta$)",
        _legend_list = ["BDF order {}".format(BDF_order) for BDF_order in order_BDF_list])
    # plt.show()

    animation_BDF_FD.save(
        'Benney_equation_code\\FD_BDF_order_comparison_order_1236_Nx_{}.mp4'.format(N_x_plot))  





###### SPECTRAL METHOD #########

##Test on DFT: to have the good normalization constant
#Test of the differentiation with DFT 
if False:
    fq_tab = N_x*np.fft.rfftfreq(N_x) #to justify better after
    print("\n Shape of the array of frequencies: ", fq_tab.shape)


    h_test = sincos(domain_x, 0, 0, _ampl_s=1, _freq_c=0, _freq_s=1)
    plt.plot(domain_x, h_test)
    plt.show()

    h_x = np.fft.irfft( (1j *fq_tab)*np.fft.rfft(h_test))
    print(h_x.real)
    plt.plot(domain_x, h_x.real)
    plt.title("Derivative of h_0 (sin -> cos) \n with Fourier (Gibbs phenomenon ?)")
    plt.show()

    h_xx = np.fft.irfft( (1j *fq_tab)**2*np.fft.rfft(h_test))
    print(h_xx)
    plt.plot(domain_x, h_xx.real)
    plt.title("2nd Derivative of u_0 (sin -> -sin) \n with Fourier (Gibbs phenomenon ?)")
    plt.show()

    h_xxxx = np.fft.irfft( (1j *fq_tab)**4*np.fft.rfft(h_test))
    print(h_xxxx)
    plt.plot(domain_x, h_xxxx.real)
    plt.title("4th Derivative of u_0 (sin -> sin) \n with Fourier (Gibbs phenomenon ?)")
    plt.show()
    plt.plot(domain_x, np.absolute(h_xxxx-np.sin(domain_x*2*np.pi/L_x)))
    plt.title("some difference")
    plt.show()


### Solving
title_file = 'Benney_equation_code\\Spectral_method_BDF_order{BDF_order}_Nx_{N_x}_for_mass M.txt'.format(
                    BDF_order=order_BDF_scheme, N_x=N_x)
title_anim = ('Benney_equation_code\\Spectral_method_animation_Ns{A_Ns}_{sigma}'+
              '_BDF_order{BDF_order}_Nx_{N_x}_for_mass_M.mp4'.format(
                    BDF_order=order_BDF_scheme, N_x=N_x, A_Ns=A_Ns, sigma= sigma_Ns))

for i in range(len(space_steps_array)):
    N_x, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(
        _N_x=space_steps_array[i], _CFL_factor = CFL_factor)
    dx_2, dx_3, dx_4 = dx**2, dx**3, dx**4
    Initial_Conditions = sincos(domain_x, h_mean, ampl_c, ampl_s, (2*np.pi/L_x)*freq_c, (2*np.pi/L_x)*freq_s)
    
    
    if bool_solve_Spectral:
        h_mat_spectral = solver_BDF.solver_Benney_BDF_Spectral(
            N_x=N_x, N_t= N_t, dx=dx, dt=dt, IC=Initial_Conditions,
            theta=theta, order_BDF_scheme=order_BDF_scheme, Ca=Ca, Re=Re,
            N_s_function=N_s_function)

        ##Saving the solution
        if bool_solve_Spectral and bool_save_spectral: 
            np.savetxt(title_file, h_mat_spectral)

 
    if bool_load_spectral:
        h_mat_spectral= np.loadtxt(title_file)
        assert ((h_mat_spectral.shape[0]==N_t)
                and(h_mat_spectral.shape[1]==N_x)),"Solution loading: Problem of shape "


    ###Animation
    if bool_anim_spectral:#Animation of benney numerical solution
        animation_Benney = solver_BDF.func_anim(
            _time_series=np.array([h_mat_spectral]), _anim_space_array = domain_x,
            _anim_time_array = domain_t,
            title= "Benney height for (N_x, N_t, L_x, T, Re, Ca) =({N_x}, {N_t}, {L_x}, {T}, {Re}, {Ca})".format(
            N_x=N_x, N_t=N_t, L_x=L_x, T=T, Re=solver_BDF.round_fct(Re,3), Ca=solver_BDF.round_fct(Ca, 3)), 

            title_x_axis=r"x axis: horizontal inclined by $\theta$",
            title_y_axis= r"y-axis (inclined by $\theta$)",
            _legend_list = ["h(x,t) with spectral method & BDF order {}".format(order_BDF_scheme)])
        plt.show()

    if bool_anim_spectral and bool_save_anim_spectral:
        animation_Benney.save(title_anim) #needs the program ffmpeg installed and in the PATH




###### Verification of Spectral Method

#Load the arrays
print("List of the tested space steps:", space_steps_array)
list_h_mat_spectral = [] #list of the Spectral & BDF numerical solution for different space steps
for space_step in space_steps_array:
    list_h_mat_spectral.append(
        np.loadtxt(
        'Benney_equation_code\\Saved_numerical_solutions\\Spectral_method_BDF_order{BDF_order}_Nx_{N_x}.txt'.format(
        BDF_order=order_BDF_scheme, N_x=space_step)))


if False: #Animation & Graph Spectral method: Fixed BDF order, compare the different steps
    plot_difference_graph(
        list_h_mat_spectral, 
        general_subplot_title=
        "Spectral + order {} BDF Scheme: Difference of 2 computed h from 2 different dx at the final time T."
        .format(order_BDF_scheme),
        save_plot=True,
        regression_lin=True,
        file_name="Benney_equation_code\\BDF_order_{}_Spectral_difference_graph".format(order_BDF_scheme))


if False: #Spectral method Animation & Graph: Fixed step,  Compare the different BDF Orders
    #Plot
    N_x_plot = 1024
    N_x, N_t, _, _, domain_x, domain_t = set_steps_and_domain(_N_x=N_x_plot, _CFL_factor=CFL_factor)
    list_result_BDF_spectral_N_x, order_BDF_list = [], [1, 2, 3,6]
    index_eval_time = int(N_t)-1

    for i in range( len(order_BDF_list)):
        
        list_result_BDF_spectral_N_x.append(
            np.loadtxt(
            'Benney_equation_code\\Saved_numerical_solutions'
            '\\Spectral_method_BDF_order{BDF_order}_Nx_{N_x}.txt'.format(
            BDF_order=order_BDF_list[i], N_x=N_x))
        )

        plt.plot(domain_x, list_result_BDF_spectral_N_x[i][index_eval_time,:], 
                 label="BDF {}".format(order_BDF_list[i]))
        

    plt.xlabel("inclined plane"), plt.ylabel("height h(x, t)")
    plt.legend()
    plt.title("Spectral with BDF 1, 2, 3 at final time with N_x = {}".format(N_x_plot))
    plt.savefig("Benney_equation_code\\Spectral_BDF_comparison_order_1236_Nx_{}.png".format(N_x_plot))
    plt.show()

    Animation


    animation_BDF_Spectral = solver_BDF.func_anim(
        _time_series=np.array(list_result_BDF_spectral_N_x), _anim_space_array = domain_x,
        _anim_time_array = domain_t,
        title="Benney height with Spectral method and BDF for (N_x, CFL, Re, Ca)"
          +"=({N_x}, {CFL}, {Re}, {Ca})".format(
        N_x=1024, CFL=30, Re=solver_BDF.round_fct(Re,3), Ca=solver_BDF.round_fct(Ca, 3)), 

        title_x_axis=r"x axis: horizontal inclined by $\theta$",
        title_y_axis= r"y-axis (inclined by $\theta$)",
        _legend_list = ["BDF order {}".format(BDF_order) for BDF_order in order_BDF_list])
    # plt.show()

    animation_BDF_Spectral.save(
        'Benney_equation_code\\Spectral_BDF_order_comparison_order_1236_Nx_{}.mp4'.format(N_x_plot))  


if False: #Spectral method Animation & Graph: Fixed step, and order 1: compare 2 methods
    #Plot
    N_x, N_t, _, _, domain_x, domain_t = set_steps_and_domain(_N_x=1024, _CFL_factor=CFL_factor)
    list_result_BDF_1_spectral_N_x_1024, order_BDF_list = [], [1, 2, 3, 6]
    index_eval_time = int(N_t/2)

    list_result_BDF_1_spectral_N_x_1024.append(
            np.loadtxt(
            'Benney_equation_code\\Spectral_method_BDF_order{BDF_order}_Nx_{N_x}.txt'.format(
            BDF_order=order_BDF_list[0], N_x=1024))
        ) 
    list_result_BDF_1_spectral_N_x_1024.append(
            np.loadtxt(
            'Benney_equation_code\\Spectral_method_BDF_order{BDF_order}_Nx_{N_x}_other_method.txt'.format(
            BDF_order=order_BDF_list[0], N_x=1024))
        )

      #Animation


    animation_BDF_Spectral = solver_BDF.func_anim(
        _time_series=np.array(list_result_BDF_1_spectral_N_x_1024), _anim_space_array = domain_x,
        _anim_time_array = domain_t,
        title="Benney height with Spectral method and BDF for (N_x, CFL, Re, Ca) =({N_x}, {CFL}, {Re}, {Ca})".format(
        N_x=1024, CFL=30, Re=solver_BDF.round_fct(Re,3), Ca=solver_BDF.round_fct(Ca, 3)), 

        title_x_axis=r"x axis: horizontal inclined by $\theta$",
        title_y_axis= r"y-axis (inclined by $\theta$)",
        _legend_list = ["original method", "my weird method"])
    # plt.show()

    animation_BDF_Spectral.save(
        'Benney_equation_code\\Spectral_BDForder_1_comparison_nrm_with_other.mp4')  









###### Comparison of the FD & Spectral methods  #######
bool_anim = False
if bool_anim:#Animation of benney numerical solution

    N_x_plot = 128
    _ , N_t, _, _, domain_x, domain_t = set_steps_and_domain(_N_x=N_x_plot, _CFL_factor=CFL_factor)

    h_mat_FD = np.loadtxt(
            'Benney_equation_code\\Saved_numerical_solutions'
            + '\\FD_method_BDF_order{BDF_order}_Nx_{N_x}.txt'.format(
            BDF_order=order_BDF_scheme, N_x=N_x_plot))
    h_mat_spectral = np.loadtxt(
            'Benney_equation_code\\Saved_numerical_solutions'
            +'\\Spectral_method_BDF_order{BDF_order}_Nx_{N_x}.txt'.format(
            BDF_order=order_BDF_scheme, N_x=N_x_plot))
    
    animation_FD_Spectral = solver_BDF.func_anim(
        _time_series=np.array([h_mat_spectral-h_mat_FD]), _anim_space_array = domain_x,
        _anim_time_array = domain_t,
        title="Spectral-FD BDF order {}".format(order_BDF_scheme) +
             "and (N_x, N_t, L_x, T, Re, Ca) =({N_x}, {N_t}, {L_x}, {T}, {Re}, {Ca})".format(
            N_x=N_x_plot, N_t=N_t, L_x=L_x, T=T, Re=solver_BDF.round_fct(Re,3), 
            Ca=solver_BDF.round_fct(Ca, 3)),

        title_x_axis=r"x axis: horizontal inclined by $\theta$",
        title_y_axis= r"y-axis (inclined by $\theta$)",
        _legend_list = ["h_spectral-h_FD"])
    # plt.show()
    animation_FD_Spectral.save(
        'Benney_equation_code\\Spectral-FD_anim_BDF_order{order_BDF}_1236_Nx_{N_x}.mp4'.format(
            order_BDF=order_BDF_scheme, N_x=N_x_plot))


##Plot of the Difference between Spectral and FD Method with BDF
bool_linear_reg = False
if False: 
    arr_L2_diff, arr_Linf_diff = np.zeros(len(space_steps_array)), np.zeros(len(space_steps_array))
    for i in range(len(space_steps_array)):
        arr_L2_diff[i] = np.linalg.norm(list_h_mat_FD[i][-1]-list_h_mat_spectral[i][-1])
        arr_Linf_diff[i] = np.max(np.absolute(list_h_mat_FD[i][-1]-list_h_mat_spectral[i][-1]))



    ##plot
    fig, axs = plt.subplots(1,2, figsize=(15, 5))
    dx_array = L_x/space_steps_array

    #Make a linear regression
    if bool_linear_reg:
        #Linear regression of the differences
        x_lin_reg_array = np.log10(dx_array).reshape(-1,1) #Necessary reshape for sklearn

        #L2 Case
        y_lin_reg_array = np.log10(arr_L2_diff).reshape(-1, 1)
        reg = LinearRegression(fit_intercept=True).fit(x_lin_reg_array, y_lin_reg_array) #sklearn function
        #slope a, intercept b and  Determination coefficient R²
        Reg_lin_coef_a_L2 , Reg_lin_coef_b_L2= reg.coef_[0][0], reg.intercept_[0]
        Reg_lin_coef_r2_L2 = reg.score(x_lin_reg_array, y_lin_reg_array)

        #Linf case
        y_lin_reg_array = np.log10(arr_Linf_diff).reshape(-1, 1)
        reg = LinearRegression(fit_intercept=True).fit(x_lin_reg_array, y_lin_reg_array) #sklearn function
        #slope a, intercept b and  Determination coefficient R²
        Reg_lin_coef_a_Linf , Reg_lin_coef_b_Linf= reg.coef_[0][0], reg.intercept_[0]
        Reg_lin_coef_r2_Linf = reg.score(x_lin_reg_array, y_lin_reg_array) 

    axs[0].scatter(dx_array, arr_L2_diff,label="Difference between Spectral & FD")
    rescaled_lin_reg_L2 = 10**(Reg_lin_coef_b_L2)*(dx_array**Reg_lin_coef_a_L2)
    axs[0].plot(dx_array,rescaled_lin_reg_L2, 
            label=r"Linear Regression, (a, b, R²) $\approx$ ({a}, {b}, {R2})".format(
            a=solver_BDF.round_fct(Reg_lin_coef_a_L2, 2), b=solver_BDF.round_fct(Reg_lin_coef_b_L2, 2)
            , R2=solver_BDF.round_fct(Reg_lin_coef_r2_L2, 4)),
            color='r')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    # Adding annotations
    text_steps = [ 'N_x = 128', 'N_x = 256', 'N_x = 512', 'N_x = 1024']
    annotations = [{'text': text_steps[i], 'xy': (dx_array[i], arr_L2_diff[i])} for i in range(len(text_steps))]
    for annotation in annotations:
        axs[0].annotate(
            annotation['text'],
            xy=annotation['xy'],
            xytext=(0, 5),
            textcoords= "offset pixels")
    axs[0].set_xlabel("dx")
    axs[0].set_title(r"diff in $L^2$ norm at the final time T")
    axs[0].legend(fontsize="small")

    axs[1].scatter(dx_array, arr_Linf_diff, label="Difference between FD & Spectral")
    rescaled_lin_reg_Linf = 10**(Reg_lin_coef_b_Linf)*(dx_array**Reg_lin_coef_a_Linf)
    axs[1].plot(dx_array,rescaled_lin_reg_Linf, 
            label=r"Linear Regression, (a, b, R²) $\approx$ ({a}, {b}, {R2})".format(
            a=solver_BDF.round_fct(Reg_lin_coef_a_Linf, 2), b=solver_BDF.round_fct(Reg_lin_coef_b_Linf, 2)
            , R2=solver_BDF.round_fct(Reg_lin_coef_r2_Linf, 4)),
            color='r')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    annotations = [{'text': text_steps[i], 'xy': (dx_array[i], arr_Linf_diff[i])} for i in range(len(text_steps))]
    for annotation in annotations:
        axs[1].annotate(
            annotation['text'],
            xy=annotation['xy'],
            xytext=(0, 5),
            textcoords= "offset pixels")

    axs[1].set_xlabel("dx")
    axs[1].set_title(r"diff in $L^{\infty}$ norm at the final time T")
    axs[1].legend(fontsize="small")

    fig.suptitle("Comparison between the Spectral and FD methods with BDF of order {}".format(order_BDF_scheme))
    plt.savefig("Benney_equation_code\\Plot_Comparison_Spectral_FD_BDF_order_{}.png".format(order_BDF_scheme))
    plt.show()




##### All sort of different Verifications #####
##### Linear Analysis
bool_linear_analysis = False

print("\n LINEAR ANALYSIS")
if False:
    nu = 2*np.pi/L_x
    def coef_dispersion(k):
        '''Cf Oscar's paper for the formula. He does it on a 2pi-per function whereas I'm doing it
        on a L-per function. So I scale by 2pi/L as derive in L-per case <=> multiply
         by k2pi/L '''
        k_L = k*nu
        return (8*Re/15-(2/3)*np.cos(theta)/np.sin(theta)-k_L**2/(3*Ca))*k_L**2 + (-2*k_L)*(1j)

    k_0_squared = (Ca*(8/5*Re - 2*np.cos(theta)/np.sin(theta)))/(nu**2) #scaling in the L-per dommain
    print("k_0**2 is:", k_0_squared)
    print("bool: ", k_0_squared <=0)
    if k_0_squared <=0:
        print("bite")
        absciss = np.linspace(k_0_squared-1, 0, 100, endpoint=True)
        plt.plot(absciss, coef_dispersion(absciss).real)
        plt.title("k_0**2 <0")
        plt.axvline(x=k_0_squared, color='b')
        plt.axhline(y=0, color='r', linestyle='-') #draws a line at x=0 to see more clearly the sign of the points
        plt.show()
    else:
        k_0 = np.sqrt(k_0_squared)
        print("k_0", k_0)
        absciss = np.linspace(0, k_0+3, 100, endpoint=True)
        plt.plot(absciss, coef_dispersion(absciss).real)
        plt.axhline(y=0, color='r', linestyle='-') #draws a line at x=0 to see more clearly the sign of the points
        plt.axvline(x=k_0, color='b')
        (y_min_plot, _) = plt.ylim()
        plt.annotate(
            text=r"$k_0 = +\sqrt{Ca\left(\frac{8}{5}Re -2cot(\theta)) \right)}/\nu\approx$ "
            +str(solver_BDF.round_fct(k_0, 3)),

            xy=(k_0,y_min_plot),
            xycoords= "data",
            xytext= (0.1, 0.1),
            textcoords="axes fraction",
            arrowprops=dict(facecolor='black', width=1, shrink=0.05)
                    )
        
        plt.title("Real part of the dispersion coefficient")
        plt.show()


if bool_linear_analysis:
    N_x, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(
    _N_x=512, _CFL_factor = CFL_factor, T=3)
    print("Number of time points:", N_t)
    #IC and order of the Fourier decomposition that we are interested in
    order_fourier_mode = 5
    IC_amplitude = 0.001
    Initial_Conditions = sincos(
        domain_x, _h_mean=1, _ampl_c=IC_amplitude, _ampl_s=0, _freq_c=(2*np.pi/L_x)*order_fourier_mode, _freq_s=0 )
    

    #loading or computing solutions
    title_simulation = 'Benney_equation_code\\Linear_verif_Sp_method_BDF_order{BDF_order}_Nx_{N_x}.txt'.format(
                     BDF_order=order_BDF_scheme, N_x=N_x)
    if False: ##Loading the solution. Be carefull to take the relevant IC
        h_mat_lin= np.loadtxt(title_simulation) 
        assert ((h_mat_lin.shape[0]==N_t)and(h_mat_lin.shape[1]==N_x)), "Solution loading: Problem of shape "
    if True: #Computing
        h_mat_lin = solver_BDF.solver_Benney_BDF_Spectral(
            N_x=N_x, N_t= N_t, dx=dx, dt=dt, IC=Initial_Conditions,
            theta=theta, order_BDF_scheme=order_BDF_scheme, Ca=Ca, Re=Re, N_s_function=N_s_function)
        if True:
            np.savetxt(title_simulation, h_mat_lin)

    #Fourier matrix 
    h_Fourier_mat = np.fft.rfft(h_mat_lin, norm="forward", axis=1)
    print("Shape Fourier mat", h_Fourier_mat.shape)
    print("Shape h matrix", h_mat_lin.shape)
    assert ((h_Fourier_mat.shape[0]==h_mat_lin.shape[0])), "Shape of the Fourier matrix: Problem"
    # H_c = 2*h_Fourier_mat.real  
    # H_s = -2*h_Fourier_mat.imag 

    if True: #VIsualisation of the behavior (amplify, reduces)
        animation_Benney = solver_BDF.func_anim(
            _time_series=np.array([h_mat_lin]),
                _anim_space_array = domain_x, _anim_time_array = domain_t,
            title="Benney height forIC= {A}cos(v{k}) (N_x, N_t, L_x, T, Re, Ca) =({N_x}, {N_t}, {L_x}, {T}, {Re}, {Ca})".
            format(
                N_x=N_x, N_t=N_t, L_x=L_x, T=T, Re=solver_BDF.round_fct(Re,3), Ca=solver_BDF.round_fct(Ca, 3),
                k=order_fourier_mode, A=IC_amplitude), 

            title_x_axis=r"x axis: horizontal inclined by $\theta$",
            title_y_axis= r"y-axis (inclined by $\theta$)",
            _legend_list = ["height with Spectral method"]
            )
        # plt.close()
        plt.show()

        if True:
            animation_Benney.save(
                'Benney_equation_code\\Linear_theory_FD_method_animation_BDF_order{order_BDF}_Nx_{N_x}_k_{k}.mp4'.
                format(
                order_BDF = order_BDF_scheme, N_x=N_x, k=order_fourier_mode))  
    
    #Theoretical Fourier coefficient in Linear theory
    lambda_k = coef_dispersion(order_fourier_mode)
    print("lambda k:", lambda_k)
    ck_0 = IC_amplitude/2
    ck = ck_0*np.exp(lambda_k*domain_t) #cos: /2; sin: /2j
    print("Experimental ck_0: ", h_Fourier_mat[0, order_fourier_mode])
    print("Lin theory ck_0: ", ck_0)
    if True: #Plot of the FOurier coef
        plt.plot(domain_t, h_Fourier_mat[:, order_fourier_mode].real, label="real part")
        plt.plot(domain_t, h_Fourier_mat[:, order_fourier_mode].imag, label="imaginary part")
        plt.plot(domain_t, ck.real, label="Linear th real")
        plt.plot(domain_t, ck.imag, label="Linear th imag")
        plt.xlabel("time t")
        plt.title("{k}th fourier mode".format(k=order_fourier_mode))
        plt.legend()
        plt.savefig('Benney_equation_code\\Linear_theory_Sp_method_graph_BDF_order{order_BDF}_Nx_{N_x}_k_{k}.pdf'.
                    format(order_BDF=order_BDF_scheme, N_x=N_x, k=order_fourier_mode))
        plt.show()
        

    if False: #Plot of h_max
        plt.plot(domain_t, np.max(np.absolute(h_mat_lin), axis=1))
        plt.xlabel('time t')
        # plt.yscale('log')
        # plt.xscale('log')
        plt.title("max_x |h(x, t)|")
        plt.show()


    if True: #measurement of Oscilation and dampening rate for the k>k_0 monochromatic case
        ##find the main frequency. We get rid of the 0-th harmonic component: the mean value
        fft_Fourier_coef = np.array( [np.fft.fft(h_Fourier_mat[:, order_fourier_mode].real),
                            np.fft.fft(h_Fourier_mat[:, order_fourier_mode].imag) ] )
        freq_time =np.fft.fftfreq(len(fft_Fourier_coef[0])) #same for imag part
        #Find the peak in the coefficient
        # print(fft_Fourier_coef[0])
        freq_max = np.array([freq_time[np.argmax(np.abs(fft_Fourier_coef[0]))],
                              freq_time[ np.argmax(np.abs(fft_Fourier_coef[1])) ]]
                              )
        
        print("ARGMAX:", np.argmax(np.abs(fft_Fourier_coef[0])))
        freq_exp, freq_theory = np.abs(freq_max)/dt, np.abs(lambda_k.imag/(2*np.pi))
        error_oscilation = max(np.abs(freq_exp-freq_theory)/freq_theory)
        print("freq for Re(c_k) and Img(c_k) and linear th freq:", freq_exp, freq_theory )
        print("max relative error:", error_oscilation*100, "%")
        # array of the fq errors in % compared to the linear theory fq. We take a fq from 3
        #to 10
        array_err_oscilation = np.array([0.2235807950298624, 25.41170401280628])

        ##Find dampening rate
        #The exponential is 'hidden' by the cos/sin so I just take the max points of the absolute value
        #and I measure its decay. If not enough point, maybe do a exponential regression on these points.
        tab = [abs(h_Fourier_mat[:, order_fourier_mode].real), 
               abs(h_Fourier_mat[:, order_fourier_mode].imag)]
        list_time_maxima, list_maxima = [[], []], [[], []]
        find_rate = [False, False]
        idx_decay_time = -1*np.ones(2)

        #Reglin
        array_LinReg_a, array_LinReg_b, array_LinReg_r2 = np.zeros(2), np.zeros(2), np.zeros(2)
        for i in range(2):
            for n_t in range(0, N_t-1):#Don't consider the extreme parts
                if tab[i][n_t]>tab[i][n_t-1] and tab[i][n_t] > tab[i][n_t+1]:
                    list_time_maxima[i].append(dt*n_t)
                    list_maxima[i].append(tab[i][n_t])


            #Linear regression of the differences
            x_lin_reg_array = np.array(list_time_maxima[i]).reshape(-1,1) #Necessary reshape for sklearn

            #L2 Case
            y_lin_reg_array = np.log(list_maxima[i]).reshape(-1, 1)
            reg = LinearRegression(fit_intercept=True).fit(x_lin_reg_array, y_lin_reg_array) #sklearn function
            #slope a, intercept b and  Determination coefficient R²
            array_LinReg_a[i] , array_LinReg_b[i]= reg.coef_[0][0], reg.intercept_[0]
            array_LinReg_r2[i] = reg.score(x_lin_reg_array, y_lin_reg_array)

            plt.plot(domain_t, tab[i], label= "abs value")
            plt.plot(np.array(list_time_maxima[i]), np.array(list_maxima[i]), label="envelop")
            plt.plot(np.array(list_time_maxima[i]), 
                    np.exp(array_LinReg_a[i]*np.array(list_time_maxima[i]) + array_LinReg_b[i]),
                    label="Linear reg, (a, b, R²)= ({a}, {b},{r2})".format(
                        a=solver_BDF.round_fct(array_LinReg_a[i], 2), b=solver_BDF.round_fct(array_LinReg_a[i], 2),
                          r2=solver_BDF.round_fct(array_LinReg_r2[i], 2) ))
            plt.legend()
            plt.show()

        

        # assert (idx_decay_time[0]>0), "Decay time not determined. "
        decay_time_exp, decay_time_th = 1/array_LinReg_a, 1/lambda_k.real
        error_decay_rate = max( abs(abs(abs(decay_time_exp)-abs(decay_time_th))/abs(decay_time_th) ))
        print("carac times, exp & th:", decay_time_exp, decay_time_th)
        print("error for the exponential decay:", error_decay_rate*100, "%")
        # array of the carac decay time errors in % compared to the linear theory time.
        #  We take a fq from 3 to 10.
        array_err_oscilation = np.array([13.415033612929134,  5.884803310469058])
        

##### Mass conservation check
#We integrate the height to check the mass (rho_l is constant so mass is proportionnal to the volume). 
#For more details, see the Obsidian document.
print("\n\nMASSE CONSERVATION CHECK\n")

print("Parameters of the external pressure function:", A_Ns, mu_Ns, sigma_Ns)
plt.plot(domain_x, N_s_function(domain_x)[0])
plt.show()
if bool_solve_Spectral or bool_load_spectral:
    M_0 = np.trapz(h_mat_spectral[0, :])

    #Analytical variation rate of the total mass M for the Controled Benney eq
    M_variation = (h_mat_spectral[:, 0]**3)/3*(N_s_function(L_x)[1]- N_s_function(0)[1])
    print(N_s_function(L_x)[1])
    #Analytical total mass at each time
    M_analytics = M_0 + np.array([1]+[np.trapz(M_variation[:n_t]) for n_t in range(1, N_t)])

    #Numerical total mass M
    M_numerics = np.trapz(h_mat_spectral, axis=1)
    print(N_t, M_numerics.shape)
    print("Max difference between M_analytics and M_numerics : ", np.max(M_analytics-M_numerics))
    #Supposed to be very small. 
    print("Max diff in percent of mean(M_analytics): ",
           np.max(M_analytics-M_numerics)/np.mean(M_analytics)*100, "%")
    print("Mean and variance of the the spatial integral of h during time: ", 
          np.mean(M_numerics), np.std(M_numerics))


