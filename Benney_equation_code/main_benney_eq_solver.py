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
N_x = 512 #To choose little at first and then, increase 
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


##Initial conditions 
#Somes functions to use for Initial Condition
def sincos(x, _h_mean, _ampl_c, _ampl_s, _freq_c, _freq_s):
    return _h_mean + _ampl_c*np.cos((2*np.pi/L_x)*_freq_c*x) + _ampl_s*np.sin((2*np.pi/L_x)*_freq_s*x) #Initial condition. (sinus, periodic on D)

#initial condition
h_mat = np.zeros((N_t, N_x)) #Matrix of the normalised height. Each line is for a given time from 0 to (N_t-1)*dt
h_mean = 1
ampl_c, ampl_s, freq_c, freq_s = 0.02, 0.01, 3, 1  #frequencies need to be a relative to have periodicity
h_mat[0,:] = sincos(domain_x, h_mean, ampl_c, ampl_s, freq_c, freq_s)

if True:#Plot of initial condition
    plt.plot(domain_x, h_mat[0, :])
    plt.title("Initial conditions for the normalized height h ")
    plt.show()



######## SOLVING #######

###### Finite Difference & BDF Scheme ######
## Some useful functions
def mat_FD_periodic(length_h, list_coef):
    '''Finite difference matrix with periodic boundary conditions. Cf matdiag notation in the "Finite difference" part 
    in the solving of KS equation in the obsidian file
    input: list_coef = [a, b, c, d, ..] list of coef (int)
    output: returns a mat with diagonal a, subdiag b (with periodic extension), supdiag c (same), 2nd subdiag d etc..'''

    n = len(list_coef)
    assert (n<=length_h), "fct mat_FD_periodic: problem in the dimensions"
    mat_FD = np.zeros((length_h, length_h))
    for i in range(n):
        nb_diag = (-1)**(i+1)* int((i+1)/2) # 0:0; 1:1; 2:-1; 3:2; 4:-2...

        for j in range(length_h):
            mat_FD[(j+nb_diag)%length_h, j%length_h] = list_coef[i]

    return mat_FD
#Tests
print(mat_FD_periodic(5, [0, -1, 1, -2, 2]))
# print(mat_FD_periodic(5, [0, -1, 1, -2, 2, 3])) #Case where the assert raises

#Finite Difference matrices
mat_DF_x = mat_FD_periodic(N_x, [0, -1, 1])/(2*dx)
mat_DF_xx = mat_FD_periodic(N_x, [-2, 1, 1])/dx_2
mat_DF_xxx = mat_FD_periodic(N_x, [3, -1, -3, 0, 1])/dx_3
mat_DF_xxxx = mat_FD_periodic(N_x, [6, -4, -4, 1, 1])/dx_4
print(mat_DF_x@h_mat[0, :])

def F_space(h_arr):
    '''Input: 
        - h_arr: array of height at time t+dt; (N_x) float array
        - h_arr_before: array of height at time t; (N_x) float array
    '''
    h_x = mat_DF_x@h_arr
    h_xx = mat_DF_xx@h_arr #no definition of h_xxx and h_xxxx bcs they are computed just once in the function

    return ( h_x*(h_arr**2)*(2*np.ones_like(h_arr)-2*h_x/np.tan(theta) + (1/Ca)*mat_DF_xxx@h_arr) 
            + (1/3)*(h_arr**3)*(-(2/np.tan(theta))*h_xx + (1/Ca)*mat_DF_xxxx@h_arr) 
            + (8*Re/15)*(6*(h_arr**5)*(h_x**2) + (h_arr**6)*h_xx))

def time_scheme_coef(_p):
    '''give the coefficient of the numerical time method'''
    match _p:
        case 1:
            return (np.array([-1, 1]), np.array([1]))
        case 2:
            return (np.array([1/2, -2, 3/2]), np.array([-1, 2]))
        case 3:
            return (np.array([-1/3, 3/2, -3, 11/6]), np.array([1, -3, 3]))
        case 4:
            return (np.array([1/4, -4/3, 3, -4, 25/12]), np.array([-1, 4, -6, 4]))
        case 5:
            return (np.array([-1/5, 5/4, -10/3, 5, -5, 137/60]), np.array([1, -5, 10, -10, 5]))
        case 6:
            return (np.array([1/6, -6/5, 15/4, -20/3, 15/2, -6, 147/60]), np.array([-1, 6, -15, 20, -15, 6]))
        case _:
            raise Exception("Function alpha_gamma_coef: Error in the calculus, wrong p value.")
        
def F_time(h_arr, h_arr_before):
    '''Function to step the order of the time step method'''
    return (h_arr-h_arr_before)/dt


## Testing the first time step
f_objective = lambda h_arr: F_time(h_arr, h_arr_before=h_mat[0,:]) + F_space(h_arr)

if False: #Testing of Newton's method: scipy.optimize.newton
    ##Test of the newton method with 1st order implicit time scheme
    print("\n#### TEST TO SET THE NEWTON METHOD ####")

    print("len(h_mat[0, :]) = N_x:", len(h_mat[0, :]) == N_x)

    '''One loop of Newton's method on the Initial Condition. 
    - Returns (root, converged, derivative). root: root of the function, converged: array of the converged dimensions,
    derivative: say if a derivative is 0 (means that the method struggles to CV)
    - Used previous height as starting point as the next one should be nearby'''
    t_i = time.time()
    root_one_loop, converged_one_loop, newton_derivatives = scipy.optimize.newton(f_objective, x0=h_mat[0, :]
                                            ,maxiter=50, full_output=True) 
    calculus_time = time.time()-t_i

    print("Computation time for one loop with N_x = {N_x}:".format(N_x=N_x), calculus_time)
    print("Number of converged coordinates, ratio/N_x ", np.sum(converged_one_loop), np.sum(converged_one_loop)/N_x)
    print("Number of derivatives = 0 (not good for the method):", np.sum(newton_derivatives))
    print("maximum of the error", np.max(np.absolute(f_newton(root_one_loop) )), "\n") 

    #See the overall physical validity of the computed height
    print("minimum new height (see if it's <0):", np.min(root_one_loop))
    print("maximum new height (see if it's too big):", np.max(root_one_loop))
    plt.plot(domain_x, root_one_loop)
    plt.title("height at t=dt with Newton method")
    plt.show()

if False: #Testing of the scipy.optimize.root function 
    ##Test of the newton method with 1st order implicit time scheme
    print("\n#### TEST of the scipy.optimize.root function ####")

    t_i = time.time()
    result = scipy.optimize.root(f_objective, x0=h_mat[0, :]) 
    calculus_time = time.time()-t_i
    root_2, converged_2, msg_2 = result["x"], result["success"], result["message"]

    print("Computation time for one loop with N_x = {N_x}:".format(N_x=N_x), calculus_time)
    print("Method root converged: ",converged_2)
    print("maximum of the error", np.max(np.absolute(f_objective(root_2))))

    #See the overall physical validity of the computed height
    print("minimum new height (see if it's <0):", np.min(root_2))
    print("maximum new height (see if it's too big):", np.max(root_2))
    plt.plot(domain_x, root_2)
    plt.title("height at t=dt with root method")
    plt.show()


## Solving
#1 order
print("\n## SOLVING BENNEY EQ ##")
#computation times :  ((N_x, N_t), t computation): [(128, 229),14s), ((256, 458), 60s), ((512, 915),T= 710s)]
t_i = time.time()
if False:#main loop
    root_method_CV_arr, root_method_errors_arr = np.zeros(N_t, dtype=bool), np.zeros(N_t)

    for n_t in range(N_t-1):
        fct_objective = lambda h_arr: F_time(h_arr, h_arr_before=h_mat[n_t,:]) + F_space(h_arr)
        result = scipy.optimize.root(fun= fct_objective, x0= h_mat[n_t,:]) 
        h_mat[n_t+1, :] = result["x"]
        root_method_CV_arr[n_t] = result["success"]
        root_method_errors_arr[n_t]= np.max(np.absolute(fct_objective(result["x"])))
        
        #Display of the computation progress
        nb_percent = 5 #The step of percent at which we display the progress
        if np.floor((100/nb_percent)*(n_t+1)/(N_t-1)) != np.floor((100/nb_percent)*(n_t)/(N_t-1)): #displays the progress of the computation every nb_percent
            print("Computation progress:", np.floor(100*(n_t+1)/(N_t-1)), "%; time passed until start: ", time.time()-t_i)

    total_computation_time = time.time()-t_i
    print("Total computation time:", total_computation_time)
    print("Number of time the method didn't converge & N_t", (np.sum(~root_method_CV_arr), N_t))
    print("Max error (evaluation on the supposed root) and its index", (np.max(root_method_errors_arr), np.argmax(root_method_errors_arr)))


##Saving or loading the solution
bool_save = False
bool_load_solution = False

if bool_load_solution:
    h_mat= np.loadtxt('Benney_equation_code\\Benney_numerical_solution_Nx_128.txt')
    assert ((h_mat.shape[0]==N_t)and(h_mat.shape[1]==N_x)), "Solution loading: Problem of shape "
elif bool_save: 
    np.savetxt('Benney_equation_code\\Benney_numerical_solution_Nx_{N_x}.txt'.format(N_x=N_x), h_mat)


### VISUALISATION 
##animation function
def round_fct(r, nb_decimal):
    '''Detect the power of 10 and round the number nb_decimal further'''
    if r==0:
        return 0
    else:
        power_10 = int(np.log10(r))
        factor = 10**(power_10)
        return round(r/factor, nb_decimal)*factor

def func_anim(_time_series, _anim_space_array, _anim_time_array, title, title_x_axis = None, title_y_axis= None, _legend_list = None):
 
    #(Nb_tab, N_t, N_x) tab
    Nb_time_series =  _time_series.shape[0]
    gap = (_time_series.max()- _time_series.min())/10

    #subplot initialisation
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))

    # Initialise the plot ligns
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


if False:#Animation of benney numerical solution
    animation_Benney = func_anim(_time_series=np.array([h_mat]), _anim_space_array = domain_x,
                                        _anim_time_array = domain_t,
                                        title="Benney height for (N_x, N_t, L_x, T, Re, Ca) =({N_x}, {N_t}, {L_x}, {T}, {Re}, {Ca})".format(N_x=N_x, N_t=N_t, L_x=L_x, T=T, Re=round_fct(Re,3), Ca=round_fct(Ca, 3)), 
                                        title_x_axis=r"x axis: horizontal inclined by $\theta$",
                                        title_y_axis= r"y-axis (inclined by $\theta$)",
                                        _legend_list = ["height h(x,t)"])

    plt.show()

    if True:
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

