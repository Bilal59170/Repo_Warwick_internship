## Explanations & output
# Part of the code where we control the Benney equation and observe the behaviour of the solution. This 
# code outputs animations (in the part "Solving"). It also output a plot of the variation of the height (in the part Tests
# &Experiments& Control theory verification) and different costs associated to the control. 
# This code also saves (or loads) the animations and arrays of the dynamics of the interface heigt h and the dynamics of the control
# used. It is thus important to choose a folder to manage the files that are going to be created 

# Cf the report Bilal_BM_report.pdf in the Github repository (part III to V)
# https://github.com/Bilal59170/Repo_Warwick_internship to know more about the theoretical background 




## Structure of the code:
# - Simulation & Control Settings
#   - Definition of several boolean variable to control what action to do
# 	- Initial condition stting
# 	- Choice and construction of the control strategy (4 different)
# 		- LQR (Linear Quadratic Regulator), proportional, positive system (doesn't work, cf part V.4 of the report)
# 	- parameters of the model (like N_x), not of the system (like L or T)
# - Solving
# 	- solving the equation with Spectral method 
# 	- display of animations, saving of animations and values of the solutions
# - Tests & Experiments& Control theory verification
# 	- plot of log amplitude of h, with or without exponential regression
#   - Computation of the total cost and maximum value of the cost




###IMPORT
import numpy as np
import matplotlib.pyplot as plt
import control as ct  

import solver_BDF 
import Control_file as file_ctrl
from header import * 

import scipy.optimize 
from sklearn.linear_model import LinearRegression


print("\n\n****  main_benney_eq_Ctrl.py: Beginning of the print ****\n")







###################################

######## Simulation & Control SETTINGS ######

#####################################

print("\n##### Simulation & Control SETTINGS #####")
### Boolean variables to control what action to do. Watch out to load the good file.
bool_FB_Control = True # bool of Feedback Control
bool_open_loop_control = False # open loop control i.e predicted

##Control
bool_pos_part = False
# LQR control & positive part of LQR control
bool_LQR = True 
#proportionnal control
bool_prop_ctrl = False
#Positive ctrl with linear system
bool_positive_Ctrl, bool_solve_save_solus_opti, bool_load_solus_opti = False, False, False


## Check the asumptions 
print("Nusselt velocity, Re, Ca: ", U_N, Re, Ca )
print("Value of Re (check if O(1)):", Re)
print("Value of Ca and Ca/(eps**2) (Check if O(eps**2)):", Ca, Ca/(epsilon**2))

assert (Re<1e2) and (Re>1e-2), r"False scaling assumption: $Ca \neq O(\epsilon^2)$ "
# assert (Ca/(epsilon**2))<1e2 and (Ca/(epsilon**2))>1e-2, r"False scaling assumption: $Re \neq O(1)$ "
# assert (1e-5<U_N) and (U_N<300), "Check if the sound speed is reached and if there's no too little speed"





#########  Simulation details  ############
##Time&Space steps
CFL_factor = 1
space_steps_array = np.array([512])
order_BDF_scheme = 4 # Order BDF Scheme


print("Modelisation parameters: (N_x, N_t) = ({N_x},)")
N_x, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(_N_x=space_steps_array[0],
                                                             _CFL_factor = CFL_factor)


## Initial Condition
mode_fq= 1
h_mean, ampl_c, ampl_s, freq_c, freq_s = 1, delta, 0*delta, mode_fq, mode_fq #delta from the header
Initial_Conditions = sincos(
        domain_x, h_mean, ampl_c, ampl_s, (2*np.pi/L_x)*freq_c, (2*np.pi/L_x)*freq_s)
if False:#Plot of initial condition
    plt.plot(domain_x, Initial_Conditions)
    plt.title("IC (t=0) for the normalized height h (Ac, As, fc, fs)"+
    "=({Ac} {As}, {fc}, {fs})".format(Ac=ampl_c, As=ampl_s, fc=freq_c, fs=freq_s))
    plt.show()



##########  Control  ##########
print("\n ##### Control settings #####")
###Global parameters of the control independent from the type of control 
##External pressure parameters and external normal pressure function
sigma_Ns = 0.01
omega_Ns = 0.1
#Array of the ranking of the points used for the actuators
k_nb_act = 5
array_used_points = np.arange(1, k_nb_act+1, 1)*N_x//(k_nb_act+1) #equi-spaced actuators
assert array_used_points.shape[0] == k_nb_act, "Problem of shape for the actuators"
time_start_ctrl = 160 #Time where the control starts
idx_time_start_ctrl =int(time_start_ctrl/T*N_t)


#Actuators shape function (the peak function)
if False: #Gaussian TAKE COS GAUSSIAN FOR THE CONTROL, OTHERWISE ERROR
    N_s_function = lambda x:solver_BDF.N_s_derivatives_gaussian(
        x, A_Ns=A_Ns, sigma_Ns=sigma_Ns, array_used_points=array_used_points, L=L_x)
else:
    N_s_function = lambda x, Amplitudes_Ns:solver_BDF.N_s_derivatives_cos_gaussian(
        x, Amplitudes_Ns, omega=omega_Ns, array_used_points=array_used_points, L=L_x)



### Choice of control
#Openloop control: scheduled control, function u(t)


A_Ns = None
beta, alpha_prop_ctrl, coef_pos_part_ctrl = None, None, None #defined later in their respective control (LQR, proportional..)

if bool_FB_Control:#Linear Feedback Control, closed loop function of the type u(x(t))
    
    #Matrix A and B of the linear Control system defined in part IV of the report 
    A, B = solver_BDF.matrices_ctrl_A_B(list_Re_Ca_theta = [Re, Ca, theta], array_actuators_index= array_used_points,
                              actuator_fct= lambda x: solver_BDF.actuator_fct_cos_gaussian(x, omega_Ns, L_x),
                                N_x=N_x, L_x=N_x*dx)
    
    if bool_LQR:#LQR Control
        beta = 0.95 #LQR Control parameter
        #LQR matrices
        Q, R = solver_BDF.matrices_ctrl_Q_R(beta = beta, array_actuators_index= array_used_points,
                                             actuator_fct= lambda x: solver_BDF.actuator_fct_cos_gaussian(x, omega_Ns, L_x),
                                            N_x=N_x, L_x=N_x*dx)

        #Check if R is positive definite as said in part V.2.1 of the report
        R_semi_def = np.all(np.linalg.eigvals(R) > 0)
        print("R is positive definite: ", np.all(np.linalg.eigvals(R) > 0))
        
        
        if bool_pos_part:
            print("### Type of control: Positive part of LQR Control ###")
        else:
            print("### Type of control: LQR Control ###")

        print("Solving Riccati equation")
        K, _, _ = ct.lqr(A, B, Q, R) # Computation of the gain matrix
        print("Dimension of the gain matrix:", K.shape)
        print("highest term of K", np.max(K))

    elif bool_positive_Ctrl: ### Positive Control with QP optimization problem (cf part )
        print("Type of control: Positive Linear Control")

        #Nb of unstable modes:
        A_eigval = np.linalg.eigvals(A)
        count_pos_eigval = np.sum(A_eigval.real>0)
        print("Number of unstable modes of A_test: ", count_pos_eigval)

        #Construction of the matrix for the QP opti problem
        B_delta=delta*B
        P_QP, G_QP, A_QP = file_ctrl.Mat_QP_problem(M_1=A, M_2=B_delta)


        ###Solving with solve_qp
        file_name_solus_opti = ('Benney_equation_code\\Positive_Ctrl\\'+
                                'Pos_Ctrl_Solver_solution_Nx{N_x}_k{k}.txt'.format(N_x=N_x, k=k_nb_act))
        
        if bool_solve_save_solus_opti:
            if False:
                d_init, z_init = np.ones(N_x), K_LQR.flatten()
                v_init_temp = np.concatenate([d_init, z_init])
                v_init = np.concatenate([v_init_temp, v_init_temp])
            else:
                v_init = None

            v_opti, d, z, K = file_ctrl.solve_opti_QP(N_x, k_nb_act, P_QP, G_QP, A_QP, v_init=v_init)
            np.savetxt(file_name_solus_opti, v_opti)
        elif bool_load_solus_opti:
            v_opti = np.loadtxt(file_name_solus_opti)

        # Check the QP problem constraints
        file_ctrl.check_QP_constraints(_G_QP=G_QP, _A_QP=A_QP, _v_opti=v_opti, _K = K)

        #Check if A+deltaBK is Metzler & Hurwitz
        A_deltaBK = A+(B_delta@K)
        file_ctrl.is_Metzler_and_Hurwitz(A_deltaBK)
  
    elif bool_prop_ctrl:
        if bool_pos_part:
            print("### Type of control: positive part of proportional control ###")
        print("### Type of control: Proportionnal Control ###")
        #Computation of the constant alpha
        def gain_mat_prop_ctrl(alpha):
            K = np.zeros((k_nb_act, N_x))
            for i in range(k_nb_act):
                K[i, array_used_points[i]] = alpha
            return K
        

        #Computation of the proportional control coefficient (eigenvalue problem, part V.3.1)
        prop_fct = lambda alpha: max(np.linalg.eigvals(A+B@gain_mat_prop_ctrl(alpha)).real)
        result_prop_ctrl_opti = scipy.optimize.root(fun= prop_fct, x0= file_ctrl.alpha_B_AJ) 
        alpha_prop_num = result_prop_ctrl_opti["x"][0]
        root_method_CV = result_prop_ctrl_opti["success"]
        root_method_errors= np.max(np.absolute(prop_fct(alpha_prop_num)))

        # if False:
        #     X_array = np.linspace(file_ctrl.alpha_B_AJ, 3*alpha_prop_num, 10)
        #     evaluation_array = np.array([prop_fct(x) for x in X_array])
        #     plt.axvline(x= alpha_prop_num, color = 'k')
        #     plt.plot(X_array, evaluation_array)
        #     plt.show()

        print("alpha num, alpha linear theory:", alpha_prop_num, file_ctrl.alpha_B_AJ)
        print("Method converged: ", root_method_CV)
        print("error of the method: ", root_method_errors)


        alpha_prop_ctrl = 100 #Coefficient for the proportional control
        K = gain_mat_prop_ctrl(-alpha_prop_ctrl)

    if bool_pos_part:
        #Multiplicative coefficient of the gain matrix (for LQR control, cf part V.2.2 of the report)
        #used when we take the positive part of the LQR control. Try to compense the loss of energy (so should be around 2 maybe).
        coef_pos_part_ctrl = 1
        K = coef_pos_part_ctrl*K #cf the definition of po_part_coef for explanations

else:  #No Control
    idx_time_start_ctrl = None
    A_Ns = np.zeros((N_t, k_nb_act)) #schedule of the amplitudes
    K=None #no feedback matrix



print("The parameters for the Controls:")
print("- beta : ", beta)
print("- alpha_prop_ctrl: ", alpha_prop_ctrl)
print("- coef_pos_part_ctrl: ", coef_pos_part_ctrl)










##########################

######## SOLVING #######

##########################

print("\n##### SOLVING #####")

### Boolean variables to control what action to do. Watch out to load the good file.

##Spectral method
#solve and save, load the solution respectively
bool_solve_save_Sp, bool_load_Sp = False, True 
#display or save the animation of the dynamics of h respectively
bool_anim_display_Sp, bool_save_anim_Sp = False, False 





## Fonction to write the name of the saved files
def file_anim_name_ctrl(method, Ctrl_name, pos_part, _N_x, _order_BDF, _alpha, _beta, _coef_pos_part_ctrl):
    '''Function to write automatically the file names to avoid making typos. 
    Returns the names of the animation and the file with numerical values.'''


    assert (method == "FD" or method == "Sp"), "file_anim_name_ctrl fct: problem in the name of the space scheme."
    assert (Ctrl_name in ["LQR", "prop", "positive"] ), "file_anim_name_ctrl fct: problem in the name of the Control."
    
    _str_pos_part = ""
    title_file_LQR, title_file_prop_ctrl = "", ""

    if pos_part:
        _str_pos_part = "pospart"+str(_coef_pos_part_ctrl)

    if Ctrl_name == "LQR":
        title_file_LQR = r"_beta{}".format(_beta,4)

    if Ctrl_name == "prop":
        title_file_prop_ctrl = r"_alpha{}".format(round_fct(_alpha,3))

    title_file = ('Benney_equation_code\\Control_verifications\\Ctrl_'+ Ctrl_name + _str_pos_part + '_' +
                    method + '_' + 'BDF{order_BDF}_Nx{N_x}'.format(order_BDF=_order_BDF, N_x=_N_x)
                    +title_file_prop_ctrl+ title_file_LQR + '.txt')
    title_anim = ('Benney_equation_code\\Control_verifications\\Anim_Ctrl_'+ Ctrl_name + _str_pos_part + '_' +
                    method + '_' + 'BDF{order_BDF}_Nx{N_x}'.format(order_BDF=_order_BDF, N_x=_N_x)
                    +title_file_prop_ctrl+ title_file_LQR + '.mp4')
    
    return title_file, title_anim


if bool_LQR:
    Ctrl_name = 'LQR'
elif bool_prop_ctrl:
    Ctrl_name = 'prop'
elif bool_positive_Ctrl:
    Ctrl_name = 'positive'





########## SOLVING #############

#title of the file with the value of the height and the animation of its dynamics
title_file, title_anim = file_anim_name_ctrl('Sp', Ctrl_name=Ctrl_name, pos_part=bool_pos_part,
                                             _N_x=N_x, _order_BDF=order_BDF_scheme, _alpha = alpha_prop_ctrl, _beta=beta,
                                             _coef_pos_part_ctrl=coef_pos_part_ctrl)

#Title of the file with the amplitude of the control, i.e the variable "amplitudes_spectral"
title_amplitude = title_file[:-4]+"_Ampl.txt" 


_, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(
    _N_x=N_x, _CFL_factor = CFL_factor)
dx_2, dx_3, dx_4 = dx**2, dx**3, dx**4    
print("Number of Space and Time points:", (N_x, N_t))


###Solves 
if bool_solve_save_Sp:
    h_mat_spectral, amplitudes_spectral = solver_BDF.solver_Benney_BDF_Spectral(
        N_x=N_x, N_t= N_t, dx=dx, dt=dt, IC=Initial_Conditions, theta=theta, Ca=Ca, Re=Re,
        order_BDF_scheme=order_BDF_scheme, N_s_function=N_s_function, Amplitudes_Ns=A_Ns, 
        FB_Control=bool_FB_Control,  bool_pos_part= bool_pos_part, 
        positive_ctrl= bool_positive_Ctrl, K=K, idx_time_start_ctrl=idx_time_start_ctrl)

    if bool_FB_Control:
        assert (amplitudes_spectral.shape[1] == k_nb_act), "Shape problm gain matrix"
        amplitudes_spectral = np.concatenate((np.zeros((1, k_nb_act)), amplitudes_spectral[:-1,:]), axis=0) 
        assert (amplitudes_spectral.shape[1] == k_nb_act), "Shape problm gain matrix"
    ##Saving the solution
    np.savetxt(title_file, h_mat_spectral)
    np.savetxt(title_amplitude, amplitudes_spectral)


###Loads an already computed numerical solution
if bool_load_Sp:
    h_mat_spectral, amplitudes_spectral = np.loadtxt(title_file), np.loadtxt(title_amplitude)

    assert ((h_mat_spectral.shape[0]==N_t)
            and(h_mat_spectral.shape[1]==N_x)),"Solution loading: Problem of shape "


###Animation
if bool_anim_display_Sp or bool_save_anim_Sp:#Animation of benney numerical solution
    #Construction of a function for plotting. Tee pressure is showed upside down and normalized. (cf Obsidian file)
    
    if bool_FB_Control or bool_open_loop_control:
        N_s_mat_spectral = np.array([solver_BDF.N_s_derivatives_cos_gaussian(
            domain_x, amplitudes_spectral[n_t],omega_Ns, array_used_points, L_x)[0] for n_t in range(N_t)]) 
        N_s_mat_spectral[:idx_time_start_ctrl,:] = np.zeros_like(N_s_mat_spectral[:idx_time_start_ctrl,:])
        N_s_mat_spectral = 2-N_s_mat_spectral/np.max(np.absolute(N_s_mat_spectral)) #Normalization & upside down
        print("Shape N_s_mat_spectral:", N_s_mat_spectral.shape)
    else:
        print(amplitudes_spectral.shape)
        N_s_mat_spectral = 2*np.ones((N_t, N_x))

    
    array_animation_spectral = np.array([h_mat_spectral, N_s_mat_spectral])
    if bool_FB_Control:
        legend_list =  ["h(x,t) with spectral method & BDF order {}".format(order_BDF_scheme),
                        "Ns Feedback Control"]
    else:
        legend_list =  ["h(x,t) with spectral method & BDF order {}".format(order_BDF_scheme),
                            "Ns Open loop Control"]

    animation_Benney = func_anim(
        _time_series=array_animation_spectral, _anim_space_array = domain_x,
        _anim_time_array = domain_t,
        title= "Benney height for (N_x, N_t, L_x, T, Re, Ca) =({N_x}, {N_t}, {L_x}, {T}, {Re}, {Ca})".format(
        N_x=N_x, N_t=N_t, L_x=L_x, T=T, Re=solver_BDF.round_fct(Re,3), Ca=solver_BDF.round_fct(Ca, 3)), 

        title_x_axis=r"x axis: horizontal inclined by $\theta$",
        title_y_axis= r"y-axis (inclined by $\theta$)",
        _legend_list = legend_list)
    if bool_anim_display_Sp:
        plt.show()
    else:
        plt.close()

    if bool_save_anim_Sp:
        animation_Benney.save(title_anim) #needs the program ffmpeg installed and in the PATH








###################################################

#### Tests & Experiments& Control theory verification

######################################################

print("##### Tests & Experiments& Control theory verification #####")


### Boolean variables to control what action to do. Watch out to load the good file.
bool_plot = False #plot the log of h_max 
bool_reg_lin = False #Do a exponential regression of this plot after that the control has been switched on


#Function to name the plots and title them automatically
def title_plot_ctrl(method, Ctrl_name, pos_part, _N_x, _order_BDF, _alpha, _beta, _coef_pos_part_ctrl):
    '''Function to write automatically the file names to avoid making typos. 
    Returns the names of the animation and the file with numerical values.'''


    assert (method == "FD" or method == "Sp"), "file_anim_name_ctrl fct: problem in the name of the space scheme."
    assert (Ctrl_name in 
            ["LQR", "prop", "positive"] ), "file_anim_name_ctrl fct: problem in the name of the Control."
    

    str_pos_part, str_pos_part_plot = "", ""
    title_plot_pos_part = ""
    title_file_LQR, title_plot_LQR = "", ""
    title_file_prop_ctrl, title_plot_prop_ctrl = "", ""


    if pos_part:
        str_pos_part, str_pos_part_plot = ("pospart"+str(_coef_pos_part_ctrl), 
                                           " positive part of ")
        title_plot_pos_part = r", $\zeta_+$={}".format(_coef_pos_part_ctrl)

    if Ctrl_name == "LQR":
        title_file_LQR = r"_beta{}".format(_beta)
        title_plot_LQR = r", beta={}".format(_beta)
    if Ctrl_name == "prop":
        title_file_prop_ctrl = r"_alpha{}".format(_alpha)
        title_plot_prop_ctrl = r", $\alpha$={}".format(_alpha)



    title_plot_file = ('Benney_equation_code\\Control_verifications\\Plot_Ctrl_'+ Ctrl_name + str_pos_part + '_' +
                    method + '_' + 'BDF{order_BDF}_Nx{N_x}'.format(order_BDF=_order_BDF, N_x=_N_x)+
                      title_file_LQR+ title_file_prop_ctrl+ '.png')
    
    if Ctrl_name =="prop":
        Ctrl_name = "proportional"

    title_plot = ("Log(max|h-1|) with "+str_pos_part_plot + Ctrl_name+ " control , \n with "+
                        r"N_x={N_x}".format(N_x=N_x)+title_plot_LQR + title_plot_prop_ctrl+
                        title_plot_pos_part)
    
    return title_plot_file, title_plot



### Log amplitude of the Control
h_amplitude = np.max(np.absolute((h_mat_spectral-1)), axis=1) #Eventhough we do a control on (h-1)/delta, we are interested in |h-1|


##Exponential regression to compute the dampening rate  
if bool_reg_lin: 
    t1, t2 = time_start_ctrl, T
    N1, N2 = int(t1/T*N_t), int(t2/T*N_t)
    h_ampl_lin_reg, array_t_reg_lin = h_amplitude[N1:N2], domain_t[N1:N2]

    #Linear regression of the differences
    x_lin_reg_array = np.log10(array_t_reg_lin).reshape(-1,1) #Necessary reshape for sklearn

    #L2 Case
    y_lin_reg_array = np.log10(h_ampl_lin_reg).reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True).fit(x_lin_reg_array, y_lin_reg_array) #sklearn function
    #slope a, intercept b and  Determination coefficient R²
    Reg_lin_coef_a , Reg_lin_coef_b= reg.coef_[0][0], reg.intercept_[0]
    Reg_lin_coef_r2 = reg.score(x_lin_reg_array, y_lin_reg_array)
    rescaled_lin_reg = 10**(Reg_lin_coef_b)*(array_t_reg_lin**Reg_lin_coef_a)

    print("LIN REG: ", Reg_lin_coef_a, Reg_lin_coef_b, Reg_lin_coef_r2)

if bool_plot:
    plt.plot(domain_t, h_amplitude)
    if bool_reg_lin:
        plt.plot(array_t_reg_lin, rescaled_lin_reg, 
            label="Linear Regression, \n "+r"(a, b, R²) $\approx$ ({a}, {b}, {R2})".format(
            a=solver_BDF.round_fct(Reg_lin_coef_a, 2), b=solver_BDF.round_fct(Reg_lin_coef_b, 2)
            , R2=solver_BDF.round_fct(Reg_lin_coef_r2, 4)),
            color='r')
    # plt.xscale("log")
    plt.yscale('log')
    plt.xlabel("time t", fontsize=15)
    plt.ylabel(r"$h_{max}$", fontsize=15)


    if bool_FB_Control:
        title_plot_file, title_plot = title_plot_ctrl(
            method='Sp', Ctrl_name= Ctrl_name, pos_part=bool_pos_part, _N_x=N_x, _order_BDF=order_BDF_scheme,
                _alpha = round_fct(alpha_prop_ctrl,3), _beta=beta, _coef_pos_part_ctrl=coef_pos_part_ctrl)
        # plt.title(title_plot)
        plt.axvline(x=time_start_ctrl, color='k')

    plt.legend(fontsize=15, framealpha=1)
    # plt.savefig(title_plot_file)
    plt.show()



## Computation: quadratic cost of the control and maximum value of the control
max_ampl_ctrl = np.max(np.absolute(amplitudes_spectral))
print("infinite norm of the control:", max_ampl_ctrl)
quad_cost_ctrl = np.sum(amplitudes_spectral**2)*dx**2*dt**2
print("Total cost before the control starts (expected to be 0): ",
       np.sum(amplitudes_spectral[0:idx_time_start_ctrl, :]**2))
print("Total cost of the control: ", quad_cost_ctrl)


