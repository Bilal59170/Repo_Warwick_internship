#CODE FOR THE BENNEY EQUATION


###IMPORT
import numpy as np
import matplotlib.pyplot as plt
import control as ct  

import solver_BDF 
import Control_file as file_ctrl
from header import * 


print("\n\n#### BEGINING OF THE PRINT ###\n")


###################################

######## Simulation & Control SETTINGS ######

#####################################

### Boolean variables to control what action to do. Watch out to load the good file.
bool_FB_Control = True # bool of Feedback Control
bool_open_loop_control = False # open loop control i.e predicted

##Control
bool_LQR, bool_LQR_pos_part = False, False # LQR control & positive part of LQR control
#Positive ctrl with linear system
bool_positive_Ctrl, bool_solve_save_solus_opti, bool_load_solus_opti = False, False, False
#proportionnal control
bool_prop_ctrl = True



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
space_steps_array = np.array([128])
print("Modelisation parameters: (N_x, N_t) = ({N_x},)")
N_x, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(_N_x=space_steps_array[0],
                                                             _CFL_factor = CFL_factor)


## Initial Condition
h_mean, ampl_c, ampl_s, freq_c, freq_s = 1, delta, 0*delta, 1, 0 #delta from the header
Initial_Conditions = sincos(
        domain_x, h_mean, ampl_c, ampl_s, (2*np.pi/L_x)*freq_c, (2*np.pi/L_x)*freq_s)
if True:#Plot of initial condition
    plt.plot(domain_x, Initial_Conditions)
    plt.title("IC (t=0) for the normalized height h (Ac, As, fc, fs)"+
    "=({Ac} {As}, {fc}, {fs})".format(Ac=ampl_c, As=ampl_s, fc=freq_c, fs=freq_s))
    plt.show()



##########  Control  ##########

###Global parameters of the control independent from the type of control 
##External pressure parameters and external normal pressure function
sigma_Ns = 0.01
omega_Ns = 0.1
#Array of the ranking of the points used for the actuators
k_nb_act = N_x
array_used_points = np.arange(1, k_nb_act+1, 1)*N_x//(k_nb_act+1) #equi-spaced actuators
assert array_used_points.shape[0] == k_nb_act, "Problem of shape for the actuators"

#Actuators shape function (the peak function)
if False: #Gaussian TAKE COS GAUSSIAN FOR THE CONTROL, OTHERWISE ERROR
    N_s_function = lambda x:solver_BDF.N_s_derivatives_gaussian(
        x, A_Ns=A_Ns, sigma_Ns=sigma_Ns, array_used_points=array_used_points, L=L_x)
else:
    N_s_function = lambda x, Amplitudes_Ns:solver_BDF.N_s_derivatives_cos_gaussian(
        x, Amplitudes_Ns, omega=omega_Ns, array_used_points=array_used_points, L=L_x)



### Choice of control
#Openloop control: schelduled control, function u(t


A_Ns = None

if bool_FB_Control:#Linear Feedback Control, closed loop function of the type u(x(t))
    beta = 0.95#Only matter for LQR Control, not the positive one. 
    time_start_ctrl = 25 #Time where the control starts
    idx_time_start_ctrl =int(time_start_ctrl/T*N_t)


    from solver_BDF import matrices_ctrl

    # assert (Amplitudes_Ns is None), "fct solver_BDF: Problem of input" #Not supposed to be in input as computed by the ctrl
    A, B, Q, R = matrices_ctrl(beta, list_Re_Ca_theta= [Re, Ca, theta], array_actuators_index=array_used_points,
                                actuator_fct= lambda x: solver_BDF.actuator_fct_cos_gaussian(x, omega_Ns, L_x), 
                                N_x=N_x, L_x=N_x*dx)


    # print("Sum of the columns of A: ", A@np.ones(A.shape[1]))
    if bool_LQR or bool_LQR_pos_part:
        if bool_LQR:
            print("## LQR Control ##")
        else:
            print("## Positive part of LQR Control ##")
        print("Solving Riccati equation")
        K, _, _ = ct.lqr(A, B, Q, R) #gain matrix
        print("Dimension of the gain matrix:", K.shape)
        print("highest term of K", np.max(K))

    elif bool_positive_Ctrl: ### Positive Control with QP optimization problem
        print("POSITIVE Linear CONTROL")

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
        print("Proportionnal Control")
        K = np.zeros((k_nb_act, N_x))
        #We have -K@h in the solver, so we put -alpha here to have N_s=alpha (h-1)/delta.
        for i in range(k_nb_act):
            K[i, array_used_points[i]] = -file_ctrl.alpha_B_AJ 

else:  #No Control
    idx_time_start_ctrl = None
    A_Ns = np.zeros((N_t, k_nb_act)) #schedule of the amplitudes
    K=None #no feedback matrix





















##########################

######## SOLVING #######

##########################

### Boolean variables to control what action to do. Watch out to load the good file.
#FD method 
bool_solve_save_FD, bool_load_FD = False, False 
bool_anim_FD, bool_save_anim_FD = False, False

#Spectral method
bool_solve_save_spectral, bool_load_spectral = True, False
bool_anim_spectral, bool_save_anim_spectral = True, True

## Order BDF Scheme
order_BDF_scheme = 2



###### Finite Difference & BDF Scheme ######

### Solving & animation
title_file = 'Benney_equation_code\\FD_method_BDF_order{BDF_order}_Nx_{N_x}.txt'.format(
                BDF_order=order_BDF_scheme, N_x=N_x)
title_anim = 'Benney_equation_code\\anim_FD_anim_BDF_order{order_BDF}_Nx_{N_x}_multi_jet.mp4'.format(
        order_BDF = order_BDF_scheme, N_x=N_x)


_, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(
    _N_x=N_x, _CFL_factor = CFL_factor)
dx_2, dx_3, dx_4 = dx**2, dx**3, dx**4

if bool_solve_save_FD:
    h_mat_FD = solver_BDF.solver_Benney_BDF_FD(
        N_x=N_x, N_t= N_t, dx=dx, dt=dt, IC=Initial_Conditions,
        theta=theta, order_BDF_scheme=order_BDF_scheme, Ca=Ca, Re=Re, N_s_function=N_s_function,
        nb_percent=1)

    ##Saving the solution
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





###### SPECTRAL METHOD #########
### Solving
if bool_FB_Control:
    title_file = 'Benney_equation_code\\Spectral_method_BDF_order{BDF_order}_Nx_{N_x}_prop_Ctrl.txt'.format(
                        BDF_order=order_BDF_scheme, N_x=N_x, beta=beta)
    title_anim = (('Benney_equation_code\\Anim_Spectral_Ns_'+
                '_BDF{BDF_order}_Nx{N_x}_theta{theta}_prop_Ctrl.mp4').format(
                        BDF_order=order_BDF_scheme, N_x=N_x, theta=solver_BDF.round_fct(theta, 3), beta=beta))
else:
    title_file = 'Benney_equation_code\\Spectral_method_BDF_order{BDF_order}_Nx_{N_x}_NoCtrl.txt'.format(
                        BDF_order=order_BDF_scheme, N_x=N_x)
    title_anim = (('Benney_equation_code\\Anim_Spectral_Ns_'+
                '_BDF{BDF_order}_Nx{N_x}_theta{theta}_Test_theta{T}.mp4').format(
                        BDF_order=order_BDF_scheme, N_x=N_x, theta=solver_BDF.round_fct(theta, 3), T=T))

title_amplitude = title_file[:-4]+"_Ampl.txt"


_, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(
    _N_x=N_x, _CFL_factor = CFL_factor)
dx_2, dx_3, dx_4 = dx**2, dx**3, dx**4    
print("Number of Space and Time points:", (N_x, N_t))

if bool_solve_save_spectral:
    h_mat_spectral, amplitudes_spectral = solver_BDF.solver_Benney_BDF_Spectral(
        N_x=N_x, N_t= N_t, dx=dx, dt=dt, IC=Initial_Conditions, theta=theta, Ca=Ca, Re=Re,
        order_BDF_scheme=order_BDF_scheme, N_s_function=N_s_function, Amplitudes_Ns=A_Ns, 
        FB_Control=bool_FB_Control,  bool_LQR_pos_part= bool_LQR_pos_part, 
        positive_ctrl= bool_positive_Ctrl, K=K, idx_time_start_ctrl=idx_time_start_ctrl)

    if bool_FB_Control:
        assert (amplitudes_spectral.shape[1] == k_nb_act), "Shape problm gain matrix"
        amplitudes_spectral = np.concatenate((np.zeros((1, k_nb_act)), amplitudes_spectral[:-1,:]), axis=0) 
        assert (amplitudes_spectral.shape[1] == k_nb_act), "Shape problm gain matrix"
    ##Saving the solution
    np.savetxt(title_file, h_mat_spectral)
    np.savetxt(title_amplitude, amplitudes_spectral)


if bool_load_spectral:
    h_mat_spectral, amplitudes_spectral = np.loadtxt(title_file), np.loadtxt(title_amplitude)

    assert ((h_mat_spectral.shape[0]==N_t)
            and(h_mat_spectral.shape[1]==N_x)),"Solution loading: Problem of shape "


###Animation
if bool_anim_spectral:#Animation of benney numerical solution
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

    animation_Benney = solver_BDF.func_anim(
        _time_series=array_animation_spectral, _anim_space_array = domain_x,
        _anim_time_array = domain_t,
        title= "Benney height for (N_x, N_t, L_x, T, Re, Ca) =({N_x}, {N_t}, {L_x}, {T}, {Re}, {Ca})".format(
        N_x=N_x, N_t=N_t, L_x=L_x, T=T, Re=solver_BDF.round_fct(Re,3), Ca=solver_BDF.round_fct(Ca, 3)), 

        title_x_axis=r"x axis: horizontal inclined by $\theta$",
        title_y_axis= r"y-axis (inclined by $\theta$)",
        _legend_list = legend_list)
    plt.show()

if bool_anim_spectral and bool_save_anim_spectral:
    animation_Benney.save(title_anim) #needs the program ffmpeg installed and in the PATH














###################################################

#### Tests & Experiments& Control theory verification

######################################################

print("******** Tests & Experiments& Control theory verification ***********")


### Log amplitude of the Control
if bool_solve_save_spectral or bool_load_spectral:
    h_amplitude = np.max(np.absolute(h_mat_spectral-1), axis=1) #Eventhough we do a control on (h-1)/delta, we are interested in |h-1|
    plt.plot(domain_t, h_amplitude)
    # plt.xscale("log")
    plt.yscale('log')
    plt.xlabel("time t")
    plt.ylabel(r"$log(max_{x\in[0,L_x]}|h(x,t)-1|)$")


    if bool_FB_Control:
        if bool_LQR_pos_part:
            title_plot = (('Benney_equation_code\\plot_Spectral_Ns_'+
            '_BDF{BDF_order}_Nx{N_x}_theta{theta}_trunc_Ctrl_beta{beta}.png').format(
                    BDF_order=order_BDF_scheme, N_x=N_x, theta=round_fct(theta, 3), beta=beta))
            plt.title("Log amplitude of |h-1| along time with the  positive part of LQR Control,with \n"+
                        r"N_x={N_x}, $\beta$={beta}".format(N_x=N_x, beta=beta))
        elif bool_positive_Ctrl:
            title_plot = (('Benney_equation_code\\plot_Spectral_Ns_'+
            '_BDF{BDF_order}_Nx{N_x}_theta{theta}_pos_Ctrl.png').format(
                    BDF_order=order_BDF_scheme, N_x=N_x, theta=round_fct(theta, 3), beta=beta))
            plt.title("Log amplitude of |h-1| along time with positive Control from QP optimisation ,with \n"+
                        r"N_x={N_x},".format(N_x=N_x))
        elif bool_LQR:#Normal LQR Control
            title_plot = (('Benney_equation_code\\plot_Spectral_Ns_'+
                    '_BDF{BDF_order}_Nx{N_x}_theta{theta}_trunc_Ctrl_beta{beta}.png').format(
                            BDF_order=order_BDF_scheme, N_x=N_x, theta=round_fct(theta, 3), beta=beta))
            plt.title("Log amplitude of |h-1| along time with LQR Control,with \n" +
                       r"N_x={N_x}, $\beta$={beta}".format(N_x=N_x, beta=beta))
        elif bool_prop_ctrl:
            title_plot = (('Benney_equation_code\\plot_Spectral_Ns_'+
                    '_BDF{BDF_order}_Nx{N_x}_theta{theta}_propc_Ctrl.png').format(
                            BDF_order=order_BDF_scheme, N_x=N_x, theta=round_fct(theta, 3), beta=beta))
            plt.title("Log amplitude of |h-1| along time with Proportionnal Control, with \n" +
                       "N_x={N_x}, k={k}".format(N_x=N_x, k=k_nb_act))
        plt.axvline(x=time_start_ctrl, color='r')
    else:
        title_plot = (('Benney_equation_code\\plot_Spectral_Ns_'+
                    '_BDF{BDF_order}_Nx{N_x}_theta{theta}_NoCtrl.pdf').format(
                            BDF_order=order_BDF_scheme, N_x=N_x, theta=solver_BDF.round_fct(theta, 3)))
        
    plt.savefig(title_plot)
    plt.show()




