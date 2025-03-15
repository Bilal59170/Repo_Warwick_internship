## Explanations & output
#Code to make all the verifications and comparisons for the numerical schemes. This code can output a lot of 
# annimations and plots for the verification of the schemes.  
# Cf the part (IV.3) of report Bilal_BM_report.pdf in the Github repository  
# https://github.com/Bilal59170/Repo_Warwick_internship to know more about the theoretical background 


## Structure of the Code
# -  Simulation SETTINGS 
#    - System setting
#    - Choice to make air jets or not
# - Verification of the Convergence and comparison of the Finite Difference & Spectral scheme 
#    - Boolean variables to control what to do
#    - Convergence of the schemes: animations and plots
#    - Comparison of Finite Difference & Spectral methods schemes
# - Linear theory verification (obsolete, not in the report)
# - Check for mass conservation




###IMPORT
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import solver_BDF 
from header import *

print("\n\n******  Verification of the Scheme: Beginning of the print *****\n")


###################################

######## Simulation SETTINGS ######

##################################### 


# Using of air jets or not. Interpretated as open loop control
bool_open_loop_control = False 



#########  Simulation details  ############
##Simulation constants
CFL_factor = 1
# array of the space steps that we compute/use to make the verifications.
#  They are several as we compute several solutions in a for loop.
space_steps_array = np.array([128, 256, 512, 1024]) 
list_BDF_order = [1]  #Same than above but for the BDF Orders
order_BDF_scheme_verif = 4 #BDF Order for the verification with a fixed order and varying N_x which are the convergence
#verification, and comparison between the Finite Difference and Spectral method 
print("Space steps: ", space_steps_array)

##Initial Condition
mode_fq = 1 # frequency mode of the initial condition.
h_mean, ampl_c, ampl_s, freq_c, freq_s = 1, 0, delta, mode_fq, mode_fq

if False:#Plot of initial condition
    plt.plot(domain_x, Initial_Conditions)
    plt.title("IC (t=0) for the normalized height h (Ac, As, fc, fs)"+
    "=({Ac} {As}, {fc}, {fs})".format(Ac=ampl_c, As=ampl_s, fc=freq_c, fs=freq_s))
    plt.show()


## Air jets (Can be seen as a sort of open loop control)
k_nb_act = 5 ## numbers of actuators

#Actuators shape function (the peak function)
omega_Ns = 0.1 # width of the air jet for the peak function  (funcition "d" defined in part 5.1.2 of the report)
sigma_Ns =0.01 # width of the air jet for a gaussian peak function 

# Choose a gaussian actuatio, gives no mass conservation because it is not L-periodic (cf end of the code),
#  better to chose the cos-gaussian 
if bool_open_loop_control:
    if False: #Gaussian actuation
        N_s_function = lambda x:solver_BDF.N_s_derivatives_gaussian(
            x, A_Ns=A_Ns, sigma_Ns=sigma_Ns, array_used_points=array_used_points, L=L_x)
    else: # cos gaussian actuation i.e the L-periodic function 
        N_s_function = lambda x, Amplitudes_Ns:solver_BDF.N_s_derivatives_cos_gaussian(
            x, Amplitudes_Ns, omega=omega_Ns, array_used_points=array_used_points, L=L_x)

else: # No control 
    N_s_function = lambda x, Amplitudes_Ns: np.array([np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)])









##############################################################

######## Verification of the Convergence and comparison of
#  the Finite Difference & Spectral scheme (Cf report part IV.3) 

##############################################################



### Boolean variables to control what action to do. Watch out to load the good file.
#FD method 
bool_solve_save_FD, bool_load_FD = False, False 
bool_anim_display_FD, bool_save_anim_FD = False, False
bool_verif_BDF_scheme_FD, bool_verif_FD = False, False

#Spectral method
bool_solve_save_Sp, bool_load_Sp = False, False
bool_anim_display_Sp, bool_save_anim_Sp = False, False
bool_verif_BDF_scheme_Sp, bool_verif_Sp = False, False

bool_comparison_FD_Sp = True
assert not(bool_verif_BDF_scheme_FD and bool_verif_BDF_scheme_Sp), "BDF scheme verif: Can't make 2 verification in the same time"



######### Solving & animation #########

#Function to name automatically the file names that will be saved or loaded in the computer
def file_anim_name(method, _N_x, _order_BDF):
    '''Function to write automatically the file names to avoid making typos. 
    Returns the names of the animation and the file with numerical values.
    Input: 
    - Method (string): "FD" or "Sp" for Finite Difference or Spectral method
    - _N_x, _order_BDF (int) : Space step et BDF order
    '''

    if method == "FD": #Finite differences Method
        title_file = ('Benney_equation_code\\Schemes_verification\\Simulation_values\\'+
                      'Verif_FD_BDF{order_BDF}_Nx{N_x}.txt'.format(order_BDF=_order_BDF, N_x=_N_x))
        title_anim = ('Benney_equation_code\\Schemes_verification\\Simulation_values\\'+
                      'Anim_Verif_FD_BDF{order_BDF}_Nx{N_x}.mp4'.format(order_BDF = _order_BDF, N_x=_N_x))
        
    elif method == "Sp": #Spectral Method
        title_file = ('Benney_equation_code\\Schemes_verification\\Simulation_values\\'+
                      'Verif_Sp_BDF{order_BDF}_Nx{N_x}.txt'.format(order_BDF=_order_BDF, N_x=_N_x))
        title_anim = ('Benney_equation_code\\Schemes_verification\\Simulation_values\\'+
                      'Anim_Verif_Sp_BDF{order_BDF}_Nx{N_x}.mp4'.format(order_BDF = _order_BDF, N_x=_N_x))
        
    else:
        return "fct file_name: Error in the name of the method"
    
    return title_file, title_anim



#### Spectral method & BDF Scheme

for  order_BDF_solve in list_BDF_order:
    print("### Order of the BDF Scheme for Spectral method: ", order_BDF_solve)
    for i in range(len(space_steps_array)):

        #Simulation setting
        N_x, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(
            _N_x=space_steps_array[i], _CFL_factor = CFL_factor)
        print("Modelisation parameters: (N_x, N_t) = ({N_x},{N_t})".format(N_x=N_x, N_t=N_t))
        
        # File name 
        title_file, title_anim = file_anim_name('Sp', N_x, order_BDF_solve)
        
         #Initial condition
        Initial_Conditions = sincos(
            domain_x, h_mean, ampl_c, ampl_s, (2*np.pi/L_x)*freq_c, (2*np.pi/L_x)*freq_s)
        
        #Open loop Control
        array_used_points = np.arange(1, k_nb_act+1, 1)*N_x//(k_nb_act+1) #equi-spaced actuators
        A_Ns = np.zeros((N_t, k_nb_act)) #schedule of the amplitudes

        if bool_solve_save_Sp:
            #Don't save the control just the height dynamics
            h_mat_spectral, _ = solver_BDF.solver_Benney_BDF_Spectral(
                N_x=N_x, N_t= N_t, dx=dx, dt=dt, IC=Initial_Conditions, theta=theta, Ca=Ca, Re=Re,
                order_BDF_scheme=order_BDF_solve, Amplitudes_Ns=A_Ns, N_s_function=N_s_function,
                nb_percent=5)

            ##Saving the solution
            np.savetxt(title_file, h_mat_spectral)

    
        if bool_load_Sp:
            h_mat_spectral= np.loadtxt(title_file)

            assert ((h_mat_spectral.shape[0]==N_t)
                    and(h_mat_spectral.shape[1]==N_x)),"Solution loading: Problem of shape "


        ###Animation
        if bool_anim_display_Sp or bool_save_anim_Sp:
            ## Construction of the animation
            if bool_open_loop_control:
                # #Construction of a function for plotting. Tee pressure is showed upside down and normalized. (cf Obsidian file)
                # N_s_mat_spectral = np.array([solver_BDF.N_s_derivatives_cos_gaussian(
                #     domain_x, ctrl_mat_spectral[n_t],omega_Ns, array_used_points, L_x)[0] for n_t in range(N_t)]) 
                # N_s_mat_spectral[:idx_time_start_ctrl,:] = np.zeros_like(N_s_mat_spectral[:idx_time_start_ctrl,:])
                # N_s_mat_spectral = 2-N_s_mat_spectral/np.max(np.absolute(N_s_mat_spectral)) #Normalization & upside down
                # print("Shape N_s_mat_spectral:", N_s_mat_spectral.shape)

                # array_animation_spectral = np.array([h_mat_spectral, N_s_mat_spectral])
                # legend_list =  ["h(x,t) with spectral method & BDF order {}".format(order_BDF_scheme), "Ns Control"]
                pass 
            else:
                array_animation_spectral = np.array([h_mat_spectral])
                legend_list =  ["h(x,t) with spectral method & BDF order {}".format(order_BDF_solve)]

            animation_Benney = solver_BDF.func_anim(
                _time_series=array_animation_spectral, _anim_space_array = domain_x,
                _anim_time_array = domain_t,
                title= "Benney height for (N_x, N_t, L_x, T, Re, Ca) =({N_x}, {N_t}, {L_x}, {T}, {Re}, {Ca})".format(
                N_x=N_x, N_t=N_t, L_x=L_x, T=T, Re=solver_BDF.round_fct(Re,3), Ca=solver_BDF.round_fct(Ca, 3)), 

                title_x_axis=r"x axis: horizontal inclined by $\theta$",
                title_y_axis= r"y-axis (inclined by $\theta$)",
                _legend_list = legend_list)
            
            if bool_anim_display_Sp: #Animation of benney numerical solution
                plt.show()  
            else:
                plt.close()

            if bool_save_anim_Sp:
                animation_Benney.save(title_anim) #needs the program ffmpeg installed and in the PATH



####Finite Difference & BDF Scheme 

for  order_BDF_solve in list_BDF_order:
    print("### Order of the BDF Scheme for FD method: ", order_BDF_solve)
    for i in range(len(space_steps_array)):

        #SImulation setting
        N_x, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(
            _N_x=space_steps_array[i], _CFL_factor = CFL_factor)
        print("Modelisation parameters: (N_x, N_t) = ({N_x},{N_t})".format(N_x=N_x, N_t=N_t))

        # File name 
        title_file, title_anim = file_anim_name('FD', N_x, order_BDF_solve)

        #Initial condition
        Initial_Conditions = sincos(
            domain_x, h_mean, ampl_c, ampl_s, (2*np.pi/L_x)*freq_c, (2*np.pi/L_x)*freq_s)
        
        #Open loop Control
        array_used_points = np.arange(1, k_nb_act+1, 1)*N_x//(k_nb_act+1) #equi-spaced actuators
        A_Ns = np.zeros((N_t, k_nb_act)) #schedule of the amplitudes

        if bool_solve_save_FD:
            #computation times : ((N_x, N_t), t computation): 
            # [(128, 229),14s), ((256, 458), 60s), ((512, 915),T= 710s)]
            h_mat_FD,  _ = solver_BDF.solver_Benney_BDF_FD(
                N_x=N_x, N_t= N_t, dx=dx, dt=dt, IC=Initial_Conditions, theta=theta, Ca=Ca, Re=Re,
                order_BDF_scheme=order_BDF_solve, Amplitudes_Ns=A_Ns, N_s_function=N_s_function,
                nb_percent=5)

            ##Saving the solution
            np.savetxt(title_file, h_mat_FD)


        ##Loading the solution
        if bool_load_FD:
            h_mat_FD= np.loadtxt(title_file)
            # print(h_mat_FD.shape)
            assert ((h_mat_FD.shape[0]==N_t)and(h_mat_FD.shape[1]==N_x)), "Solution loading: Problem of shape"


        ### VISUALISATION 

        ##animation function
        if bool_anim_display_FD or bool_save_anim_FD:
            animation_Benney = func_anim(_time_series=np.array([h_mat_FD]), 
                    _anim_space_array = domain_x, _anim_time_array = domain_t,
                    title="Benney height for (N_x, N_t, L_x, T, Re, Ca) =({N_x}, {N_t}, {L_x}, {T}, {Re}, {Ca})"
                    .format( N_x=N_x, N_t=N_t, L_x=L_x, T=T, Re=round_fct(Re,3), Ca=round_fct(Ca, 3)), 

                    title_x_axis=r"x axis: horizontal inclined by $\theta$",
                    title_y_axis= r"y-axis (inclined by $\theta$)",
                    _legend_list = ["height h(x,t) with FD method and BDF order {}".format(order_BDF_solve)])

            if bool_anim_display_FD:#Animation of benney numerical solution
                plt.show()
            else:
                plt.close()
            if bool_save_anim_FD: #saves the animation
                animation_Benney.save(title_anim)  #needs the program ffmpeg installed and in the PATH






############## Verifications #############


#Load the arrays
list_h_mat_FD, list_h_mat_Sp = [], [] #list of the Spectral & BDF numerical solution for different space steps
for space_step in space_steps_array:
    title_file_FD, _ = file_anim_name("FD", space_step, order_BDF_scheme_verif)
    list_h_mat_FD.append(np.loadtxt(title_file_FD))

    title_file_Sp, _ = file_anim_name("Sp", space_step, order_BDF_scheme_verif)
    list_h_mat_Sp.append(np.loadtxt(title_file_Sp))



##Verification for the BDF Scheme: For a fixed N_x we compute the height for all the BDF orders (1 until 6)
# We make an Animation on the whole time domaine & a plot at the final time

if bool_verif_BDF_scheme_FD or bool_verif_BDF_scheme_Sp:
    N_x_plot = 512 #Fixed number of space point 
    N_x, N_t, _, _, domain_x, domain_t = set_steps_and_domain(_N_x=N_x_plot, _CFL_factor=CFL_factor)
    list_result_BDF_FD_N_x, order_BDF_list = [], [1, 2, 3,4, 5,6]
    index_eval_time = int(N_t)-1

    plt.rc('font', size=18)


    # Plot of the final time for 
    for i in range( len(order_BDF_list)):
        if bool_verif_BDF_scheme_FD:
            name_method, full_name_method = "FD", "Finite Difference"
            file_name, _ = file_anim_name("FD", N_x_plot, order_BDF_list[i] )
            print("Visualisation of several BDF Scheme solutions for FD method for fixed step size N_x = ", N_x_plot)
        elif bool_verif_BDF_scheme_Sp:
            name_method, full_name_method = "Sp", "Spectral"
            file_name, _ = file_anim_name("FD", N_x_plot, order_BDF_list[i] )
            print("Visualisation of several BDF Scheme solutions for Sp method for fixed step size N_x = ", N_x_plot)
        list_result_BDF_FD_N_x.append(np.loadtxt(file_name))

        label_plot = fr"$N_{{BDF}} = {order_BDF_list[i]}$"
        plt.plot(domain_x, list_result_BDF_FD_N_x[i][index_eval_time,:], 
                 label=label_plot)
        
    plt.xlabel("x axis: inclined plane"), 
    if bool_verif_BDF_scheme_FD:
        plt.title(fr"$h_{{FD}}^{{{N_x_plot}, N_{{BDF}}}}(t=T)$", fontsize=20)
    elif bool_verif_BDF_scheme_Sp:
        plt.title(fr"$h_{{Sp}}^{{{N_x_plot}, N_{{BDF}}}}(t=T)$", fontsize=20) 
    plt.legend(fontsize=16, loc="upper center")
    plt.title(full_name_method+ " method with BDF order from 1 to 6 at final time\n" +
               "T = {T} with N_x = {N_x}".format(T=T, N_x = N_x_plot))
    plt.savefig('Benney_equation_code\\Schemes_verification\\BDF_order_test\\'+"plot_"+
            name_method+'_BDF_order_comparison_order_1-6_Nx_{}.png'.format(N_x_plot))
    plt.show()


    # Animation
    if False:
        if bool_verif_BDF_scheme_FD:
            legend_list = [fr"$h_{{FD}}^{{{N_x_plot},{BDF_order}}}$" for BDF_order in order_BDF_list]
        elif bool_verif_BDF_scheme_Sp:
            legend_list = [fr"$h_{{Sp}}^{{{N_x_plot},{BDF_order}}}$" for BDF_order in order_BDF_list]

        # Animation
        animation_BDF_FD = solver_BDF.func_anim(
            _time_series=np.array(list_result_BDF_FD_N_x), _anim_space_array = domain_x,
            _anim_time_array = domain_t,
            title="Benney height with "+ name_method +" method and several BDF order. \n"+
            r" (Re, Ca, $\theta$, $\delta$)="+"({Re}, {Ca}, {theta}, {delta}), (N_x, N_t)=({N_x}, {N_t})".format(
            N_x=N_x_plot, N_t=N_t, Re=round_fct(Re,3), Ca=round_fct(Ca, 3), theta=round_fct(theta, 3),
            delta=delta), 

            title_x_axis=r"x axis: horizontal inclined by $\theta$",
            title_y_axis= r"y-axis (inclined by $\theta$)",
            _legend_list = legend_list
            )
        # plt.show()

        animation_BDF_FD.save(
            'Benney_equation_code\\Schemes_verification\\BDF_order_test\\'+
            name_method+'_BDF_order_comparison_order_1-6_Nx_{}.mp4'.format(N_x_plot))  



###Verification of the convergence of the schemes: Fixed BDF order and varying N_x

# (Big) function to plot the graphs to study the convergence of the methods 
def plot_difference_graph(list_h_mat, general_subplot_title, order_BDF, save_plot = False, 
                          file_name= None, linear_regression=False): #Difference in loglog graph
    '''
    Output: Make a loglog (in log10 scale) graph of several computed solutions with the solution 
    that we computed with the most number of space points (in the code, it's N_x=1024). 

    Example: The comparison between (128, 1024) would be at dx = L_x/128   
    Input: 
        - list_h_mat: list of the numerical solutions computed with INCREASING number of steps. 
        Usually [128, 256, 512, 1024]
        - general_subplot_title: head title of the subplot
        - order_BDF: order of the BDF scheme used
        - save_plot (bool): save the plot or not
        - file_name (string): if save_plot, path of the file to save
         - linear_regression (bool): to make and display a linear regression on the plot or not '''
    
    plt.rc('font', size=16)

    ##Compute the differences : with increasing number of points
    arr_L2_diff, arr_Linf_diff = np.zeros(len(space_steps_array)-1), np.zeros(len(space_steps_array)-1)
    dx_array = L_x/space_steps_array[:-1]
    #remove some part of the array so that they can be substracted: 128=2**7, 1024=2**10 
    for i in range(len(space_steps_array)-1):
        #For L2 norm: need to scale with the dx of the compared array
        arr_L2_diff[i] = np.sqrt(dx_array[i]*np.sum((list_h_mat[-1][-1,0::(2**(3-i))]-list_h_mat[i][-1])**2))
        arr_Linf_diff[i] = np.max(np.absolute((list_h_mat[-1][-1,0::(2**(3-i))]-list_h_mat[i][-1])))

    ###Plot
    fig, axs = plt.subplots(1,2, figsize=(15, 5))


    #Make a linear regression
    if linear_regression:
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
                label="Linear Regression, \n "+r"(a, b, R²) $\approx$ ({a}, {b}, {R2})".format(
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
    axs[0].set_xlabel("dx, space step of the least precise simulation")
    axs[0].set_title(fr"$||h^{{ 1024,{order_BDF}}}-h^{{ N, {order_BDF} }}||_2$"+
        r" for $N \in \{128, 256, 512\}$")
    axs[0].legend(loc='upper left', fontsize= 15)
 
    axs[1].scatter(dx_array, arr_Linf_diff, label="Scheme differences")
    rescaled_lin_reg_Linf = 10**(Reg_lin_coef_b_Linf)*(dx_array**Reg_lin_coef_a_Linf)
    axs[1].plot(dx_array,rescaled_lin_reg_Linf, 
            label="Linear Regression, \n "+r"(a, b, R²) $\approx$ ({a}, {b}, {R2})".format(
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
    axs[1].set_xlabel("dx, space step of the least precise simulation")
    axs[1].set_title(fr"$||h^{{ 1024,{order_BDF}}}-h^{{ N, {order_BDF} }}||_{{\infty}}$"+
        r" for $N \in \{128, 256, 512\}$ ")
    axs[1].legend(loc='upper left', fontsize=15)

    fig.suptitle(general_subplot_title)

    if save_plot:
        assert not(file_name is None), "fct \"plot_difference_graph\": Problem in the save_plot"
        plt.savefig(file_name)

    plt.show()


#Plot the convergence graph for either one of the method
if bool_verif_FD or bool_verif_Sp: 

    if bool_verif_FD:
        print("Plot of the difference graphs for Finite Difference method")
        name_method, full_name_method = "FD", "Finite Difference"
    elif bool_verif_Sp:
        print("Plot of the difference graphs for Spectral method")
        name_method, full_name_method = "Sp", "Spectral"
    #list of the FD & BDF numerical solution for different space steps
  
    
    #plot the difference graph
    file_name = ("Benney_equation_code\\Schemes_verification\\Pseudo_convergence\\"+"Report"+
                 name_method+" BDF_order{}_difference_graph".format(order_BDF_scheme_verif))
    general_subplot_title = ""

    # general_subplot_title= (full_name_method + r" method, $N_{{BDF}}$ = {}: Convergence plot "+
    #     "at t=T in a log-log plot.\n  ")
    
    plot_difference_graph(
        list_h_mat_FD, 
        general_subplot_title= general_subplot_title,
        order_BDF=order_BDF_scheme_verif,
        save_plot=False,
        linear_regression=True,
        file_name=file_name
        )








###### Comparison of the FD & Spectral methods  #######
bool_anim = False

#Animation of the difference between a solution computed with Finite Difference and Spectral method (not in the 
# report, just by curiosity.)
if bool_anim:

    N_x_plot = 128
    _ , N_t, _, _, domain_x, domain_t = set_steps_and_domain(_N_x=N_x_plot, _CFL_factor=CFL_factor)

    h_mat_FD = np.loadtxt(
            'Benney_equation_code\\Saved_numerical_solutions'
            + '\\FD_method_BDF_order{BDF_order}_Nx_{N_x}.txt'.format(
            BDF_order=order_BDF_scheme_verif, N_x=N_x_plot))
    h_mat_spectral = np.loadtxt(
            'Benney_equation_code\\Saved_numerical_solutions'
            +'\\Spectral_method_BDF_order{BDF_order}_Nx_{N_x}.txt'.format(
            BDF_order=order_BDF_scheme_verif, N_x=N_x_plot))
    
    animation_FD_Spectral = solver_BDF.func_anim(
        _time_series=np.array([h_mat_spectral-h_mat_FD]), _anim_space_array = domain_x,
        _anim_time_array = domain_t,
        title="Spectral-FD BDF order {}".format(order_BDF_scheme_verif) +
             "and (N_x, N_t, L_x, T, Re, Ca) =({N_x}, {N_t}, {L_x}, {T}, {Re}, {Ca})".format(
            N_x=N_x_plot, N_t=N_t, L_x=L_x, T=T, Re=solver_BDF.round_fct(Re,3), 
            Ca=solver_BDF.round_fct(Ca, 3)),

        title_x_axis=r"x axis: horizontal inclined by $\theta$",
        title_y_axis= r"y-axis (inclined by $\theta$)",
        _legend_list = ["h_spectral-h_FD"])
    # plt.show()
    animation_FD_Spectral.save(
        'Benney_equation_code\\Spectral-FD_anim_BDF_order{order_BDF}_1236_Nx_{N_x}.mp4'.format(
            order_BDF=order_BDF_scheme_verif, N_x=N_x_plot))


##Plot of the Difference between Spectral and FD Method with BDF
bool_linear_reg = False
if bool_comparison_FD_Sp: 
    # plt.rc('font', size=15)
    arr_L2_diff, arr_Linf_diff = np.zeros(len(space_steps_array)), np.zeros(len(space_steps_array))
    dx_array = L_x/space_steps_array
    for i in range(len(space_steps_array)):
        arr_L2_diff[i] = np.sqrt(dx_array[i]*np.sum((list_h_mat_FD[i][-1]-list_h_mat_Sp[i][-1])**2))
        arr_Linf_diff[i] = np.max(np.absolute(list_h_mat_FD[i][-1]-list_h_mat_Sp[i][-1]))

    ##plot
    fig, axs = plt.subplots(1,2, figsize=(15, 5))


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

        rescaled_lin_reg_L2 = 10**(Reg_lin_coef_b_L2)*(dx_array**Reg_lin_coef_a_L2)
        axs[0].plot(dx_array,rescaled_lin_reg_L2, 
        label=r"Linear Regression, (a, b, R²) $\approx$ ({a}, {b}, {R2})".format(
        a=solver_BDF.round_fct(Reg_lin_coef_a_L2, 2), b=solver_BDF.round_fct(Reg_lin_coef_b_L2, 2)
        , R2=solver_BDF.round_fct(Reg_lin_coef_r2_L2, 4)),
        color='r')

        rescaled_lin_reg_Linf = 10**(Reg_lin_coef_b_Linf)*(dx_array**Reg_lin_coef_a_Linf)
        axs[1].plot(dx_array,rescaled_lin_reg_Linf, 
        label=r"Linear Regression, (a, b, R²) $\approx$ ({a}, {b}, {R2})".format(
        a=solver_BDF.round_fct(Reg_lin_coef_a_Linf, 2), b=solver_BDF.round_fct(Reg_lin_coef_b_Linf, 2)
        , R2=solver_BDF.round_fct(Reg_lin_coef_r2_Linf, 4)),
        color='r')

    axs[0].scatter(dx_array, arr_L2_diff,label="Difference between Spectral & FD")
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
            textcoords= "offset pixels",
            fontsize = 15)
    axs[0].set_xlabel("dx", fontsize = 15)
    axs[0].set_title(r"$||h_{FD}^{N_x, 4}(T)-h_{Sp}^{N_x,4}||_{L^p}$, $p=2$", fontsize=17)
    # axs[0].legend(fontsize=15)

    axs[1].scatter(dx_array, arr_Linf_diff, label="Difference between FD & Spectral")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    annotations = [{'text': text_steps[i], 'xy': (dx_array[i], arr_Linf_diff[i])} for i in range(len(text_steps))]
    for annotation in annotations:
        axs[1].annotate(
            annotation['text'],
            xy=annotation['xy'],
            xytext=(0, 5),
            textcoords= "offset pixels",
            fontsize=15)

    axs[1].set_xlabel("dx", fontsize=15)
    axs[1].set_title(r"$||h_{FD}^{N_x, 4}(T)-h_{Sp}^{N_x,4}||_{L^{p}}$, $p = +\infty$", fontsize=17)
    # axs[1].legend(fontsize=15)

    # fig.suptitle("Comparison between the Spectral and FD methods with BDF of order {}".format(order_BDF_scheme_verif))
    plt.savefig("Benney_equation_code\\Schemes_verification\\Comparison_FD_Sp\\"
                +"Report_Plot_Comparison_Spectral_FD_BDF_order_{}.png".format(order_BDF_scheme_verif))
    plt.show()











##############################################################

########  Verification of linear theory (obsolete, not in the report)######

##############################################################
# The goal of this part was to see if the linear theory was respected or not. We derived analytically
# the Fourier coefficient in the linear theory and we compared them to the numerical one. The analytical 
# coefficients have either a dampening or increasing exponential regime. Hence, two parameters characterise 
# the Fourier coefficient: the frequency of oscillation and the variation rate of the envelop ( i.e the exponential rate)

# What we did: I computed the frequency rate by making a FFT (Fast Fourier Transform) and took the dominant mode.
# As for the exponential rate, we compute max(log(h-1)) and look make a linear regression on it (sort of exponential
# regression) and look at the slope coefficient.

## Bool variables to control what to do 
bool_linear_analysis = False
bool_solve_save_linear, bool_load_linear = False, False
bool_anim_linear, bool_save_anim_linear = False, False

#Check that it doesn't load and solve at the same time
assert (not(bool_solve_save_linear) or not(bool_load_linear)), "Linear analysis: Problem in booleans initialisation"


if False: #Visualisation of linear modes (dampening or amplificating regimes) 
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
        absciss = np.linspace(k_0_squared-1, 0, 100, endpoint=True)
        plt.plot(absciss, coef_dispersion(absciss).real)
        plt.title("k_0**2 <0")
        plt.axvline(x=k_0_squared, color='b')
        plt.axhline(y=0, color='r', linestyle='-') #draws a line at x=0 to see more clearly the sign of the points
        plt.show()
    else:
        k_0 = np.sqrt(k_0_squared)
        print("k_0", k_0)
        absciss = np.linspace(0, k_0, 100, endpoint=True)
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
    print("\n LINEAR ANALYSIS")
    ##parameters & IC
    N_x, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(
    _N_x=128, _CFL_factor = CFL_factor, T=50)
    A_Ns = np.zeros((N_t, k_nb_act)) #schedule of the amplitudes

    print("Number of time points:", N_t)
    #IC and order of the Fourier decomposition that we are interested in
    order_fourier_mode = 1
    IC_amplitude = 0.001
    ampl_c, ampl_s= IC_amplitude, 0
    Initial_Conditions = sincos(
        domain_x, _h_mean=1, _ampl_c=ampl_c, _ampl_s=ampl_s, 
        _freq_c=nu*order_fourier_mode, _freq_s=nu*order_fourier_mode )
    

    ##loading or computing solutions + Animations
    title_simulation = 'Benney_equation_code\\Linear_verif_Sp_method_BDF_order{BDF_order}_Nx_{N_x}.txt'.format(
                     BDF_order=order_BDF_scheme_verif, N_x=N_x)
    if bool_load_linear: ##Loading the solution. Be carefull to take the relevant IC
        h_mat_lin= np.loadtxt(title_simulation) 
        assert ((h_mat_lin.shape[0]==N_t)and(h_mat_lin.shape[1]==N_x)), "Solution loading: Problem of shape "
    if bool_solve_save_linear: #Computing
        h_mat_lin, _ = solver_BDF.solver_Benney_BDF_Spectral(
            N_x=N_x, N_t= N_t, dx=dx, dt=dt, IC=Initial_Conditions, theta=theta, Ca=Ca, Re=Re,
            order_BDF_scheme=order_BDF_scheme_verif, N_s_function=N_s_function, Amplitudes_Ns=A_Ns, 
            LQR_Control=bool_open_loop_control, K=K, 
            idx_time_start_ctrl=idx_time_start_ctrl)
        np.savetxt(title_simulation, h_mat_lin)

    if bool_anim_linear: #VIsualisation of the behavior (amplify, reduces)
        animation_Benney = func_anim(
            _time_series=np.array([h_mat_lin]),
                _anim_space_array = domain_x, _anim_time_array = domain_t,
            title="Benney height forIC= {A}cos(v{k}) (N_x, N_t, L_x, T, Re, Ca) =({N_x}, {N_t}, {L_x}, {T}, {Re}, {Ca})".
            format(
                N_x=N_x, N_t=N_t, L_x=L_x, T=T, Re=round_fct(Re,3), Ca=round_fct(Ca, 3),
                k=order_fourier_mode, A=IC_amplitude), 

            title_x_axis=r"x axis: horizontal inclined by $\theta$",
            title_y_axis= r"y-axis (inclined by $\theta$)",
            _legend_list = ["height with Spectral method"]
            )
        # plt.close()
        plt.show()
        if bool_save_anim_linear:
            animation_Benney.save(
                'Benney_equation_code\\Linear_theory_FD_method_animation_BDF_order{order_BDF}_Nx_{N_x}_k_{k}.mp4'.
                format(
                order_BDF = order_BDF_scheme, N_x=N_x, k=order_fourier_mode))  
    
    ##Comparison with the theoretical linear regime
    #Fourier matrix 
    h_Fourier_mat = np.fft.rfft(h_mat_lin, norm="forward", axis=1)
    print("Shape Fourier mat", h_Fourier_mat.shape)
    print("Shape h matrix", h_mat_lin.shape)
    assert ((h_Fourier_mat.shape[0]==h_mat_lin.shape[0])), "Shape of the Fourier matrix: Problem"

    #Theoretical Fourier coefficient c_k in Linear theory
    lambda_k = coef_dispersion(order_fourier_mode)
    print("lambda k:", lambda_k)
    ck_0 = (ampl_c-(ampl_s)*(1.j))/2
    ck = ck_0*np.exp(lambda_k*domain_t) #cos: /2; sin: /2j
    print("Experimental ck_0: ", h_Fourier_mat[0, order_fourier_mode])
    print("Lin theory ck_0: ", ck_0)

    if True: #Plot of the Fourier coef c_k
        plt.plot(domain_t, h_Fourier_mat[:, order_fourier_mode].real, label="real part")
        plt.plot(domain_t, h_Fourier_mat[:, order_fourier_mode].imag, label="imaginary part")
        plt.plot(domain_t, ck.real, label="Linear th real")
        plt.plot(domain_t, ck.imag, label="Linear th imag")
        plt.xlabel("time t")
        plt.title("{k}th fourier mode c_k(t)".format(k=order_fourier_mode))
        plt.legend()
        plt.savefig('Benney_equation_code\\Linear_theory_Sp_method_graph_BDF_order{order_BDF}_Nx_{N_x}_k_{k}.pdf'.
                    format(order_BDF=order_BDF_scheme, N_x=N_x, k=order_fourier_mode))
        plt.show()
        

    if True: #Plot of h_max
        plt.plot(domain_t, np.max(np.absolute(h_mat_lin-1), axis=1))
        plt.xlabel('time t')
        plt.yscale('log')
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
        










##############################################################

######## Physical properties check (cf part IV.3 of the report) ######

##############################################################
bool_mass_conservation = False

if bool_mass_conservation:##### Mass conservation check
    #We integrate the height to check the mass (rho_l is constant so mass is proportionnal to the volume). 
    #Pay attention that the integration domain (domain_t or domain_x )is in the inputs otherwise there 
    #will be some dt or dx multiplicative error.
    #For more details, see the Obsidian document.

    print("\n\nMASS CONSERVATION CHECK\n")

    N_s_der_distribution  = N_s_function(domain_x, A_Ns[0, :])

    print("External pressure function at initial time:", A_Ns[0, :], sigma_Ns)
    print("shape", N_s_der_distribution.shape)
    plt.plot(domain_x, N_s_der_distribution[0])
    # plt.plot(domain_x, N_s_der_distribution[1], label="_x")
    # plt.plot(domain_x, N_s_der_distribution[2], label="_xx")
    # plt.legend()
    plt.show()
    print("extremal values N_s_function and derivatives:", N_s_der_distribution[:, 0], N_s_der_distribution[:, -1])
    if (bool_solve_save_Sp or bool_load_Sp):
        N_x, N_t, dx, dt, domain_x, domain_t = set_steps_and_domain(
            _N_x=space_steps_array[0], _CFL_factor = CFL_factor)
        M_0 = np.trapz(y=h_mat_spectral[0, :], x=domain_x) #initial mass
        print("Initial mass:", M_0)
        #Analytical variation rate of the total mass M for the Controled Benney eq
        M_variation = (h_mat_spectral[:, 0]**3)/3*(N_s_der_distribution[1, -1]- N_s_der_distribution[1, 0])
        print("extremal values N_s_function and derivatives:", N_s_der_distribution[:, 0], N_s_der_distribution[:, -1])
        #Analytical total mass at each time (integration)
        M_analytics = M_0 + np.array([0]+[np.trapz(y=M_variation[:n_t+1], 
                                                x=domain_t[:n_t+1]) for n_t in range(1, N_t)])
        # print("M_Analytics: ", M_analytics)
        #Numerical total mass M
        M_numerics = np.trapz(y=h_mat_spectral, x=domain_x, axis=1)
        print("Max difference between M_analytics and M_numerics : ", np.max(np.absolute(M_analytics-M_numerics)))
        #Supposed to be small. 
        print("Max diff in percent of mean(M_analytics): ",
            np.max(np.absolute(M_analytics-M_numerics))/np.mean(M_analytics)*100, "%")
        print("Mean and variance of the the spatial integral of h during time: ", 
            np.mean(M_numerics), np.std(M_numerics))
