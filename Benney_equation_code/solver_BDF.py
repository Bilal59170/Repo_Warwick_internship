## Explanations & output
# Part of the code where we solve the Benney equation. This code does not output anything in itself
# Cf the report Bilal_BM_report.pdf in the Github repository (part III to V)
# https://github.com/Bilal59170/Repo_Warwick_internship to know more about the theoretical background 


## Structure of the code
# - Some useful functions fot the solvers
#     - Function for the normal stress N_s
#     - Some unused tests
# - Solvers
#     - Solvers for Finite Difference and Spectra methods



###Imports
import numpy as np
import scipy.optimize 
import matplotlib.pyplot as plt
import time
from header import * 




#######################################################

######## Some useful functions for the solvers  ######

#####################################################


# Creates Finite Difference matrices: cf report part IV.2
def mat_FD_periodic(length_h, list_coef):
    '''Finite difference matrix with periodic boundary conditions, cf report part IV.2
    Inputs: 
    - list_coef = [a, b, c, d, ..] list of coef (int)
    output: 
    - returns a mat with diagonal a, subdiag b (with periodic extension), 
    supdiag c (same), 2nd subdiag d etc..'''

    n = len(list_coef)
    assert (n<=length_h), "fct mat_FD_periodic: problem in the dimensions"
    mat_FD = np.zeros((length_h, length_h))
    for i in range(n):
        nb_diag = (-1)**(i+1)* int((i+1)/2) # 0:0; 1:1; 2:-1; 3:2; 4:-2...

        for j in range(length_h):
            mat_FD[(j+nb_diag)%length_h, j%length_h] = list_coef[i]

    return mat_FD
if False:
    print(mat_FD_periodic(5, [0, -1, 1, -2, 2]))
    print(mat_FD_periodic(5, [0, -1, 1, -2, 2, 3])) #Case where the assert raises


#Computes BDF scheme part of the equation to solve: cf report part IV.2
def F_time(h_arr, h_arr_before, _p, dt):
    '''Output: 
    - The Benney equation part with the time derivative with a BDF Scheme. cf report part IV.2
    Input: 
    - h_arr: the height array at time t, shape (N_x)
    -h_arr: the height arrays at previous time until t-_p*dt, shape (_p, N_x)
    - _p: coefficient of the order BDF scheme'''

    _N_x = h_arr.shape[0]
    h_tot = np.concatenate((h_arr_before.reshape((_p, _N_x)), h_arr.reshape((1, _N_x))), axis=0)
    match _p:
        case 1: #increasing time from left to right
            return np.array([-1, 1])@h_tot/dt
        case 2:
            return np.array([1/2, -2, 3/2])@h_tot/dt
        case 3:
            return np.array([-1/3, 3/2, -3, 11/6])@h_tot/dt
        case 4:
            return np.array([1/4, -4/3, 3, -4, 25/12])@h_tot/dt
        case 5:
            return np.array([-1/5, 5/4, -10/3, 5, -5, 137/60])@h_tot/dt
        case 6:
            return np.array([1/6, -6/5, 15/4, -20/3, 15/2, -6, 147/60])@h_tot/dt
        case _:
            raise Exception("BDF Scheme function: Error in the calculus, wrong p value.")



## Function for the normal stress 
#Normal stress of each actuators (cf report part 5.1.2)
def actuator_fct_cos_gaussian(x, omega, L):
    '''Peak function d to model the normal stress of the control. 
    Described in the report part 5.1.2'''
    omega_L2_inv = 1 / (omega**2)
    nu = 2*np.pi/L
    x_nu = x*nu

    d = np.exp((np.cos(x_nu)-1)*omega_L2_inv)
    d_x = -d*(nu*np.sin(x_nu)*omega_L2_inv)
    d_xx=  d*(nu**2)*omega_L2_inv*(omega_L2_inv*np.sin(x_nu)**2
                                     -np.cos(x_nu))
    return d, d_x, d_xx


# Total Normal pressure N_s.
def N_s_derivatives_cos_gaussian(x, Amplitudes_NS, omega, array_used_points, L): 
    '''
    Array of the normal stress on the liquid-gas interface. It is a weighted sum of shape functions d defined
    with the function actuator_fct_cos_gaussian. The amplitudes are the variables being controled by the control 
    algorithm. 
    Input:
    -  
    Takes Amplitudes & actuators placements and outputs the total pressure on all the spatial points
    at for a given time.
    Inputs:
    - x: spatial domain 
    - Amplitudes_NS (array of size k): Array of the amplitude of each actuators
    - omega: parameter of the thickness of the peak. Taken to 0.1 normaly
    - array_used_points: Space localisation of the actuators
    -L: length of the domain'''
    
    assert (Amplitudes_NS.shape[0]==array_used_points.shape[0]), ("fct N_s_derivatives_cos_gaussian:Problem of input")

    nu = 2*np.pi/L
    # Amplitudes_NS = Amplitudes_NS[array_used_points] # shape (n,)
    mat_x_difference_nu = nu*(x[:, None]-x[None, array_used_points]) # (x_i-x_j)_{1<= i,j <= N_x}
    omega_L2_inv = 1 / (omega**2)
    
    N_s = Amplitudes_NS*np.exp((np.cos(mat_x_difference_nu)-1)*omega_L2_inv)
    N_s_x = -N_s*(nu*np.sin(mat_x_difference_nu)*omega_L2_inv)
    N_s_xx=  N_s*(nu**2)*omega_L2_inv*(omega_L2_inv*np.sin(mat_x_difference_nu)**2
                                     -np.cos(mat_x_difference_nu))

    return  np.array([np.sum(N_s, axis=1), np.sum(N_s_x, axis=1), np.sum(N_s_xx, axis=1)])
if False:
    plt.plot(np.linspace(0, 5, 100), N_s_derivatives(np.linspace(0, 5, 100), 2, 0, 1, L=30)[0])
    plt.show()

#Same than N_s_derivatives_cos_gaussian but with a gaussian peak function
def N_s_derivatives_gaussian(x, A_Ns, sigma_Ns, array_used_points, L): #Gaussian pressure profile
    '''Computes the Gaussian Normal pressure profile.
    Input: 
        x:points, 
        A: array of the used amplitude
        array_used_points: array of the index of the air jet actuators. It allows to have better performances. We 
                            always have to have array_used_points included in points of x. 
        L (float): Length of the plane. Used to normalize the gaussian
    Remark:
        - Watch out: a compressive air jet as modelled with A_Ns <0 as the liquid-gas 
        interface is modelled with a normal from the liquid to the gas. 
    '''

    if array_used_points is None:
        mat_x_difference = x[:, None]-x[None, :] # (x_i-x_j)_{1<= i,j <= N_x}
    else:
        A_Ns = A_Ns[array_used_points] # shape (n,)
        mat_x_difference = x[:, None]-x[None, array_used_points] # (x_i-x_j)_{i,j}, shape (N_x, n)

    # A_Ns = A_Ns[None, :]
    #Precomputation of usable quantities
    sigma_L=sigma_Ns*L #equivalent to have -((x-mu)/L)**2/(2sigma**2)
    sigma_L2_inv = 1 / (sigma_L ** 2)
    mat_x_diff_sigma = mat_x_difference * sigma_L2_inv
    # print(mat_x_difference[0])
    # print(mat_x_difference.T[0])

    N_s = A_Ns*np.exp(-(mat_x_difference**2)*0.5*sigma_L2_inv) # ()*(N_x, n)
    N_s_x = -N_s*mat_x_diff_sigma
    N_s_xx=  N_s*(mat_x_diff_sigma**2 - sigma_L2_inv)
    
    return  np.array([np.sum(N_s, axis=1), np.sum(N_s_x, axis=1), np.sum(N_s_xx, axis=1)])
if False: #Test to check the form of Ns, Ns_x, Ns_xx in simple cases
    N_x, L_x = 128, 30
    x_test = np.linspace(0, L_x, N_x)
    A = np.zeros(N_x)
    A[N_x//2] = 10
    sigma_Ns = 0.01
    print(A)
    N_der_array = N_s_derivatives_gaussian(x_test, A, sigma_Ns , L_x)
    plt.plot(x_test, N_der_array[2])
    plt.show()




### Some tests of the solving methods with Fd & Spectral methods: Newton and scipy.optimize. (Not used in a long time)          
## Testing the first time step for FD equation
if False:
    f_objective = lambda h_arr: F_time(h_arr, h_arr_before=h_mat[0,:],
        _p=order_BDF_scheme, dt=dt) + F_space(h_arr)

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

# testing the first time step For Spectral eq
if False:
    f_objective = lambda h_arr: F_time(h_arr, h_arr_before=h_mat[0,:], _p=order_BDF_scheme, dt=dt) + F_space(h_arr)

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










###################################

######## Solvers ######

##################################### 


###### Solver for the Benney equation with Finite DIfferences & BDF scheme

def solver_BDF(N_x, N_t, dt, IC, order_BDF_scheme, F_time, F_space, FB_Control, bool_pos_part,
               positive_ctrl, Amplitudes_Ns, K, idx_time_start_ctrl, nb_percent=5):

    '''
    Computes & outputs the computed numerical solution of the benney equation. Uses a BDF scheme to solve along the time axis.
    Call either a Finite Difference or a Spectral method for solving along the space axis. This choice is encoded in the 
    input F_space.



    Inputs:
    - N_x, N_t, dx, dt, IC, order_BDF_scheme: all the space-time discretization parameters, Initial condition
    and order of the BDF scheme used.
    - F_time: Function that outputs the time part of the equation (BDF scheme)
    - F_space: same as F_time but for space (finite difference or spectral method).

    - FB_Control (bool): Feedback Control is used or not
    - bool_pos_part (bool): Take the positive part of the control or not. Used in LQR or Proportional control 
    (cf report part V)
    - Amplitude_Ns: Array of the input amplitudes of the control for each actuators
    - K: Gain matrix 
    -idx_time_start_ctrl: time index where the control is turned on

    - nb_percent: the step in percent to show the progress of the computation.

    Outputs:
    - h_mat: (N_t, N_x) ndarray of the dynamics of h, the heigt of the interface gas-liquid
    - Amplitudes_NS: (N_t, N_x) ndarray. Schedule of the openloop control.
    - U_array: (N_t, N_x) ndarray. Distribution of the feedback control.

    '''

    # assert (N_s_function is N_s_derivatives_cos_gaussian)
    ##Initial conditions & steps
    h_mat = np.zeros((N_t, N_x)) #Matrix of the normalised height. 
    #Each line is for a given time from 0 to (N_t-1)*dt
    h_mat[0,:] = IC 

    ### Solving using a root finding method of scipy
    t_i = time.time()
    root_method_CV_arr, root_method_errors_arr = np.zeros(N_t, dtype=bool), np.zeros(N_t)

    if FB_Control:
        print("\n## SOLVING BENNEY EQ WITH FEEDBACK CTRL##")
        U_array = np.zeros((N_t, K.shape[0]))
    else:
        print("\n## SOLVING BENNEY EQ WITH OPEN LOOP CONTROL##")

    for n_t in range(N_t-1):
        #Computation of the control
        if FB_Control:
            u_ctrl = np.zeros(K.shape[0])

            if n_t>idx_time_start_ctrl: # After the starting time of the control
                if positive_ctrl:
                    u_ctrl = -K@h_mat[n_t, :] #Direct control on the height h
                    U_array[n_t] = u_ctrl
                else:
                    h_tilde = (h_mat[n_t, :]-1)/delta # Useless in the computation as we rescale again later with Ampl_Ns = delta*u_ctrl
                    u_ctrl = -K@h_tilde #Feedback ctrl with the previous state
                    if bool_pos_part:
                        u_ctrl = np.maximum(u_ctrl, np.zeros_like(u_ctrl))
                    U_array[n_t] = delta*u_ctrl #the rescalled real control on the height h=1+delta*\tilde{h}
                    
            Ampl_Ns = U_array[n_t]
        else:
            Ampl_Ns = Amplitudes_Ns[n_t, :]

            
        # Solving
        if n_t < order_BDF_scheme-1: #solving the first step with 1 order BDF (i.e backwards Euler)
            fct_objective = lambda h_arr: F_time(h_arr, h_arr_before=h_mat[n_t,:],
                                                 _p=1, dt=dt) + F_space(h_arr, Amplitudes_Ns=Ampl_Ns)
            
        else:
            fct_objective = lambda h_arr: F_time(h_arr, h_arr_before=h_mat[(n_t+1-order_BDF_scheme):n_t+1,:],
                                                _p=order_BDF_scheme, dt=dt) + F_space(h_arr, Amplitudes_Ns=Ampl_Ns)
            
        result = scipy.optimize.root(fun= fct_objective, x0= h_mat[n_t,:]) 
        h_mat[n_t+1, :] = result["x"]
        root_method_CV_arr[n_t] = result["success"]
        root_method_errors_arr[n_t]= np.max(np.absolute(fct_objective(result["x"])))
        

        #Display of the computation progress
        if np.floor((100/nb_percent)*(n_t+1)/(N_t-1)) != np.floor((100/nb_percent)*(n_t)/(N_t-1)):
            #displays the progress of the computation every nb_percent
            print("Computation progress:", np.floor(100*(n_t+1)/(N_t-1)), 
                "%; time passed until start: ", time.time()-t_i)

    total_computation_time = time.time()-t_i
    print("Total computation time:", total_computation_time)
    print("Number of time the method didn't converge & N_t", (np.sum(~root_method_CV_arr), N_t))
    print("Max error (evaluation on the supposed root) and its index",
           (np.max(root_method_errors_arr), np.argmax(root_method_errors_arr)))

    if FB_Control:
        return h_mat, U_array
    else:
        return h_mat, Amplitudes_Ns


def solver_Benney_BDF_FD(N_x, N_t, dx, dt, IC, theta, Ca, Re, order_BDF_scheme, N_s_function, 
                                Amplitudes_Ns, FB_Control=False, bool_pos_part=False, positive_ctrl=False,
                                K=None, idx_time_start_ctrl=None, nb_percent=5):
    '''
    Define the appropriate F_space for the Finite Difference scheme (F_space_FD) and calls the function solver_BDF 
    to solve the Benney equation with Finite Difference scheme.

    INPUTS:
        - N_x, N_t, dx, dt : space & time number of point and steps
        - IC: Initial Condition; 
        - theta, Ca & Re: slope angle of the plane, Capillary & Reynolds numbers 
        - order_BDF_Scheme: order of the BDF scheme used

        - N_s_function: Function that outputs the distribution of the normal stress
        - FB_Control (bool): Feedback control used or not
        - bool_pos_part (bool): take the positive part of the control or not (used in LQR or proportional control)
        - positive_ctrl (bool): use the positive control methodology defined in part V.4 of the report
        - K: Gain Matrix of the linear Feedback Control
        -idc_time_start_ctrl: time index of the time when the control is switched on

        - nb_percent (int): The step of percent at which we display the progress

    
    '''

    ##Steps
    L_x=N_x*dx
    domain_x = np.linspace(0, L_x, N_x, endpoint=False) 
    dx_2, dx_3, dx_4 = dx**2, dx**3, dx**4

    #Finite Difference matrices
    mat_DF_x = mat_FD_periodic(N_x, [0, -1, 1])/(2*dx)
    mat_DF_xx = mat_FD_periodic(N_x, [-2, 1, 1])/dx_2
    mat_DF_xxx = mat_FD_periodic(N_x, [3, -1, -3, 0, 1])/dx_3
    mat_DF_xxxx = mat_FD_periodic(N_x, [6, -4, -4, 1, 1])/dx_4


    def F_space_FD(h_arr, Amplitudes_Ns):
        '''
        Input: 
            - h_arr: array of height at time t+dt (Implicit method);
        Output: 
            - The Benney equation part with the space derivatives
        '''
        h_x = mat_DF_x@h_arr
        h_xx = mat_DF_xx@h_arr #no definition of h_xxx and h_xxxx bcs they are computed just once in the function

        N_s_der = N_s_function(domain_x, Amplitudes_Ns)

        return ( h_x*(h_arr**2)*(2*np.ones_like(h_arr)-N_s_der[1]-2*h_x/np.tan(theta) + (1/Ca)*mat_DF_xxx@h_arr) 
                - (1/3)*(h_arr**3)*(N_s_der[2]+(2/np.tan(theta))*h_xx - (1/Ca)*mat_DF_xxxx@h_arr) 
                + (8*Re/15)*(6*(h_arr**5)*(h_x**2) + (h_arr**6)*h_xx))


    return solver_BDF(N_x, N_t, dt, IC, order_BDF_scheme, F_time=F_time, F_space=F_space_FD, 
                        Amplitudes_Ns=Amplitudes_Ns, FB_Control=FB_Control,
                          bool_pos_part= bool_pos_part, K=K, positive_ctrl=positive_ctrl,
                        idx_time_start_ctrl=idx_time_start_ctrl, nb_percent=nb_percent)

 
def solver_Benney_BDF_Spectral(N_x, N_t, dx, dt, IC, theta, Ca, Re, order_BDF_scheme, N_s_function, 
                                Amplitudes_Ns, FB_Control=False, bool_pos_part=False, positive_ctrl=False,
                                K=None, idx_time_start_ctrl=None, nb_percent=5):
    '''
    Define the appropriate F_space for the Spectral method scheme (F_space_Spectral) and calls the function solver_BDF 
    to solve the Benney equation with Spectral method scheme.
    
    INPUTS:
        - N_x, N_t, dx, dt : space & time number of point and steps
        - IC: Initial Condition; 
        - theta, Ca & Re: slope angle of the plane, Capillary & Reynolds numbers 
        - order_BDF_Scheme: order of the BDF scheme used

        - N_s_function: Function that outputs the distribution of the normal stress
        - FB_Control (bool): Feedback control used or not
        - bool_pos_part (bool): take the positive part of the control or not (used in LQR or proportional control)
        - positive_ctrl (bool): use the positive control methodology defined in part V.4 of the report
        - K: Gain Matrix of the linear Feedback Control
        -idc_time_start_ctrl: time index of the time when the control is switched on

        - nb_percent (int): The step of percent at which we display the progress

    
    '''

    L_x = N_x*dx
    nu = (2*np.pi)/L_x
    fq_tab = nu*N_x*np.fft.rfftfreq(N_x) # or 2*np.pi*np.fft.rfftfreq(N_x, d=dx) ?
    domain_x = np.linspace(0, L_x, N_x, endpoint=False)
    print("\n Shape of the array of frequencies: ", fq_tab.shape)

    
    def F_space_Spectral(h_arr, Amplitudes_Ns):
        '''
        Input: 
            - h_arr: array of height at time t+dt (Implicit method);
        Output: 
            - The Benney equation part with the space derivatives. Computed with spectral method
        '''
        h_x = np.fft.irfft( (1j *fq_tab)*np.fft.rfft(h_arr))
        h_xx = np.fft.irfft( (1j *fq_tab)**2*np.fft.rfft(h_arr))
        h_xxx= np.fft.irfft( (1j *fq_tab)**3*np.fft.rfft(h_arr))
        h_xxxx= np.fft.irfft( (1j *fq_tab)**4*np.fft.rfft(h_arr))

        N_s_der = N_s_function(domain_x, Amplitudes_Ns)

        return ( h_x*(h_arr**2)*(2*np.ones_like(h_arr)-N_s_der[1]-2*h_x/np.tan(theta) + (1/Ca)*h_xxx) 
                - (1/3)*(h_arr**3)*(N_s_der[2]+(2/np.tan(theta))*h_xx - (1/Ca)*h_xxxx) 
                + (8*Re/15)*(6*(h_arr**5)*(h_x**2) + (h_arr**6)*h_xx))


    return solver_BDF(N_x, N_t, dt, IC, order_BDF_scheme, F_time=F_time, F_space=F_space_Spectral, 
                        Amplitudes_Ns=Amplitudes_Ns, FB_Control=FB_Control,
                          bool_pos_part= bool_pos_part, K=K, positive_ctrl=positive_ctrl,
                        idx_time_start_ctrl=idx_time_start_ctrl, nb_percent=nb_percent)

