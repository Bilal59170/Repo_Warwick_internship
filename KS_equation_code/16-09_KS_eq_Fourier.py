##Imports
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

## Initialisations
dx = 1/100
dt = dx/10
N_x, N_t = int(1/dx)+1, int(1/dt)+1 #number of space (resp.time) point(s)
print("\n NB OF SPACE AND TIME POINTS:", N_x, N_t)
print("Time T:", dt*N_t)

U_mat = np.zeros((N_t, N_x)) #Matrix of the velocity. Each line is for a given time from 0 to N_t*dt
D = np.linspace(0, 2*np.pi, N_x) #Domain
U_mat[0,:] = np.sin(D) #Initial condition. (sinus, periodic on D)




##Some test about DFT
#Test of the differentiation with DFT 
fq_tab = N_x*np.fft.rfftfreq(N_x) #to justify better after
print("\n Shape of the array of frequencies: ", fq_tab.shape)
if False:
    U_x = np.fft.irfft( (1j *fq_tab)*np.fft.rfft(U_mat[0, :]))
    print(U_x.real)
    plt.plot(D, U_x.real)
    plt.title("Derivative of u_0 (sin -> cos) \n with Fourier (Gibbs phenomenon ?)")
    plt.show()

    U_xx = np.fft.irfft( (1j *fq_tab)**2*np.fft.rfft(U_mat[0, :]))
    print(U_xx)
    plt.plot(D, U_xx.real)
    plt.title("2nd Derivative of u_0 (sin -> -sin) \n with Fourier (Gibbs phenomenon ?)")
    plt.show()

    U_xxxx = np.fft.irfft( (1j *fq_tab)**4*np.fft.rfft(U_mat[0, :]))
    print(U_xxxx)
    plt.plot(D, U_xxxx.real)
    plt.title("4th Derivative of u_0 (sin -> sin) \n with Fourier (Gibbs phenomenon ?)")
    plt.show()




## Definitions and visualisation
#Fourier matrix
U_Fourier_mat = np.zeros((N_t, len(fq_tab)), dtype=complex)
U_Fourier_mat[0, :] = np.fft.rfft(U_mat[0, :])
print("\nShape of the Fourier matrix:", U_Fourier_mat.shape)

#H Coefficient (like the paper)
H_0 = U_Fourier_mat[0, 0]/N_x
print("\n Mean of the initial Function:", H_0)
H_c = 2*U_Fourier_mat[0, 1:].real #Cf calculus. We link the convention of the paper with the DFT convention
H_s = -2*U_Fourier_mat[0, 1:].imag 
print("\n Shape of Hc and H_s: ", H_c.shape)
M = len(H_c) #Order of trucation

#plot of the IC in frequency domain
plt.plot(fq_tab[1:], H_c, label="Real part")
plt.plot(fq_tab[1:], H_s, label="Imag part")
plt.legend()
plt.title("IC in frequency domain")
plt.show()




##compute the F + some speed tests
def F_slow(H_c, H_s):
    F_c = np.zeros(H_c.shape)
    F_s = np.zeros(H_s.shape)

    for j in range(M):
        for m in range(M):
            for n in range(M): 
                if (m+n == j ):
                    F_c[j] -= H_c[m]*H_s[n]
                    F_s[j] += (H_c[m]*H_c[n] - H_s[m]*H_s[n])/2
                elif (m-n == j):
                    F_c[j] += H_c[m]*H_s[n] - H_c[n]*H_s[m]
                    F_s[j] += H_c[m]*H_c[n] + H_s[m]*H_s[n]
        F_c[j] = F_c[j]*j/2
        F_s[j] = F_s[j]*j/2
    
    return F_c, F_s

def F(H_c, H_s):
    '''We suppose here that the mean value of the fct is always 0. Hence we manipulate only H_s and H_c'''
    F_c = np.zeros(M)
    F_s = np.zeros(M)

    # Use broadcasting to calculate H_c * H_s and H_c * H_c and H_s * H_s
    H_c_matrix = H_c[:, None]  # Convert to column vector
    H_s_matrix = H_s[:, None]  # Convert to column vector

    H_c_H_s = H_c_matrix * H_s_matrix.T
    H_s_H_c = H_s_matrix * H_c_matrix.T
    H_c_H_c = H_c_matrix * H_c_matrix.T
    H_s_H_s = H_s_matrix * H_s_matrix.T

    m_indices = np.arange(M)
    n_indices = np.arange(M)

    for j in range(M):
        # Find m+n == j
        mask1 = m_indices[None, :] + n_indices[:, None] == j  # Shape (M+1, M+1)
        F_s[j] += np.sum( (H_c_H_c - H_s_H_s) * mask1)/2
        F_c[j] -= np.sum(H_c_H_s* mask1)
        
        # Find m-n == j
        mask2 = m_indices[:, None] - n_indices[None, :] == j  # Shape (M+1, M+1)
        F_s[j] += np.sum( (H_c_H_c + H_s_H_s) * mask2)
        F_c[j] += np.sum( (H_c_H_s - H_s_H_c) * mask2) #watch out: the summation isn't symetric i.e the order of the indices m&n is important

        # Apply the scaling
        F_c[j] *= j / 2
        F_s[j] *= j / 2

    return F_c, F_s

#Speed tests
bool_speed_test = True
if bool_speed_test:
    t_i = time.time()
    F(np.ones(len(H_c)), 2*np.ones(len(H_c)))
    t_f = time.time()
    print("\nTemps de calcul de F: ", int(100*(t_f-t_i))/100, "s.")

    t_i = time.time()
    F_slow(np.ones(len(H_c)), 2*np.ones(len(H_c)))
    t_f = time.time()
    print("\nTemps de calcul de F slow (without masks): ", int(100*(t_f-t_i))/100, "s.")



##main loop
bool_main_loop = True
save = False
lambda_arr = np.array([n**4-n**2 for n in range(1, M+1)]) 
t_i = time.time()

#1st scheme
if False:
    lambda_diag = np.diag(lambda_arr)
    if bool_main_loop:
        for n_t in range(1, N_t):
            F_c, F_s = F_slow(H_c, H_s)
            H_c = (np.identity(M)-dt*lambda_diag)@H_c + dt*F_c
            H_s = (np.identity(M)-dt*lambda_diag)@H_s + dt*F_s
            U_Fourier_mat[n_t, 1:] = (H_c - (1j)*H_s)/2 #Not taking the first element as it is the mean value supposed to be zero.
#2sn scheme
if True:
    A_lambda = np.diag(np.array([(1+dt)/(1+dt*lambda_arr[i]+dt) for i in range(len(lambda_arr))]))
    B_lambda = np.diag(np.array([dt/(1+dt*lambda_arr[i]+dt) for i in range(len(lambda_arr))]))

    if bool_main_loop:
        for n_t in range(1, N_t):
            F_c, F_s = F(H_c, H_s)
            H_c = A_lambda@H_c + B_lambda@F_c
            H_s = A_lambda@H_s + B_lambda@F_s
            U_Fourier_mat[n_t, 1:] = (H_c - (1j)*H_s)/2 #Not taking the first element as it is the mean value supposed to be zero.

t_f = time.time()

#Some check
computation_time = t_f-t_i 
print("Computation duration: ", int(computation_time*100)/100)
print("\n", U_Fourier_mat[0, 0])
print("Number of terms in the U_Fourier: ", U_Fourier_mat.shape[0]*U_Fourier_mat.shape[1])
print("number of non nan: ", np.count_nonzero(np.isnan(U_Fourier_mat)))
print("Maximum of the Fourier coef:", np.max(np.nan_to_num(U_Fourier_mat)), "\n")

# if save:    
#     np.savetxt('KS_eq_Fourier.txt', U_Fourier_mat, fmt='%d')
# else:
#     U_Fourier_mat = np.loadtxt('KS_eq_Fourier.txt', dtype=complex)

# print("\n", U_Fourier_mat[0, 0])
# print("Number of terms in the U_Fourier: ", U_Fourier_mat.shape[0]*U_Fourier_mat.shape[1])
# print("number of non nan: ", np.count_nonzero(np.isnan(U_Fourier_mat)))
# print("Maximum of the Fourier coef:", np.max(np.nan_to_num(U_Fourier_mat)), "\n")


##Back to the velocity U

for n_t in range(1, N_t):
    U_mat[n_t, 1:] = np.fft.irfft(U_Fourier_mat[n_t, :])



def func_anim(Y, title, gap=0.5):

    # dt = T/N
    # dx = 2*A/J
    T = dt*N_t
    A = dx*N_x/2


    # Initialise les figures et le subplot
    t = np.arange(0, T, dt)
    x = np.arange(-A, A, dx)
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    print(Y.shape)




    # Initialise les lignes à modifier dans l'animation
    line_analytical, = axs.plot([], [], label=title)

    axs.set_xlim([-A, A])
    axs.set_ylim([Y.min()-gap, Y.max()+gap])

    # Met à jour la fonction dans l'animation
    def update(frame):
        t_1 = t[frame]
        y = Y[frame]
        line_analytical.set_data(x, y)
        
        axs.set_title(title + ' en temps {}'.format(t_1))
        
        return line_analytical,

    # Crée l'animation

    animation = FuncAnimation(fig, update, frames=len(t)-1)
    plt.show()
    # plt.close() #to avoid having another plot after the animation

    # return HTML(animation.to_jshtml())

U_anim = np.array([U_mat[i] for i in range(len(U_mat)) if i%3==0])

anim = func_anim(U_mat, "Dynamics of the Velocity")
# FuncAnimation.display(anim)