This is the folder for the solving of the Benney equation with air jet control.  

Convention with the name of the files:
- method: (``FD``: Finite Difference; ``Sp``: Spectral Method)
- ``Verif`` or ``Ctrl`` : Verification of Control 
- Backward differentiation scheme: ``BDF`` + the order of the BDF scheme (i.e "BDF3")
- Type of Control: ``prop`` : Proportional control; ``LQR``: LQR Control;
  ``pospart``: if we just the positive part of the control. We specify the parameter $\zeta_+$ just after: e.g ``Ctrl_proppospart1_Sp_BDF4_Nx512_alpha6.84.txt`` has $\zeta_+ = 1$.
- Number of space point ``N_x``, LQR parameter $\beta$ (``beta``), proportional control parameter $\alpha$ (``alpha``):
same as the BDF order. E.g : ``Anim_Ctrl_prop_Sp_BDF4_Nx512_alpha6.84.mp4``

The outline is as follows
- ``Schemes_verification`` :
    - ``Simulation_values``: .txt files with the values of the solutions.
    - ``Comparison_FD_Sp``: File of the comparison between the Finite Difference and Spectral method
    - ``Pseudo_convergence``: File with the plots of the convergence study of the FD and Sp methods. (watch out, the $L^2$ was not scaled with $dx^2$ in these ones compared to the report)
    - ``BDF_order_test``: Videos and plot of the last time of the dynamics of the numerical solutions with the 6 BDF orders in the same times. 
- ``Control_verifications``: Folder with all the experiments with LQR (``LQR`` folder) and proportional (``prop_ctrl`` folder) controls.
  - ``pos_part`` folder : control where we only took its positive part.
  -  ``normal`` folder : Control with no sign restrictions.
  -  ``Cost_dampening_rate_beta.txt`` and  ``Prop_ctrl_cost_dampening_rate_alpha.txt``: notes of the cost $\kappa_u$ and $\max_{\substack{1\leq i \leq N_t \\ 1\leq j \leq k}}|u_j^i|$ (cf arrays part V of ``Bilal_BM_report.pdf``)
