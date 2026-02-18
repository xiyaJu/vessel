import time
import matplotlib.pyplot as plt
from .rendering import animate_ship, animate_static_obstacle

# Function to animate ship and obstacles

def animate_step(x, y, psi, LOA_own, BOL_own, CPA_own, Xob, Yob, psiob, LOA_ob, BOL_ob, CPA_ob, Risk, u_ob, step, l):
    
    
    if step % 100 == 0:
        animate_ship(x, y, psi, LOA_own * 3, BOL_own * 3, CPA_own, [0.41, 0, 0.41])
        #plt.xlabel(r'$X$ (nmi)', fontsize=20)  # Set x-axis label
        #plt.ylabel(r'$Y$ (nmi)', fontsize=20)  # Set y-axis label
        #plt.xlim(15, 20)
        #plt.ylim(-5, 2)
        plt.draw()  # Update the plot
        plt.pause(0.1)  


    # Uncomment this section if you want to handle obstacles
    if step % 100 == 0:
        for j in range(len(u_ob)):
            obs_col = [0.0, 0.7, 0.0]
            if Risk[j] > 0.75:
                obs_col = [1.0, 0.0, 0.0]
            elif Risk[j] > 0.6:
                obs_col = [1.0, 0.6, 0.0]
            elif Risk[j] > 0.35:
                obs_col = [1.0, 0.9, 0.0]

            x_ob = Xob[step, j]
            y_ob = Yob[step, j]

            print("j:", j, "u_ob[j]:", u_ob[j])
            if u_ob[j] > 0.5:
                if step % 100 == 0:
                    # Define colors for ships
                    colors = [
                        [0, 0, 1],  # Blue
                        [1, 0.5, 0],  # Orange
                        [0, 1, 0]   # Green
                    ]

                    animate_ship(x_ob, y_ob, psiob[j], LOA_ob[j] * 3, BOL_ob[j] * 3, CPA_ob[j], colors[j])
                    
                    plt.draw()  # Update the plot
                    plt.pause(0.1)  

            else:
                animate_static_obstacle(x_ob, y_ob, CPA_ob[j], obs_col)
