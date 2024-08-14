import numpy as np
import tensorflow as tf
import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt

from pinn_model import SimplePINN

def gen_traindata():
    data = pd.read_csv("df.csv")
    t_data = data['T'].values[:,None]
    x1_data = data['X1'].values[:,None]
    x2_data = data['X2'].values[:,None]

    return t_data, x1_data, x2_data
    
def equation(x1,x2,a=26.7979, b=30.12,c=8.2565):
    x1_dot = c*(x1 -x1**3/3 + x2)
    x2_dot = (-1/c)*(x1 - a + b*x2)

    return x1_dot, x2_dot

def fit_equation():
    loaded_data = gen_traindata()
    #parameters
    pars = [26.7979,30.12,8.2565]
    
    dt = 0.01
    num_steps = 1000

    # Calculating derivatives
    ts = np.empty((num_steps + 1, 1), dtype=np.float32)
    x1s = np.empty((num_steps + 1, 1), dtype=np.float32)
    x2s = np.empty((num_steps + 1, 1), dtype=np.float32)

    # Set initial values
    ts[0],x1s[0], x2s[0] = 0., -1.00713321, 0.86365752

    # estimating derivates of the next time step
    for i in range(num_steps):
        x1_dot, x2_dot = equation(x1s[i], x2s[i])
        x1s[i + 1] = x1s[i] + (x1_dot * dt)
        x2s[i + 1] = x2s[i] + (x2_dot * dt)
        ts[i + 1] = ts[i] + dt
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ts[:,0], x1s[:,0], lw=0.75)
    plt.savefig('groundtruth.png')
    plt.close()

    pinn = SimplePINN(bn=True, log_opt=True, lr=1e-2, layers=3, layer_width=32)

    for i in range(6):
        pinn.fit(loaded_data, pars, 1000,verbose=True)
        curves = pinn.predict_curves(loaded_data[0])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(curves[0], curves[1], lw=.75)

        plt.savefig(str((i+1)*10000)+" Epochs.png")
        plt.close()
    
    print(
        "Estimated parameters: a = {:3.2f}, b = {:3.2f}, c = {:3.2f}".format(
            np.exp(pinn.a.numpy().item()), np.exp(pinn.b.numpy().item()), np.exp(pinn.c.numpy().item())
        )  
    )
    print('\n')
    print(
        "True parameters: a = {:3.2f}, b = {:3.2f}, c = {:3.2f}".format(
            pars[0], pars[1], pars[2]
        )
    )                    

if __name__ == "__main__":
    fit_equation()
    
   

