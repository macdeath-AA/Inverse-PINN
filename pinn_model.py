import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from decimal import Decimal
from dense_encoder import DenseBlock

class SimplePINN(tf.keras.Model):
    def __init__(self, layers=4, layer_width=20, bn = False, log_opt=False, lr = 1e-2):

        super(SimplePINN,self).__init__()

        self.a = tf.Variable(20.000)
        self.b = tf.Variable(25.000)
        self.c = tf.Variable(4.000)

        self.NN = DenseBlock(layers, layer_width, bn)
        self.epochs = 0
        self.log_opt = log_opt
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)


    @tf.function
    def call(self,t):

        vars = self.predict()

        with tf.GradientTape(persistent=True) as g:
            g.watch(t)
            [x1,x2] = self.NN(t)
            print(self.NN(t))

        #gradients wrt t
        dx1_dt = g.gradient(x1,t)
        dx2_dt = g.gradient(x2,t)

        #deleting tape
        del g

        #working physical equations
        fx1 = dx1_dt - vars[2]*(x1 - x1**3/3 + x2)
        fx2 = dx2_dt + (1/vars[2]) * (x1 - vars[0] + vars[1]*x2)

        return [x1,x2,fx1,fx2]
    
    def set_lr(self, lr):
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)
    
    def get_loss(self, t,x1dat, x2dat):
        return self.__mse([x1dat,x2dat], self(t))
    
    def get_error(self, true):
        pred = tf.squeeze(self.predict())
        true = tf.convert_to_tensor(true)
        return tf.reduce_sum(tf.abs(pred - true))
    
    def predict(self):
        var_tensor_mu = tf.convert_to_tensor([self.a, self.b,self.c])
        exp_var_tensor_mu = tf.exp(var_tensor_mu) if self.log_opt else var_tensor_mu

        return exp_var_tensor_mu
    
    def predict_curves(self,t):
        return self.NN(t)
    
    def optimize(self,t,x1dat,x2dat):
        loss = lambda: self.get_loss(t, x1dat,x2dat)
        self.optimizer.minimize(loss=loss, var_list=self.trainable_weights)
    
    def save_model(self):
        self.save_weights("model_weights/{}.tf".format(self.string))

    def fit(self, observed_data, true_pars, epochs, verbose=False):

        if len(observed_data) != 3:
            raise ValueError(f"Expected 3 items in observed_data, got {len(observed_data)}")
        
        t_data, x1_data, x2_data = observed_data
        for ep in range(1000 ):
            self.optimize(t_data, x1_data, x2_data )

            if ep%100 ==0 or ep ==1:
                loss= self.get_loss(t_data, x1_data, x2_data )/ t_data.shape[0]
                error = self.get_error(true_pars)        
                curves = self.predict_curves(t_data)

                if verbose:
                    print('\n')
                    print(
                        "Epoch: {:5d}, loss: {:.2E}, error: {:3.2f}".format(
                            ep, Decimal(loss.numpy().item()), error.numpy().item()
                        )                    
                    )

                    print(
                        "a: {:3.2f}, b: {:3.2f}, c: {:3.2f}".format(
                            np.exp(self.a.numpy().item()), np.exp(self.b.numpy().item()), np.exp(self.c.numpy().item())
                        )                    
                    )
        self.epochs += epochs
        self.save_model()
    
    def __mse(self, x1dat,x2dat, y_pred):
        x1_pred, x2_pred, fx1_pred, fx2_pred = y_pred
        loss_x1 = tf.reduce_mean(tf.square(x1dat - x1_pred))
        loss_x2 = tf.reduce_mean(tf.square(x2dat - x2_pred))

        loss_fx1 = tf.reduce_mean(tf.square(fx1_pred))
        loss_fx2 = tf.reduce_mean(tf.square(fx2_pred))

        return 10 * (loss_x1 + loss_x2) + loss_fx1 + loss_fx2


                
            








    





        
        

