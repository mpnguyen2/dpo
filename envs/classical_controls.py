### Continuous version of classical controls.
### Environment setup is based on OpenAI Gym

import numpy as np
import math
from envs.continuous_env import ContinuousEnv

#### CartPole for PMP ####
class CartPole(ContinuousEnv):
    def __init__(self, q_dim=4, u_dim=1, num_comp=1):
        super().__init__(q_dim, num_comp)
        self.u_dim = u_dim
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.viewer = None

        # For continous version
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4   
        
    def f(self, q, u):
        _, x_dot, theta, theta_dot = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        force = self.force_mag * u.reshape(-1)
        costheta, sintheta = np.cos(theta), np.sin(theta)
        temp = (
            force + self.polemass_length * (theta_dot ** 2) * sintheta
        ) / self.total_mass
        
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        thetaacc, xacc = thetaacc.reshape(-1, 1), xacc.reshape(-1, 1), 
    
        return np.concatenate((x_dot.reshape(-1, 1), xacc, theta_dot.reshape(-1, 1), thetaacc), axis=1)
    
    def f_u(self, q):
        theta = q[:, 2]
        N = q.shape[0]
        costheta = np.cos(theta)
        tmp_u = self.force_mag /self.total_mass
        xacc_u = tmp_u * np.ones((N, 1))
        thetaacc_u = -costheta*tmp_u/(
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        thetaacc_u = thetaacc_u.reshape(-1, 1)
        return np.concatenate((np.zeros((N, 1)), xacc_u, np.zeros((N, 1)), thetaacc_u), axis=1)\
            .reshape(-1, self.q_dim, self.u_dim)
    
    def obj(self, q):
        #noise = np.random.normal(scale=0.001, size=(q.shape[0]))
        #t = [self.x_threshold/2, self.theta_threshold_radians/2]
        #a = 0.005
        return (q[:, 2]/self.theta_threshold_radians)**2 #(a**2-q[:, 0]**2)
    
    def sample_q(self, num_examples, mode='train'):
        if mode == 'train':
            a = 0.01
        else:
            a = 0.05
        return np.random.uniform(low=-a, high=a, size=(num_examples, 4))
    
    def render(self, q):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        cartx = q[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-q[2])

        return self.viewer.render(return_rgb_array=True)
    
    # Close rendering
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi