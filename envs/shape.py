import numpy as np
import cv2

from shapely.geometry import Polygon
from scipy.interpolate import CubicSpline
from scipy import interpolate

from envs.continuous_env import ContinuousEnv   

from collections import namedtuple
ImgDim = namedtuple('ImgDim', 'width height')
EPS = 1e-18

# Environment for optimizing shape with boundary parametrization.
class ShapeBoundary(ContinuousEnv):
    def __init__(self, q_dim=16):
        super().__init__(q_dim, num_comp=2)
        self.num_coef = q_dim//2
        self.ts = np.linspace(0, 1, 80)
        self.viewer = None
    
    # Get objective components including area and length of shapes.
    def obj_comps(self, q):
        ans = np.zeros((q.shape[0], 2))
        for i in range(q.shape[0]):
            cs = CubicSpline(np.linspace(0,1,self.num_coef), q[i].reshape(2, self.num_coef).T)
            coords = cs(self.ts)
            polygon = Polygon(zip(coords[:,0], coords[:,1]))
            ans[i, 0] = polygon.length
            ans[i, 1] = polygon.area
        return ans

    # Get objective peri-area ratio
    def obj(self, q):
        comps = self.obj_comps(q)
        peris = comps[:, 0]; areas = comps[:, 1] 
        ret = np.zeros(q.shape[0])
        for i in range(q.shape[0]):
            ret[i] = peris[i]/(np.sqrt(areas[i]) + EPS)
        
        return ret
          
    def sample_q(self, num_examples, mode='random'):
        qs = np.zeros((num_examples, self.q_dim))
        q = np.zeros(self.q_dim)
        for i in range(num_examples):
            if mode == 'ellipse':
                t = np.arange(self.num_coef)/self.num_coef
                q[:self.num_coef] = 0.1*np.sin(2*np.pi*t)
                q[self.num_coef:] = np.cos(2*np.pi*t)
            elif mode == 'square':
                # Assume n%4 == 0
                n = self.num_coef//4
                # x-coord
                q[:n] = np.arange(n)/n
                q[n:2*n] = 1
                q[2*n:3*n] = 1 - (np.arange(n)/n)
                q[3*n:4*n] = 0
                # y-coord
                q[4*n:5*n] = 0
                q[5*n:6*n] = np.arange(n)/n
                q[6*n:7*n] = 1
                q[7*n:8*n] = 1 - (np.arange(n)/n)
            elif mode == 'structured_random':
                # Assume n%2 == 0
                n = self.num_coef//2
                # x-coord
                q[0:n] = 0.8*self.np_random.rand(n) + 0.2
                q[n:2*n] = -0.8*self.np_random.rand(n) - 0.2
                # y-coord
                q[2*n:3*n] = np.arange(n)/n
                q[3*n:4*n] = np.arange(n)/n
            elif mode == 'random':
                q = np.random.rand(self.q_dim) - 0.5
            qs[i, :] = q
          
        return qs
        
    def render(self, q):
        screen_width = 600
        screen_height = 600
        eps = 1e-5
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
        
        # Use cubic spline to smooth out the new state parametric curve
        cs = CubicSpline(np.linspace(0,1,self.num_coef), q.reshape(2, self.num_coef).T)
        coords = cs(self.ts)
        coords = coords/(np.max(np.abs(coords))+eps)*100 + 300
        verts = zip(coords[:,0], coords[:,1])
        
        self.viewer.draw_polygon(list(verts))

        if q is None:
            return None

        return self.viewer.render(return_rgb_array=True)
     
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# Environment for optimizing general shape (can be represented by its level-set fct).
class Shape(ContinuousEnv):
    def __init__(self, q_dim=64):
        super().__init__(q_dim, num_comp=2)
        self.xk, self.yk = np.mgrid[-1:1:8j, -1:1:8j]
        self.xg, self.yg = np.mgrid[-1:1:50j, -1:1:50j]
        self.viewer = ImgDim(width=self.xg.shape[0], height=self.yg.shape[1])
    
    # Get area and length
    def obj_comps(self, q):
        ans = np.zeros((q.shape[0], 2))
        for i in range(q.shape[0]):
            peri, area = isoperi_info(q[i], self.xk, self.yk, self.xg, self.yg)
            ans[i, 0] = peri; ans[i, 1] = area
        return ans

    # Get objective peri-area ratio
    def obj(self, q):
        comps = self.obj_comps(q)
        peris = comps[:, 0]; areas = comps[:, 1] 
        ret = np.zeros(q.shape[0])
        for i in range(q.shape[0]):
            ret[i] = peris[i]/(np.sqrt(areas[i]) + EPS)
        
        return ret
      
    def sample_q(self, num_examples, mode='random'):
        return generate_coords(dim=self.q_dim, num_samples=num_examples, shape=mode)
        
    def render(self, q):
        xk, yk, xg, yg = self.xk, self.yk, self.xg, self.yg
        return 255-spline_interp(q.reshape(xk.shape[0], yk.shape[0]), xk, yk, xg, yg)
    
    def close(self):
        if self.viewer:
            self.viewer = None

## Helper functions ##
# Spline interpolation for 2D density problem
def spline_interp(z, xk, yk, xg, yg):
    # Interpolate knots with bicubic spline
    tck = interpolate.bisplrep(xk, yk, z)
    # Evaluate bicubic spline on (fixed) grid
    zint = interpolate.bisplev(xg[:,0], yg[0,:], tck)
    # zint is between [-1, 1]
    zint = np.clip(zint, -1, 1)
    # Convert spline values to binary image
    C = 255/2; thresh = C
    img = np.array(zint*C+C).astype('uint8')
    # Thresholding give binary image, which gives better contour
    _, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return thresh_img

def generate_coords(dim=64, num_samples=1024, shape='random'):
    width = int(np.sqrt(dim))
    qs = np.ones((num_samples, width, width))
    if shape=='hole':
        qs[:, 2:6, 2:6] = -1
        #qs[:, :, 7] = -1
        #qs[:, 1:2, 1:width-1] = np.repeat(np.ones((1, 1, width-2)), num_samples, axis=0)
        # qs[:, 2:3, width//2] = 1
        #qs = 1 - qs
        #q[width//2, width//2] = 0
    elif shape=='random':
        qs = np.random.rand(num_samples, width, width)
    else:
        # Random with zero padding
        qs[:, 1:(width-1), :(width-1)] = np.random.rand(num_samples, width-2, width-1)
    qs -= .5
    return qs.reshape(num_samples, width*width)

def isoperi_info_from_img(img):
    # Extract contours and calculate perimeter/area   
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    peri = 0; area = 0
    for cnt in contours:
        area -= cv2.contourArea(cnt, oriented=True)
        peri += cv2.arcLength(cnt, closed=True)

    return peri, area

def isoperi_info(z, xk, yk, xg, yg):
    img = spline_interp(z, xk, yk, xg, yg)
    return isoperi_info_from_img(img)