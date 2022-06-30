import numpy as np
import math

"""
 ### math for calculating the steering angle from the lane points is below::
    #### 1. calculate x_dot, x_dot_dot, y_dot, y_dot_dot
    #### 2. calcualte delta vector what will be the eucleadian distance between the two points
    #### 3. calculate the cum sum of the delta vector (S)
    #### 4. calculate K vector which represents values from lane curvature
    #### 5. Fit a function throught the values of the K vector
    #### 6. Calculate the steering angle from the fitted function === tan_inv(K(S) * W)
 """

def steering_angle_from_points(points):

   points_x = points[0,:]
   points_y = points[1,:]

   x_dot = np.ediff1d(points_x)
   x_dot_dot = np.ediff1d(x_dot)

   y_dot = np.ediff1d(points_y)
   y_dot_dot = np.ediff1d(y_dot)

   # #delta vectir ecledian distance between x and y points and store them into a vector 
   delta_vector = np.sqrt(np.square(x_dot) + np.square(y_dot))

   #cum sum of the delta vector
   S = np.cumsum(delta_vector) ## (N-2) vector

   #calculate K vector which represents values from lane curvature
   K = (y_dot_dot * x_dot[:-1] - x_dot_dot * y_dot[:-1]) / np.power((np.square(x_dot[:-1]) + np.square(y_dot[:-1])), 1.5) #(N-2) vector

   wheel_base = 1.5 #meters
   print(S.shape, K.shape)
   function = np.polyfit(S[:-1], K, 2)
   K_S = func(function, S[0])

   st_angle = math.degrees(math.atan(K_S * wheel_base))

   return st_angle

def func(f,x):
   y = f[0]*x **2 + f[1]* x + f[2]
   return y 


   


