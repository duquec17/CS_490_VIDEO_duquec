import numpy as np

fx = 3
fy = 1
ft = 5
ux = 2
uy = 1
vx = 1
vy = -1
u = 3
v = -1

lamba = 2

flow_error = fx*u + fy*v + ft
flow_error *= flow_error

bright_error = lamba*(ux*ux + uy*uy + vx*vx + vy*vy)
error = flow_error + bright_error
print(error)

values = np.array([[3,2],
                   [5,3],
                   [5,2],
                   [4,2],
                   [3,0],
                   [4,3]], dtype="float64")

ave = np.mean(values, axis=0)
print(ave)

mean_sub = values - ave
x_sub = mean_sub[:,0]
y_sub = mean_sub[:,1]

var_x = np.mean(x_sub*x_sub)
var_y = np.mean(y_sub*y_sub)
covar = np.mean(x_sub*y_sub)

print(var_x)
print(var_y)
print(covar)

a = var_x
d = var_y
b = covar

import math as m

under = (a+d)
under *= under
under = under - 4*(a*d - b*b)
under = m.sqrt(under)

eigenval = ((a+d) + under)/2
print(eigenval)

y = -(a - eigenval)/b
print(y)

points = np.array([[3,2],
                   [5,3],
                   [5,2],
                   [4,2],
                   [3,0],
                   [4,3]])
values = np.array([2,1,1,2,5,2])
sum_weights = np.sum(values)
values = np.expand_dims(values, axis=-1)

points = points*values
print(points)
point_sum = np.sum(points, axis=0)
point_sum = point_sum.astype("float64")
point_sum /= float(sum_weights)
print(point_sum)

fx = np.array([3,-4,1,2], dtype="float64")
fy = np.array([5,0,-2,-2], dtype="float64")
ft = np.array([0,2,3,1], dtype="float64")

sfx_2 = np.sum(fx*fx)
sfy_2 = np.sum(fy*fy)
sft_2 = np.sum(ft*ft)

sfx_fy = np.sum(fx*fy)
sfy_ft = np.sum(fy*ft)
sfx_ft = np.sum(fx*ft)

denom = sfx_2*sfy_2 - (sfx_fy*sfx_fy)

u = (sfy_2*sfx_ft - sfx_fy*sfy_ft)/denom
v = (sfx_2*sfy_ft - sfx_fy*sfx_ft)/denom

print(u)
print(v)








                  