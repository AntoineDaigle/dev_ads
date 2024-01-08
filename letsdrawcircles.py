import matplotlib.pyplot as plt
import numpy as np

x0,y0 = (2,4)
x1,y1 = (6,6)
radius = 5

if np.abs(x1**2 + y1**2) > radius:
    r = np.sqrt((x1-x0)**2+(y1-y0)**2)
    angle = np.arctan((y1-y0)/(x1-x0))
    temp_x, temp_y = np.linspace(x0,x1), np.linspace(y0,y1)
    radiuses = np.sqrt(temp_x**2+temp_y**2)

    for j in range(len(radiuses)):
        rad = radiuses[j]
        if rad>radius:
            j -=1
            break

    tempx,tempy = temp_x[j], temp_y[j]
    initial_to_wall = np.sqrt((tempy - y0)**2 + (tempx - x0)**2)
    final_dist_remaining = r - initial_to_wall
    #conditions selon le cadran
    
    phi = np.arctan(tempy/tempx)%(2*np.pi)
    if tempy>0 and tempx>0:
        print('quadrant 1')
        new_angle = 2*phi - angle + np.pi
    if tempy>0 and tempx<0:
        print('quadrant 2')
        new_angle = 2*phi - angle
    if tempy<0 and tempx<0:
        print('quadrant 3')
        new_angle = 2*phi - angle
    if tempy<0 and tempx>0:
        print('quadrant 4')
        new_angle = 2*phi - angle + np.pi
    new_x = tempx + final_dist_remaining*np.cos(new_angle)
    new_y = tempy + final_dist_remaining*np.sin(new_angle)

rs = np.linspace(0,r+1)
fig, ax = plt.subplots()
ax.plot([x0,tempx,new_x],[y0,tempy,new_y])
# ax.plot(rs*np.cos(phi),rs*np.sin(phi))
ax.add_patch(plt.Circle((0,0),radius,fill=False,color = 'k'))

plt.show()  