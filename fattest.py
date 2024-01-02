import numpy as np

def MeanSquaredDisplacement1D(position):
    squared_position = np.square(position)
    msd = np.empty(len(position))
    for i in range(1, 1 + len(position)):
        msd[i-1] = np.mean(squared_position[:i])
    return msd


def MeanSquaredDisplacement2D(position_x, position_y):
    squared_position_x = np.square(position_x)
    squared_position_y = np.square(position_y)
    r = np.sqrt(squared_position_x + squared_position_y)
    msd = np.empty(len(position_x))

    for i in range(1, 1 + len(position_x)):
        msd[i-1] = np.mean(r[:i])
    return msd



##########################
### Uncomment for test ###
##########################
test_pos_x = [1, 4, 3, 7, 1]
test_pos_y = [8, 4, 9, 12, 0]

print(MeanSquaredDisplacement1D(test_pos_x))    
print(MeanSquaredDisplacement2D(test_pos_x, test_pos_y))
