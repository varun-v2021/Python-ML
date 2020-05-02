import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(8, 4),
                  index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], columns=['A', 'B', 'C', 'D'])

print('>>>>>><<<<<<<<<<<<<<')
training = np.array(df)
print('training data contents')
print(training)
print('-----------all rows of first column----------')
print(training[:, 0])
print('-----------all rows of second column----------')
print(training[:, 1])

# list of data
data = [[11, 22],
        [33, 44],
        [55, 66]]
# array of data
data = np.array(data)
print('Rows: %d' % data.shape[0])
print('Cols: %d' % data.shape[1])
# Rows: 3
# Cols: 2

# data_reshaped = data.reshape(data,(3,2,1))
# print('data_reshaped')
# print(data_reshaped)

data = np.array([11, 22, 33, 44, 55])
print(']]]]]]]]]]')
print(data.shape)
# 1D to 2D
data = data.reshape((data.shape[0], 1))
print(data)
print(data.shape)

# 2D to 3D
# The reshape function can be used directly, specifying the new dimensionality.
# This is clear with an example where each sequence has multiple time steps with
# one observation (feature) at each time step.
# We can use the sizes in the shape attribute on the array to specify the number of samples (rows)
# and columns (time steps) and fix the number of features at 1.
# shape[0] - rows, shape[1] - columns
# data.reshape((data.shape[0], data.shape[1], 1))

print('$$$$$$$$$$$$$$$$$$$$$$$')
data = [[11, 22],
        [33, 44],
        [55, 66]]
# array of data
data = np.array(data)
print(data.shape)
# reshape
data = data.reshape((data.shape[0], data.shape[1], 1))
print(data.shape)

print('::::::::::::::::::::::::::::')
x_data = []
y_data = []
# multiple rows having single column
data = np.array([[11], [22], [33], [44], [55]])
# above line is equivalent to
# data = np.array([[11],
#                [12],
#                [13],
#                [14],
#                [15]])
for i in range(1, 5):
    x_data.append(data[i - 1:i, 0])
    y_data.append(data[i,0])
print(x_data)
print('---------')
print(y_data)
x_data , y_data = np.array(x_data), np.array(y_data)
print(',,,,,,,,,,,,,,,')
print(x_data)
print(y_data)
# data = np.reshape(data,(data.shape[0],data.shape[1],1))
# print(data)
#
# .loc()
#
# Label based

# select all rows for a specific column
print(df.loc[:, 'A'])
print('############')
# Select all rows for multiple columns, say list[]
print(df.loc[:, ['A', 'C']])

# Select few rows for multiple columns, say list[]
print(df.loc[['a', 'b', 'f', 'h'], ['A', 'C']])

# Select range of rows for all columns
print(df.loc['a':'h'])

# for getting values with a boolean array
print(df.loc['a'] > 0)  # row 'a' whose column is > 0

print('***************************************************')
# .iloc()

# Integer based

# Pandas provide various methods in order to get purely integer based indexing.
# Like python and numpy, these are 0-based indexing.

df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])

# select all rows for a specific column
print(df.iloc[:4])
print(df.iloc[1:9, 2:4])  # since there are only 8 rows of random numbers

# Slicing through list of values
print(df.iloc[[1, 3, 5], [1, 3]])
print(df.iloc[1:3, :])
print(df.iloc[:, 1:3])

# Let us now see how each operation can be performed on the DataFrame object.
# We will use the basic indexing operator '[ ]'
print('=========================================')
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
print(df['A'])
print(df[['A', 'B']])
print(df[2:2])

# attribute access
print(df.A)
