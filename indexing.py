import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(8, 4),
index = ['a','b','c','d','e','f','g','h'], columns = ['A', 'B', 'C', 'D'])

print('>>>>>><<<<<<<<<<<<<<')
training = np.array(df)
print('training data contents')
print(training)
print('-----------all rows of first column----------')
print(training[:,0])
print('-----------all rows of second column----------')
print(training[:,1])

#
# .loc()
#
# Label based

#select all rows for a specific column
print (df.loc[:,'A'])
print('############')
# Select all rows for multiple columns, say list[]
print (df.loc[:,['A','C']])

# Select few rows for multiple columns, say list[]
print (df.loc[['a','b','f','h'],['A','C']])

# Select range of rows for all columns
print (df.loc['a':'h'])

# for getting values with a boolean array
print (df.loc['a']>0) # row 'a' whose column is > 0

print('***************************************************')
# .iloc()

# Integer based

# Pandas provide various methods in order to get purely integer based indexing.
# Like python and numpy, these are 0-based indexing.

df = pd.DataFrame(np.random.randn(8, 4), columns = ['A', 'B', 'C', 'D'])

# select all rows for a specific column
print (df.iloc[:4])
print (df.iloc[1:9, 2:4]) # since there are only 8 rows of random numbers

# Slicing through list of values
print (df.iloc[[1, 3, 5], [1, 3]])
print (df.iloc[1:3, :])
print (df.iloc[:,1:3])

# Let us now see how each operation can be performed on the DataFrame object.
# We will use the basic indexing operator '[ ]'
print('=========================================')
df = pd.DataFrame(np.random.randn(8, 4), columns = ['A', 'B', 'C', 'D'])
print (df['A'])
print (df[['A','B']])
print (df[2:2])

#attribute access
print (df.A)
