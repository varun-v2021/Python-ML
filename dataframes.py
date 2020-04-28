# import pandas as pd
import pandas as pd

# list of strings
lst = ['Geeks', 'For', 'Geeks', 'is',
       'portal', 'for', 'Geeks']
# dtype=str,index=['input']
# Calling DataFrame constructor on list
df = pd.DataFrame(lst)
print(df)