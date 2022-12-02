"""
Delete store data.

    Delete store information to clear as if new.

Author:
    Alejandro Dominguez
    
"""
#_________________________________Libraries____________________________________
import os

#____________________________________Main______________________________________
try:
    os.remove( os.path.join(os.path.dirname(__file__), "./data/store_data.json")) 
    os.remove( os.path.join(os.path.dirname(__file__),"./data/fridge_data.json"))
except:
    print("No files to delete")