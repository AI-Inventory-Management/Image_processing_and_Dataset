import os
try:
    os.remove( os.path.join(os.path.dirname(__file__), "./data/store_data.json")) 
    os.remove( os.path.join(os.path.dirname(__file__),"./data/fridge_data.json"))
except:
    print("No files to delete")