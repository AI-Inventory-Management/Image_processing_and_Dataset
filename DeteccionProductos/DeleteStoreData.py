import os
try:
    os.remove("./data/store_data.json")
    os.remove("./data/fridge_data.json")
except:
    print("No files to delete")