import pdb
import time
import os
import pandas as pd
from data_module import model_data as md, curation

t1 = time.time()
data_merged = md.fetch_merged_data_mongo()
t2 = time.time()
print (t2 - t1)

if not os.path.exists("data"):
    os.mkdir("data")


data_merged.drop(["_id_x", "_id_y"], axis=1, inplace=True)
data_merged.to_parquet("data/data_raw.parquet")

t1 = time.time()
data_parquet = pd.read_parquet("data/data_raw.parquet")
t2 = time.time()
print (t2 - t1)

t1 = time.time()
data_curated = curation.clean_data(data_parquet)
t2 = time.time()
print (t2-t1)

data_curated.to_parquet("data/data_curated.parquet")

# breakpoint()