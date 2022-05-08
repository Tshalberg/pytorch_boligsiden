import pymongo
import pandas as pd
from statics import statics

STATICS = statics.StaticVars()

def mongodb_client():
    """Get client for local MongoDB

    Returns:
        pymongo.MongoClient: Client for local MongoDB
    """
    client = pymongo.MongoClient(STATICS.MongoDBHost)
    return client


def fetch_all_data_mongodb(client: pymongo.MongoClient, database: str, collection: str) -> pd.DataFrame:

    """Fetch data from mongodb client, for a specific database and belonging collection

    Returns:
        pd.DataFrame: Data from with all data from collection
    """
    
    db = client[database]
    col = db[collection]
    data = col.find()
    df = pd.DataFrame(list(data))

    return df



