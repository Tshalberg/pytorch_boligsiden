import pandas as pd
import numpy as np
import re

def convert_to_numeric(data, numeric_columns):
    clean_numeric = lambda val: re.sub(r"\.", "", str(val))
    
    for num_col in numeric_columns:
        data[num_col] = pd.to_numeric(data[num_col].apply(clean_numeric), errors="coerce")
        
def convert_time_to_numeric(data, time_columns):
    for time_col in time_columns:
        data[time_col] = pd.to_numeric(data[time_col].apply(lambda x: str(x).replace("min.","")), errors="coerce").fillna(-1)
    
def convert_to_categorical(data, categorical_columns):
    for cat_col in categorical_columns:
        data["%s_encoded" % cat_col] = pd.to_numeric(data[cat_col].astype("category").cat.codes) 
        
def create_date_columns(data):
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].apply(lambda d: d.year)
    data["month"] = data["date"].apply(lambda d: d.month)


def clean_data(data):
    create_date_columns(data)

    # remove "min" from time columns and convert to numeric
    time_columns = ['Tog', "S-tog", "Metro", 'Bus', 'Sø', 'Skov', 'Kyst', 
                    'Hospital', 'Fodboldbane', 'Svømmehal', 'Sportshal']
    convert_time_to_numeric(data, time_columns)

    # convert columns to numeric
    numeric_columns = ["price", "sqrm", "lat", "lng", "værelser"]
    convert_to_numeric(data, numeric_columns)

    data["price_sqrm"] = data["price"] / data["sqrm"]

    # encode categorical columns
    categorical_columns = ["city", "floor", "year", "kommune", "zipcode", "Fiber", 'xDSL (Kobbernet)', 
                           'Kabel-tv', 'Jordforurening i nærheden', 'Drikkevandets hårdhed er',
                           'Risikoen for radon i kommunen er']
    convert_to_categorical(data, categorical_columns)

    # convert floor column to make it more general
    data = data[data["floor"] != ""]
    data["floor"] = data["floor"].apply(lambda x: x.split(" ")[1])

    # remove nan values
    data = data.dropna(subset=["værelser"])

    # only include sale_type == "Fri handel"
    data = data[ (data["sale_type"] == "Fri handel")]
    return data


def outliers_pct(x, level):
    lower = np.percentile(x, level)
    upper = np.percentile(x, 100 - level)
    return lower, upper


def get_outlier_mask(data, columns, level):
    outlier_masks = []
    for col in columns:
        x = data[col]
        lower, upper = outliers_pct(x, level)
        outlier_mask = ( data[col] > lower ) & ( data[col] < upper )
        outlier_masks.append(outlier_mask)

    return np.all(np.array(outlier_masks), axis=0)