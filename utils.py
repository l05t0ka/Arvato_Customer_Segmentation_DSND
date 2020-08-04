#Definitions of functions used in Arvato Project Jupyter Notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.impute import SimpleImputer


def change_types(df):
    """ Function changes numerical 64bit columns to their 16bit equivalents in order to preserve memory and stop kernel from dying. 
    INPUT:
        df (pd.DataFrame) - dataframe with 64bit data types
    OUTPUT:
        df2 (pd.DataFrame) - dataframe with 16bit data types

    """
    
    df2 = df.copy()
    
    types_dict = df2.dtypes.to_dict()
    
    for col, col_type in types_dict.items() :
        if col_type == "int64":
            if col == "LNR":
                continue
            df2[col] = df2[col].astype('int16')
        elif col_type == "float64":
            df2[col] = df2[col].astype('float16')

    return df2


def plot_column(col, customers, azdias):
    """Function to plot occurences of values for a given attribute in both dataframes
    INPUT:
        col (String) - a column that exists in both datasets
        customers (pd.DataFrame) - customers dataframe
        azdias (pd.DataFrame) - azdias dataframe

    OUTPUT:
        NONE """
    fig, ax =plt.subplots(1,2 ,figsize=(15,10))
    ax[0].set_title("Customers")
    sns.countplot(col, data=customers, color = "#008039", ax=ax[0])
    ax[1].set_title("General Population")
    sns.countplot(col, data=azdias , color = "#ad8039", ax=ax[1])

    return None
    

def solve_type_and_col_mismatch(df):
    """Function fixes type mismatch in columns CAMEO_DEUG_2015 and CAMEO_INTL_2015 which contained mixed type values, another column OST_WEST_KZ gets recoded to numeric. Additionally columns that are only in customers table are dropped
    INPUT:
        df (pd.DataFrame) - input dataframe
    OUTPUT: 
        df (pd.DataFrame) - processed dataframe"""
    
    mismatched_columns = ['CAMEO_DEUG_2015', 'CAMEO_INTL_2015']
    
    
    df[mismatched_columns] = df[mismatched_columns].replace({'X': np.nan, 'XX':np.nan})
    df[mismatched_columns] = df[mismatched_columns].astype(float)

    
    print("Mismatched columns ", mismatched_columns, "were taken care of.")
    
    
    print("Column OST_WEST_KZ was recoded")
    df = df.replace({'OST_WEST_KZ': {'W':0, 'O':1}})

    
    
    
    #remove additional columns from df_cust that are not in df_azdias
    cols_to_drop = set(['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE'])
    
    if cols_to_drop.issubset(set(df.columns)):
        df.drop(cols_to_drop, axis=1, errors='ignore', inplace=True)
    #cols_to_drop = list(set(df_cust.columns.values) - set(df_azdias.columns.values))  #['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE']
    
    #df_cust.drop(cols_to_drop, axis=1, errors='ignore', inplace=True)
    
        print("Columns ", cols_to_drop, "were removed from customers DataFrame.")
    
    return df


def dict_for_replace(dictionary):
    """Function to modify the dictionary so that it can be used in .replace method of pandas.
    INPUT: 
        dictionary (dict) - dictionary to modify
    OUTPUT:
        dictionary (dict) - processed dictionary"""
    for k,v in dictionary.items():
        if isinstance(v, str):
            v_new = {int(v): np.nan for v in v.split(",")}
            dictionary[k] = v_new
        else:
            v_new = {v : np.nan}
            dictionary[k] = v_new
            
    return dictionary



def all_missings_to_nans(df, dictionary_of_missings):
    """Function converts all values of df specified by the dictionary_of_missings to NAN
    INPUT:
        df (pd.DataFrame) - input dataframe
        dictionary_of_missings (dict) - dictionary containing column names as keys and all values that should be NANs
    OUTPUT: 
        df_n (pd.DataFrame) - processed dataframe"""
    #import numpy as np

    df_n = df.replace(dictionary_of_missings)

        
    return df_n


def summary_of_missings(df, rows = 1):
    """ Function to describe distribution of missing values.
    INPUT:
        df (pd.DataFrame) - input dataframe
        rows (int) - a value either 0 or 1 (rows / columns)
    OUTPUT:
        None
    
    """    
    
    if rows:
        df_perc_missing = df.isnull().sum(axis=1)*100/len(df.columns)
        print("There are",len(df_perc_missing), "rows in total in the dataframe")
        for i in np.arange(0,91,10):
            print("There are", len(df_perc_missing[df_perc_missing>i]) ,"rows with more than", i, "% missings in the dataframe (", np.round(len(df_perc_missing[df_perc_missing>i])/len(df_perc_missing),2) ,")")

    else:
        df_perc_missing = df.isnull().sum(axis=0)*100/len(df)
        print("There are",len(df_perc_missing), "cols in total in the dataframe")
        for i in np.arange(0,91,10):
            print("There are", len(df_perc_missing[df_perc_missing>i]) ,"cols with more than", i, "% missings in the dataframe (", np.round(len(df_perc_missing[df_perc_missing>i])/len(df_perc_missing),2) ,")")

    return None 

def remove_rows_with_too_much_nan(df, perc = 0.5):
    """ Function to drop rows that have more missing ratio than treshold.
    INPUT:
        df (pd.DataFrame) - input dataframe
        perc (float) - a value between 0 and 1 indicating treshold ratio
    OUTPUT:
        df_n (pd.DataFrame)
    """
    
    print("Rows before removing ", len(df))
    
    df_n = df.loc[df.isnull().mean(axis=1).lt(perc)]

    print("New table with", len(df_n), "rows created")          
    return df_n


def drop_cols(df, missing_df, perc=0.3):
    """ Function to drop columns that have more missing ratio than treshold.
    INPUT:
        df (pd.DataFrame) - input dataframe
        missing_df (pd.DataFrame) - dataframe containing info on missing values
        perc (float) - a value between 0 and 1 indicating treshold ratio
    OUTPUT:
        df_n (pd.DataFrame)
    """
    df_n = df.copy()
    # cols_to_remove = missing_df[missing_df.perc_missing_azdias_df > perc].index
    cols_to_remove = missing_df[(missing_df.perc_missing_azdias_df > perc) & (missing_df.missing_mismatch == False)].index
    df_n = df_n.drop(cols_to_remove, axis=1)
    
    return df_n


def impute_missings(df):
    """Function to impute missing values
    INPUT:
        df (pd.DataFrame) - dataframe containing missing values
    OUTPUT:
        df_n (pd.DataFrame) - imputed dataframe 
    """
    
    df_n = df.copy()

    #handle the three object columns
    
    df_n["CAMEO_DEU_2015"] = df_n["CAMEO_DEU_2015"].fillna("0")
    df_n["D19_LETZTER_KAUF_BRANCHE"] = df_n["D19_LETZTER_KAUF_BRANCHE"].fillna("0")
    #most_common for eingefuegt_am because it'll 
    df_n["EINGEFUEGT_AM"] = df_n["EINGEFUEGT_AM"].fillna(df_n.EINGEFUEGT_AM.value_counts().index[0])
    
    #handle rest -> just with 0 - this can be a possible actionpoint to change to imputed strategy
    
#     for i in df_n.columns[df_n.isnull().any(axis=0)]:     
#         df_n[i].fillna(df_n[i].median(),inplace=True)
        
    #df_n = df_n.fillna(-1) 
    
    imputer = SimpleImputer(strategy= 'most_frequent')
    for i in df_n.columns[df_n.isnull().any(axis=0)]:
        df_n[i] = imputer.fit_transform(df_n[[i]])
    
    #df_imputed = pd.DataFrame(imputer.fit_transform(df_n))
    #df_imputed.columns = df_n.columns
    #df_imputed.index = df_n.index
    
    return df_n


def process_and_generate_features(df):
    """Function recodes and creates more features, incl. One-Hot-Encoding for several categorical columns.
    INPUT: 
        df (pd.DataFrame) - customers/azdias table
    OUTPUT:
        df_ohe (pd.DataFrame) - preprocessed customers/azdias table
        """
   
    df_copy = df.copy()
    
    if "LNR" in df_copy.columns:
        df_copy = df_copy.drop(["LNR"], axis=1)

    #delete letzter_kauf_branche and cameo_deu
    df_copy = df_copy.drop(["CAMEO_DEU_2015", "D19_LETZTER_KAUF_BRANCHE"], axis=1)
    

    
    #Adding feature HH_MIT_KIND
    df_copy["HH_MIT_KIND"] = np.where(df_copy["ANZ_KINDER"] >0, 1, 0)
    df_copy["HH_MIT_KIND"] = df_copy["HH_MIT_KIND"].astype('int16')
   
    df_copy["URBAN_WEIT_ENTFERNT"] = (df_copy['BALLRAUM'] > 3)*1
    df_copy["URBAN_WEIT_ENTFERNT"] = df_copy["URBAN_WEIT_ENTFERNT"].astype('int16')

    
    #df["ONLINE_BANKING_AFFIN"] = (df["D19_BANKEN_ONLINE_QUOTE_12"] > 4)*1
    df_copy["DICHTE_S"] = (df_copy["EWDICHTE"] > 3)*2
    df_copy["DICHTE_S"] = df_copy["DICHTE_S"].astype('int16')
    
    
    df_copy["GEBAEUDETYP_PRIVAT"] = ((df_copy["GEBAEUDETYP"] == 1) |(df_copy["GEBAEUDETYP"] == 2)) *1
    df_copy["GEBAEUDETYP_PRIVAT"] = df_copy["GEBAEUDETYP_PRIVAT"].astype('int16')

    
    df_copy["LP_STATUS_GROB_NEU"] = -1
    df_copy.loc[df_copy['LP_STATUS_GROB']==10, 'LP_STATUS_GROB_NEU'] = 5
    df_copy.loc[((df_copy['LP_STATUS_GROB']>=8) & (df_copy['LP_STATUS_GROB']<=9)), 'LP_STATUS_GROB_NEU'] = 4
    df_copy.loc[((df_copy['LP_STATUS_GROB']>=6) & (df_copy['LP_STATUS_GROB']<=7)), 'LP_STATUS_GROB_NEU'] = 3
    df_copy.loc[((df_copy['LP_STATUS_GROB']>=3) & (df_copy['LP_STATUS_GROB']<=5)), 'LP_STATUS_GROB_NEU'] = 2
    df_copy.loc[((df_copy['LP_STATUS_GROB']>=1) & (df_copy['LP_STATUS_GROB']<=2)), 'LP_STATUS_GROB_NEU'] = 1
    df_copy["LP_STATUS_GROB_NEU"] = df_copy["LP_STATUS_GROB_NEU"].astype('int16')
    df_copy = df_copy.drop("LP_STATUS_GROB", axis=1)
    
    
    df_copy["LP_FAMILIE_GROB_NEU"] = 0
    df_copy.loc[df_copy['LP_FAMILIE_GROB']==10, 'LP_FAMILIE_GROB_NEU'] = 6
    
    df_copy.loc[((df_copy['LP_FAMILIE_GROB']>=9)), 'LP_FAMILIE_GROB_NEU'] = 5
    df_copy.loc[((df_copy['LP_FAMILIE_GROB']>=6) & (df_copy['LP_FAMILIE_GROB']<=8)), 'LP_FAMILIE_GROB_NEU'] = 4
    df_copy.loc[((df_copy['LP_FAMILIE_GROB']>=3) & (df_copy['LP_FAMILIE_GROB']<=5)), 'LP_FAMILIE_GROB_NEU'] = 3
    df_copy.loc[(df_copy['LP_FAMILIE_GROB']==2), 'LP_FAMILIE_GROB_NEU'] = 2
    df_copy.loc[(df_copy['LP_FAMILIE_GROB']==1), 'LP_FAMILIE_GROB_NEU'] = 1
    df_copy["LP_FAMILIE_GROB_NEU"] = df_copy["LP_FAMILIE_GROB_NEU"].astype('int16')
    df_copy = df_copy.drop("LP_FAMILIE_GROB", axis=1)
    
    df_copy["NEU_IN_STADT"] = (df_copy['WOHNDAUER_2008'] < 4)*1
    df_copy["NEU_IN_STADT"] = df_copy["NEU_IN_STADT"].astype('int16')

    
    # Extracting year from EINGEFUEGT_AM"
    df_copy["EINGEFUEGT_AM"] = pd.to_datetime(df_copy["EINGEFUEGT_AM"], format='%Y/%m/%d %H:%M')
    df_copy["EINGEFUEGT_AM"] = df_copy["EINGEFUEGT_AM"].dt.year
    df_copy["EINGEFUEGT_AM"] = pd.to_datetime(df_copy["EINGEFUEGT_AM"], format='%Y/%m/%d %H:%M')
    df_copy["EINGEFUEGT_AM"] = df_copy["EINGEFUEGT_AM"].dt.year
    df_copy["EINGEFUEGT_AM"] = df_copy["EINGEFUEGT_AM"].astype('int16')
    # Generating PRAEGENDE_JUGENDJAHRE_MAINSTREAM feature

    mainstream_dict = {-1:-1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0, 8: 1,
           9: 0, 10: 1, 11: 0, 12: 1, 13: 0, 14: 1, 15: 0}
    df_copy['PRAEGENDE_JUGENDJAHRE_MAINSTREAM'] = df_copy['PRAEGENDE_JUGENDJAHRE'].map(mainstream_dict)
    #df_copy["PRAEGENDE_JUGENDJAHRE_MAINSTREAM"] = df_copy["PRAEGENDE_JUGENDJAHRE_MAINSTREAM"]

    
    #1 - young, 2 - middle 3- advanced 4- retirement age
    lebensphase_age_dict = {-1:0, 0:0, 1: 1, 2: 2, 3: 1,
              4: 2, 5: 3, 6: 4,
              7: 3, 8: 4, 9: 2,
              10: 2, 11: 3, 12: 4,
              13: 3, 14: 1, 15: 3,
              16: 3, 17: 2, 18: 1,
              19: 3, 20: 3, 21: 2,
              22: 2, 23: 2, 24: 2,
              25: 2, 26: 2, 27: 2,
              28: 2, 29: 1, 30: 1,
              31: 3, 32: 3, 33: 1,
              34: 1, 35: 1, 36: 3,
              37: 3, 38: 4, 39: 2,
              40: 4}
    
    df_copy['LP_LEBENSPHASE_ALTER'] = df_copy['LP_LEBENSPHASE_FEIN'].map(lebensphase_age_dict).astype('int16')

    #1 - low income, 2 - average, 3 - wealthy, 4-top
    lebensphase_income_dict = {-1:0, 0:0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1,
              7: 2, 8: 2, 9: 3, 10: 3, 11: 3,
              12: 3, 13: 4, 14: 2, 15: 1, 16: 2,
              17: 3, 18: 3, 19: 3, 20: 4, 21: 1,
              22: 2, 23: 3, 24: 1, 25: 2, 26: 3,
              27: 3, 28: 4, 29: 1, 30: 2, 31: 1,
              32: 2, 33: 3, 34: 3, 35: 4, 36: 3,
              37: 3, 38: 3, 39: 4, 40: 4}

    df_copy['LP_LEBENSPHASE_GELD'] = df_copy['LP_LEBENSPHASE_FEIN'].map(lebensphase_income_dict).astype('int16')
    
    df_copy = df_copy.drop(["LP_LEBENSPHASE_FEIN", "LP_LEBENSPHASE_GROB"], axis=1)
    
    df_copy.loc[((df_copy['GEBAEUDETYP']==5)), 'GEBAEUDETYP'] = 3
    df_copy.loc[((df_copy['FINANZTYP']==5)), 'FINANZTYP'] = 4

    #OHE categorics, manually checked with documentation and distribution of values
    
    #WACHSTUMSGEBIET_NB, HAUSHALTSSTRUKTUR have different names in table so leave them out of categorics
    categorics = ["ANREDE_KZ","CAMEO_DEUG_2015", "CAMEO_INTL_2015", "FINANZTYP", "GEBAEUDETYP", "GFK_URLAUBERTYP", "KBA05_MAXHERST",
                    "KBA05_MAXSEG", "PRAEGENDE_JUGENDJAHRE"
                  , "REGIOTYP", "RETOURTYP_BK_S",  "WOHNLAGE", "ZABEOTYP", "LP_STATUS_GROB_NEU"
                ,"LP_FAMILIE_GROB_NEU", "PRAEGENDE_JUGENDJAHRE_MAINSTREAM"]
    
    df_ohe = pd.get_dummies(df_copy, columns=categorics)
    
    return df_ohe
    



def summary_of_missings(df, rows = 1):
    """Function outputs a textual summary of  and creates more features, incl. One-Hot-Encoding for several categorical columns.
    INPUT: 
        df (pd.DataFrame) - customers/azdias table
        rows (int) - if 1 
    OUTPUT:
        None
    """ 
    
    if rows:
        df_perc_missing = df.isnull().sum(axis=1)*100/len(df.columns)
        print("There are",len(df_perc_missing), "rows in total in the dataframe")
        for i in np.arange(0,91,10):
            print("There are", len(df_perc_missing[df_perc_missing>i]) ,"rows with more than", i, "% missings in the dataframe (", np.round(len(df_perc_missing[df_perc_missing>i])/len(df_perc_missing),2) ,")")

    else:
        df_perc_missing = df.isnull().sum(axis=0)*100/len(df)
        print("There are",len(df_perc_missing), "cols in total in the dataframe")
        for i in np.arange(0,91,10):
            print("There are", len(df_perc_missing[df_perc_missing>i]) ,"cols with more than", i, "% missings in the dataframe (", np.round(len(df_perc_missing[df_perc_missing>i])/len(df_perc_missing),2) ,")")

            


def calculate_perc_missings(df_cust, df_azdias, perc_treshold=0.2):
    """Function generates a table containing missing percentages of both dataframes
    INPUT:
        df_cust (pd.DataFrame) - customers dataframe
        df_azdias (pd.DataFrame) - azdias dataframe
        perc_treshold (float) - value between 0 and 1, indicates percent above which column is flagged
    OUTPUT:
        joined (pd.DataFrame) - descriptive dataframe containing info on missing for a column in both datasets.
    """
    
    df_cust_missings = df_cust.isnull().mean()
    df_azdias_missings = df_azdias.isnull().mean()
    
    joined = pd.concat([df_cust_missings, df_azdias_missings], axis=1).reset_index()
    joined.columns = ["Variable", "perc_missing_customer_df", "perc_missing_azdias_df"]
    joined["flag_customer"] = (joined.perc_missing_customer_df >= perc_treshold)
    
    joined["flag_azdias"] = (joined.perc_missing_azdias_df >= perc_treshold)
    
    joined["missing_mismatch"] = joined["flag_customer"] != joined["flag_azdias"]
        
    joined.sort_values(["perc_missing_azdias_df", "perc_missing_customer_df"], ascending=[False, False], inplace=True)
    
    joined = joined.set_index("Variable")
    
    
    return joined
    
    
   
