import pandas as pd
import numpy as np

type1_files = ["Amazon.csv", "Microsoft_Corporation.csv", "Apple.csv", "Netflix.csv", "Tesla_Inc.csv"]
type2_files = ["AMZN.csv", "MSFT.csv", "AAPL.csv", "NFLX.csv", "TSLA.csv"]

for type1_file, type2_file in zip(type1_files, type2_files):
    df_type1 = pd.read_csv("path_to_news_nlp_dir" + type1_file)
    df_type2 = pd.read_csv("path_to_stocks_dir" + type2_file)

    df_type1['Day'] = pd.to_datetime(df_type1['Day'])
    df_type2['Date'] = pd.to_datetime(df_type2['Date'])
    df_type1.rename(columns={'Day': 'Date'}, inplace=True)

    numeric_columns = df_type1.select_dtypes(include=[np.number])
    numeric_columns_with_date = pd.concat([df_type1['Date'], numeric_columns], axis=1)

    averages = numeric_columns_with_date.groupby('Date').mean().reset_index()

    merged_df = pd.merge(df_type2, averages, on='Date', how='left')

    # Fill missing values with zeros
    merged_df.fillna(0, inplace=True)

    # Standardize the last three columns
    for column in merged_df.columns[-3:]:
        min_val = merged_df[column].min()
        max_val = merged_df[column].max()
        merged_df[column] = (merged_df[column] - min_val) / (max_val - min_val)

    merged_df.to_csv("target_location_path" + f"merged_{type1_file[:-4]}_{type2_file}", index=False)
