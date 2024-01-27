import pandas as pd
import numpy as np

# List of filenames for two types of data files
type1_files = ["Amazon.csv", "Microsoft_Corporation.csv", "Apple.csv", "Netflix.csv", "Tesla_Inc.csv"]
type2_files = ["AMZN.csv", "MSFT.csv", "AAPL.csv", "NFLX.csv", "TSLA.csv"]

# Loop through each pair of files
for type1_file, type2_file in zip(type1_files, type2_files):
    # Read the first type of file (e.g., news data)
    df_type1 = pd.read_csv("path_to_news_nlp_dir" + type1_file)
    # Read the second type of file (e.g., stock data)
    df_type2 = pd.read_csv("path_to_stocks_dir" + type2_file)

    # Convert the 'Day' column in df_type1 and 'Date' column in df_type2 to datetime
    df_type1['Day'] = pd.to_datetime(df_type1['Day'])
    df_type2['Date'] = pd.to_datetime(df_type2['Date'])

    # Rename 'Day' column in df_type1 to 'Date' for consistency
    df_type1.rename(columns={'Day': 'Date'}, inplace=True)

    # Select numeric columns from df_type1
    numeric_columns = df_type1.select_dtypes(include=[np.number])
    # Concatenate the 'Date' column with the numeric columns
    numeric_columns_with_date = pd.concat([df_type1['Date'], numeric_columns], axis=1)

    # Calculate the mean of numeric columns grouped by 'Date'
    averages = numeric_columns_with_date.groupby('Date').mean().reset_index()

    # Merge df_type2 with the calculated averages on 'Date', using a left join
    merged_df = pd.merge(df_type2, averages, on='Date', how='left')

    # Fill any missing values in merged_df with zeros
    merged_df.fillna(0, inplace=True)

    # Standardize the last three columns of merged_df
    for column in merged_df.columns[-3:]:
        # Calculate the minimum and maximum values for each column
        min_val = merged_df[column].min()
        max_val = merged_df[column].max()
        # Apply standardization formula (value - min) / (max - min)
        merged_df[column] = (merged_df[column] - min_val) / (max_val - min_val)

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv("target_location_path" + f"merged_{type1_file[:-4]}_{type2_file}", index=False)
