from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import SepsisValidator, SepsisImputer
import pandas as pd
import argparse
import os 

from transformers import SepsisCleaner, VarianceSelector

def build_sepsis_pipeline():
    return Pipeline([
        ('validator', SepsisValidator()),
        ('imputer', SepsisImputer()),
        ('selector', VarianceSelector()),
        ('scaler', StandardScaler())
    ])
    # return Pipeline([
    #     ('cleaner', SepsisCleaner()),
    #     ('selector', VarianceSelector()),
    #     ('validator', SepsisValidator()),
    #     ('imputer', SepsisImputer()),
    #     ('scaler', StandardScaler())
    # ])
    # return Pipeline([
        
    #     ('cleaner', SepsisCleaner()),      # Handles NaNs
    #     ('selector', VarianceSelector()),  # Drops constant columns (σ=0)
    #     ('validator', SepsisValidator()),
    #     ('imputer', SepsisImputer()),
    #     ('scaler', StandardScaler())       # Scales the remaining features
    # ])

# def build_preprocessing_pipeline():
#     return Pipeline([
#         ('validator', SepsisValidator()),
#         ('imputer', SepsisImputer()),
#         ('scaler', StandardScaler())
#     ])

###############################################
# if __name__ == "__main__":
#     # 1. Setup Argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", type=str, required=True)
#     parser.add_argument("--output", type=str, required=True)
#     args = parser.parse_args()

#     # 2. Load Data
#     print(f"Reading from: {args.input}")
#     df = pd.read_csv(args.input, sep="|")

#     # 3. Build and Configure Pipeline
#     pipeline = build_sepsis_pipeline()
    
#     # MAGIC FIX: This forces the pipeline to return a Pandas DataFrame 
#     # instead of a NumPy array, preserving column names automatically.
#     pipeline.set_output(transform="pandas")
    
#     # 4. Transform Data
#     # Now 'processed_df' is a DataFrame with names like HR, Temp, etc.
#     processed_df = pipeline.fit_transform(df)

#     # FIX 1: Drop columns that are STILL empty after the pipeline
#     # processed_df = processed_df.dropna(axis=1, how='all')

#     # # FIX 2: Replace those weird tiny scientific notation numbers with actual 0
#     # # Anything smaller than 1e-10 is effectively noise
#     # processed_df = processed_df.mask(processed_df.abs() < 1e-10, 0)

#     # # FIX 3: Catch-all for any NaNs created by division-by-zero in the Scaler
#     # processed_df = processed_df.fillna(0)

#     # # FIX 4: Explicitly cast to float32 (PyTorch's preferred precision)
#     # processed_df = processed_df.astype('float32')
#     # # Make sure SepsisLabel stays as an integer/float 0.0 or 1.0
#     # processed_df['SepsisLabel'] = processed_df['SepsisLabel'].astype(float)
    
#     # 5. Handle the Label
#     # If your pipeline dropped the label or scaled it (which you usually don't want),
#     # ensure it's in the final output for the training script to find.
#     if 'SepsisLabel' not in processed_df.columns:
#         processed_df['SepsisLabel'] = df['SepsisLabel'].values

#     # 6. Save
#     print(f"Attempting to write to: {os.path.abspath(args.output)}")
#     processed_df.to_csv(args.output, index=False)
    
#     print(f"Success! Columns preserved: {list(processed_df.columns[:5])}...")

######################################################################


# if __name__ == "__main__":
#     # 1. Setup the listener (Argparse)
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", type=str, required=True, help="Path to input PSV")
#     parser.add_argument("--output", type=str, required=True, help="Path for output CSV")
#     args = parser.parse_args()

#     # 2. Use the variables from the terminal (args.input) instead of hardcoded strings
#     print(f"Reading from: {args.input}")
#     df = pd.read_csv(args.input, sep="|")

#     # 3. Run Pipeline
#     pipeline = build_sepsis_pipeline()
#     processed_array = pipeline.fit_transform(df)
    
#     # 4. Debugging info
#     print(f"Current Working Directory: {os.getcwd()}")
#     print(f"Attempting to write to: {os.path.abspath(args.output)}")

#     # 5. Save to the path specified in --output
#     pd.DataFrame(processed_array).to_csv(args.output, index=False)
#     print("Success! File saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="|")

    # 1. SEPARATE LABEL IMMEDIATELY
    # This ensures the label is never dropped or scaled
    y = df['SepsisLabel'].copy()
    X = df.drop(columns=['SepsisLabel'])

    # 2. PRE-CLEAN: Drop columns that are 100% empty
    # This prevents the "divide by zero" warnings in StandardScaler
    X = X.dropna(axis=1, how='all')
    
    # 3. RUN PIPELINE
    pipeline = build_sepsis_pipeline()
    
    # Use the manual approach since your custom classes had issues with set_output
    processed_array = pipeline.fit_transform(X)
    
    # 4. RECONSTRUCT DATAFRAME
    # We get the names from the selector step to see what survived
    selector = pipeline.named_steps['selector']
    # If your selector has an attribute like 'feature_names_in_', use it:
    remaining_columns = [c for c in X.columns if c not in selector.constant_cols_]
    
    processed_df = pd.DataFrame(processed_array, columns=remaining_columns)
    
    # 5. FINAL SAFETY CHECKS
    # Re-attach the un-scaled, un-touched label
    processed_df['SepsisLabel'] = y.values
    
    # Fill any NaNs that survived (like from division errors)
    processed_df = processed_df.fillna(0)
    
    # Clip extreme values to prevent 'NaN loss' in training
    processed_df = processed_df.clip(lower=-10, upper=10)

    # 6. SAVE
    processed_df.to_csv(args.output, index=False)
    print(f"Success! Processed shape: {processed_df.shape}")