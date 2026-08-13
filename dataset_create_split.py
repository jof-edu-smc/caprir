import pandas as pd
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_synthetic", type=str, default="metadata_synthetic.csv", help="Path to the dataset CSV file")
    parser.add_argument("--dataset_natural", type=str, default="metadata_natural.csv", help="Path to the dataset CSV file")
    parser.add_argument("--train_split", type=float, default=0.75, help="Fraction of the dataset to use for training (between 0 and 1)")
    parser.add_argument("--test_split", type=float, default=0.25, help="Fraction of the dataset to use for testing (between 0 and 1)")
    args = parser.parse_args()
    
    if args.train_split + args.test_split != 1.0:
        raise ValueError("Train and test split fractions must sum to 1.0")
    
    # Load Synthetic and Natural datasets
    df_synth = pd.read_csv(args.dataset_synthetic)

    df_natural = pd.read_csv(args.dataset_natural)
    
    # Find matching rows based on 'RIR' column
    test_df = df_synth[df_synth['RIR'].isin(df_natural['RIR'])]
    
    if len(test_df) != int(args.test_split * len(df_synth)):
        # Add more rows to test_df from df_synth that are not in df_natural until we reach the desired number
        additional_rows_needed = int(args.test_split * len(df_synth)) - len(test_df)
        additional_rows = df_synth[~df_synth['RIR'].isin(df_natural['RIR'])].sample(n=additional_rows_needed, random_state=42)
        test_df = pd.concat([test_df, additional_rows])
        
    # Remaining rows are the training split
    train_df = df_synth.drop(test_df.index)

    # Save output CSV files
    train_df.to_csv(
        f"train_{args.dataset_synthetic.replace('.csv', '')}_{str(args.train_split).replace('.', '')}.csv", 
        index=False
    )
    test_df.to_csv(
        f"test_{args.dataset_synthetic.replace('.csv', '')}_{str(args.test_split).replace('.', '')}.csv", 
        index=False
    )

    print(
        f"Total rows Synth: {len(df_synth)} | Total rows Natural: {len(df_natural)} | 75% split: {len(train_df)} | 25% split: {len(test_df)}"
    )