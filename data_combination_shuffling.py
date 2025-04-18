# import pandas as pd

# # Load the scam and ham datasets
# dataset_loc = "dataset/testdata/"

# scam_df = pd.read_excel(dataset_loc + "Spam Updated Dataset [NEW].xlsx", header=None, names=["message"])
# ham_df = pd.read_excel(dataset_loc + "Ham Updated Dataset [NEW].xlsx", header=None, names=["message"])

# # Assign labels (1 for scam, 0 for ham)
# scam_df["label"] = 1
# ham_df["label"] = 0

# # Combine both datasets
# dataset = pd.concat([scam_df, ham_df], ignore_index=True)

# # Reorder columns (label first)
# dataset = dataset[["label", "message"]]
# print(dataset.head())
# # Save to CSV
# dataset.to_csv("scam_ham_dataset_combined_unshuff_6k.csv", index=False)

# print("Dataset saved successfully!")


import pandas as pd

# Load your dataset
df = pd.read_csv('dataset/scam_ham_dataset_combined_v2.csv')

# Shuffle the dataset
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the shuffled dataset to a new file if needed
shuffled_df.to_csv('dataset/finaldataset_6k_shuffled_v2.csv', index=False)
print("shuffling done")