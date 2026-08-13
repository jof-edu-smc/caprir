import csv
import random
import matplotlib.pyplot as plt
import numpy as np
import umap
import hdbscan
import seaborn as sns
import plotly.express as px
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer, util

stopwords = [
        "a", "an", "the", "from", "and", ".", "of", "this", "or", "but", "are", "by", 
        "like", "with", "rt", "rir", "that", "to", "in", "on", "for", "is", "as", "at", 
        "it", "be", "by", "of", "into", "room", "space", "sound", "sounds"
    ]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--metadata", type=str, default="metadata_synthetic.csv", help="Path to the metadata CSV file for the dataset")
    parser.add_argument("--output_png", type=str, default="clustering_dbscan.png", help="Path to the output file for the dendrogram")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum number of tokens to consider")
    args = parser.parse_args()

    with open(args.metadata, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        table = list(csv_reader)

    for col in range(0, len(table[0])):
        print(f"Column {col}: {table[0][col]}")

    # 1. Load a pre-trained sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # or 'BAAI/bge-base-en-v1.5'
    # model = SentenceTransformer("BAAI/bge-base-en-v1.5") 
    # 2. Define input sentences
    # sentence1 = "The acoustic impulse response was recorded in a rectangular room."
    # sentence2 = "A room impulse response measurement was taken inside a shoe-box space."
    # sentence3 = "Stock prices fluctuated dramatically on the market today."
    rt60, drr, c50, c80, d50 = [], [], [], [], []
    locations = []
    sentences = [] 
    filenames = [] 
    location_nums = {}  # New list to store location numbers
    for row in table[1:]: 
        for col in range(5, 10): 
            if row[col]:  # Check if the cell is not empty
                # print(row[col])
                sentences.append(' '.join([word for word in row[col].split() if word.lower() not in stopwords]))  # Remove stopwords and strip whitespace
                # sentences.append(row[col].strip())  # Strip whitespace
                locations.append(row[10])
                if row[10] not in location_nums:
                    location_nums[row[10]] = len(location_nums) + 1  # Assign a new number
                rt60.append(row[12])
                drr.append(row[13])
                c50.append(row[14])
                c80.append(row[15])
                d50.append(row[16])
                filenames.append(row[0]) 
                
    # 3. Compute embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)

    print("Embeddings shape:", embeddings.shape)  # Should be (3, embedding_dim)
    
embeddings_np = embeddings.cpu().numpy()  # Convert to numpy array for UMAP and HDBSCAN
embeddings_norm = normalize(embeddings_np, norm='l2')  # Normalize embeddings for better clustering performance


# plt.title("Hierarchical Clustering Dendrogram of Sentences in CapRIR")

distance_matrix = pdist(embeddings_norm, metric='cosine') 
linkage_matrix = linkage(distance_matrix, method='ward')  
# Shorten sentences for dendrogram labels
dendrogram_sentences = [sent[:75] + "..." if len(sent) > 75 else sent for sent in sentences]  

# Vertical
plt.figure(figsize=(10, 24))
dendrogram(
    linkage_matrix,
    labels=dendrogram_sentences,
    leaf_font_size=12,
    color_threshold=None,
    orientation='left',
)
plt.tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig("clustering_sentences_dendrogram_v.png", format='png', bbox_inches='tight', dpi=300)
plt.close()

# Horizontal
plt.figure(figsize=(24, 10))
dendrogram(
    linkage_matrix,
    labels=dendrogram_sentences,
    leaf_rotation=-90,
    leaf_font_size=12,
    color_threshold=None,
)
plt.tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig("clustering_sentences_dendrogram_h.png", format='png', bbox_inches='tight', dpi=300)
plt.close()

n_neighbors = 10
min_dist = 0.01
metric = 'cosine'
random_state = 42

# 1. Reduce dimensionality using Cosine metric
# umap_clusters = umap.UMAP(
#     n_neighbors=n_neighbors,
#     min_dist=min_dist, 
#     n_components=5, 
#     metric=metric, 
#     random_state=random_state
# ).fit_transform(embeddings_norm)
# print("UMAP 5D shape:", umap_clusters.shape)  # Should be (3, 5)

umap_2d = umap.UMAP(
    n_neighbors=n_neighbors, 
    min_dist=min_dist,          
    n_components=2, 
    metric=metric, 
    random_state=random_state
).fit_transform(embeddings_norm)
print("UMAP 2D shape:", umap_2d.shape)  # Should be (3, 2)

# 2. Cluster dense regions
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,
    min_samples=2, 
    metric='euclidean',
    cluster_selection_method='eom'
)
labels = clusterer.fit_predict(umap_2d)

print(f"Number of clusters (with -1 outliers excluded) found: {len(set(labels)) - (1 if -1 in labels else 0)}")

# Create a DataFrame combining 2D coordinates, labels, and text
df = pd.DataFrame({
    "UMAP_X": umap_2d[:, 0],
    "UMAP_Y": umap_2d[:, 1],
    "Cluster": [str(lbl) for lbl in labels], # String category for discrete colors
    "Sentence": sentences,
    "Sentence_idx": [i for i in range(len(sentences))],
    "Filename": filenames,
    "Location": locations,
    "Location_Num": [num for num in [location_nums[loc] for loc in locations]],  # Map location to its number
    "RT60": rt60,  
    "DRR": drr,  
    "C50": c50, 
    "C80": c80,
    "D50": d50,
})

numerical_columns = ["RT60", "DRR", "C50", "C80", "D50"]
df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, set errors to NaN

# Define report-ready figure dimensions (11x8.5 inches fits standard A4/US Letter paper margins)
fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=300)

cluster_colors = [
    "#E69F00", "#0B7306", "#009E73", "#F0E442",
    "#0072B2", "#D50000", "#CC80A7", "#810242"
]

# --- 1. Subplot (1, 1): HDBSCAN Text Clusters ---
ax1 = axes[1]
for cluster_id, group in df.groupby("Cluster"):
    is_noise = str(cluster_id) == "-1"
    name = "Outliers (-1)" if is_noise else f"Cluster {cluster_id}"
    color = "#808080" if is_noise else cluster_colors[int(cluster_id) % len(cluster_colors)]
    
    ax1.scatter(
        group["UMAP_X"], group["UMAP_Y"],
        label=name, color=color, s=30, alpha=0.85, edgecolors="none"
    )
    for _, row in group.iterrows():
        if pd.notna(row["DRR"]):
            ax1.annotate(
                f"{row['DRR']:.2f}",          # Formats the DRR value to 2 decimal places
                (row["UMAP_X"], row["UMAP_Y"]),  # Point coordinates
                textcoords="offset points", 
                xytext=(4, 4),                  # Shifts text 4pt right and 4pt up
                fontsize=10,                   # Small font to avoid clutter
                alpha=0.85
            )

ax1.set_title("Labeled by Direct-to-Reverberant Ratio [DRR]", fontsize=12, fontweight="bold")
ax1.set_xlabel("UMAP Dim 1", fontsize=10)
ax1.set_ylabel("UMAP Dim 2", fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.legend(fontsize=10, loc="best", frameon=True, framealpha=0.9)

# --- 2. Subplot (1, 2): Location Distribution ---
ax2 = axes[0]
for i, (loc_name, group) in enumerate(df.groupby("Location")):
    color = cluster_colors[i % len(cluster_colors)]
    
    ax2.scatter(
        group["UMAP_X"], group["UMAP_Y"],
        label=f"{loc_name}", color=color, s=30, alpha=0.85, edgecolors="none"
    )
    for _, row in group.iterrows():
        if pd.notna(row["Sentence_idx"]):
            ax2.annotate(
                f"{row['Sentence_idx']}",        
                (row["UMAP_X"], row["UMAP_Y"]),  # Point coordinates
                textcoords="offset points", 
                xytext=(4, 4),                  # Shifts text 4pt right and 4pt up
                fontsize=10,                   # Small font to avoid clutter
                alpha=0.85
            )

ax2.set_title("Location Distribution", fontsize=12, fontweight="bold")
ax2.set_xlabel("UMAP Dim 1", fontsize=10)
ax2.set_ylabel("UMAP Dim 2", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.legend(fontsize=10, loc="best", frameon=True, framealpha=0.9)

# --- 3. Subplot (2, 1): HDBSCAN Clusters Labeled with RT60 Values ---
ax3 = axes[2]

for cluster_id, group in df.groupby("Cluster"):
    # Style noise (-1) as grey, assigned clusters with qualitative palette
    is_noise = str(cluster_id) == "-1"
    legend_name = "Noise (-1)" if is_noise else f"Cluster {cluster_id}"
    color = "#808080" if is_noise else cluster_colors[int(cluster_id) % len(cluster_colors)]
    
    # Plot scatter points for the cluster
    ax3.scatter(
        group["UMAP_X"], group["UMAP_Y"],
        label=legend_name, color=color, s=30, alpha=0.85, edgecolors="none"
    )
    
    # Add RT60 text label next to each point
    for _, row in group.iterrows():
        if pd.notna(row["RT60"]):
            ax3.annotate(
                f"{row['RT60']:.2f}s",          # Formats RT60 to 2 decimal places (e.g., '0.45s')
                (row["UMAP_X"], row["UMAP_Y"]),  # Point coordinates
                textcoords="offset points", 
                xytext=(4, 4),                  # Shifts text 4pt right and 4pt up
                fontsize=10,                   # Small font to avoid clutter
                alpha=0.85
            )

ax3.set_title("Labeled by Reverberation Time [RT60]", fontsize=12, fontweight="bold")
ax3.set_xlabel("UMAP Dim 1", fontsize=10)
ax3.set_ylabel("UMAP Dim 2", fontsize=10)
ax3.grid(True, linestyle="--", alpha=0.3)
ax3.legend(fontsize=10, loc="best", frameon=True, framealpha=0.9)

# # --- 4. Subplot (2, 2): GMM Acoustic Clusters ---
ax4 = axes[3]
for cluster_id, group in df.groupby("Cluster"):
    color = cluster_colors[int(cluster_id) % len(cluster_colors)]
    is_noise = str(cluster_id) == "-1"
    legend_name = "Noise (-1)" if is_noise else f"Cluster {cluster_id}"
    color = "#808080" if is_noise else cluster_colors[int(cluster_id) % len(cluster_colors)]
    
    ax4.scatter(
        group["UMAP_X"], group["UMAP_Y"],
        label=legend_name, color=color, s=30, alpha=0.85, edgecolors="none"
    )
    # Add C80 text label next to each point
    for _, row in group.iterrows():
        if pd.notna(row["C80"]):
            ax4.annotate(
                f"{row['C80']:.2f} dB",          # Formats C80 to 2 decimal places (e.g., '0.45 dB')
                (row["UMAP_X"], row["UMAP_Y"]),  # Point coordinates
                textcoords="offset points", 
                xytext=(4, 4),                  # Shifts text 4pt right and 4pt up
                fontsize=10,                   # Small font to avoid clutter
                alpha=0.85
            )

ax4.set_title("Labeled by Clarity for Music [C80]", fontsize=12, fontweight="bold")
ax4.set_xlabel("UMAP Dim 1", fontsize=10)
ax4.set_ylabel("UMAP Dim 2", fontsize=10)
ax4.grid(True, linestyle="--", alpha=0.3)
ax4.legend(fontsize=10, loc="best", frameon=True, framealpha=0.9)

# Adjust layout for clear separation and export
plt.tight_layout()
plt.subplots_adjust(top=0.92)  # Prevent title overlap
plt.savefig("clustering_umap.png", dpi=300, bbox_inches="tight")
plt.close()

# Create figure that only shows the location labels on top of the umap axes and saves to file. 
fig, axething = plt.subplots(figsize=(8, 8), dpi=300)

for i, (loc_name, group) in enumerate(df.groupby("Location")):
    color = cluster_colors[i % len(cluster_colors)]
    
    axething.scatter(
        group["UMAP_X"], group["UMAP_Y"],
        label=f"{loc_name}", color=color, s=30, alpha=0.85, edgecolors="none"
    )
axething.set_title("Colored by Location", fontsize=12, fontweight="bold")
axething.set_xlabel("UMAP Dim 1", fontsize=10)
axething.set_ylabel("UMAP Dim 2", fontsize=10)
axething.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("clustering_synthetic_location_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

metrics = ["RT60", "DRR", "C50", "C80", "D50"]

# 1. Custom aggregation function for Interquartile Range (IQR)
def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

# 2. Compute group-wise summary statistics by Location
location_stats = df.groupby("Location")[metrics].agg(
    ["count", "mean", "std", "median", iqr]
)

# Display formatted table
print(location_stats)