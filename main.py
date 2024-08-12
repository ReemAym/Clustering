
import tkinter as tk
from tkinter import filedialog, messagebox
import random
import math
import pandas as pd
from collections import defaultdict

# Import the k-means and outlier detection functions from your existing code
# Function to read data from a CSV file and select two columns
def read_csv(file_path, percent):
    """
    Load transaction data from a CSV file.
    """
    transactions = defaultdict(list)
    df = pd.read_csv(file_path)
    num_records = int(len(df) * (percent / 100))
    df = df.head(num_records)
    for index, row in df.iterrows():
        movie_name = row['Movie Name']
        rate = row['IMDB Rating']
        transactions[movie_name].append(rate)
    return transactions

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(point1, point2)]))

# Function to assign each point to the closest centroid
def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point[1], centroid) for centroid in centroids]
        closest_centroid_index = distances.index(min(distances))
        clusters[closest_centroid_index].append(point)
    return clusters

# Function to update centroids based on the mean of points in each cluster
def update_centroids(clusters):
    centroids = []
    for cluster in clusters:
        if cluster:
            # Transpose the data before computing mean along each dimension
            cluster_mean = [sum(dim) / len(dim) for dim in zip(*[point[1] for point in cluster])]
            centroids.append(cluster_mean)
        else:
            # If cluster is empty, keep the centroid unchanged
            centroids.append([])
    return centroids

# Function to implement k-means algorithm
def k_means(data, k, max_iterations=100):
    # Initialize centroids randomly
    centroids = []
    while len(centroids) < k:
        random_point = random.choice([point[1] for point in data])
        if random_point not in centroids and random_point:
            centroids.append(random_point)

    for _ in range(max_iterations):
        # Assign points to clusters
        clusters = assign_clusters(data, centroids)

        # for i, cluster in enumerate(clusters):
        #     print(f"Cluster {i + 1}:*****************************************************************")
        #     for movie, rating_list in cluster:
        #         print("-", movie, "(Ratings:", rating_list, ")")

        # Update centroids
        new_centroids = update_centroids(clusters)

        # Check for convergence
        # converged = all([euclidean_distance(new_centroids[i], centroids[i]) < 1e-5 for i in range(k)])
        # if converged:
        if new_centroids == centroids:
            break

        centroids = new_centroids

    return clusters, centroids


# Function to detect outliers using Inter Quartile Range (IQR)
def detect_outliers(clusters):
    outliers = []
    for cluster_index, cluster in enumerate(clusters):
        ratings = [rating for point in cluster for rating in point[1]]
        ratings_sorted = sorted(ratings)
        n = len(ratings_sorted)

        q1_index = int(n * 0.25)
        q3_index = int(n * 0.75)

        if q1_index >= n or q3_index >= n:
            continue

        q1 = ratings_sorted[q1_index]
        q3 = ratings_sorted[q3_index]

        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        cluster_outliers = []  # List to store outliers within the current cluster
        for point in cluster:
            for rating in point[1]:
                if rating < lower_bound or rating > upper_bound:
                    cluster_outliers.append((point[0], rating, cluster_index + 1))  # Store outlier with cluster index
        # Remove outliers from the cluster
        for outlier in cluster_outliers:
            if (outlier[0], [outlier[1]]) in cluster:
              cluster.remove((outlier[0], [outlier[1]]))  # Remove outlier from the cluster
        # Store outliers separately
        outliers.extend(cluster_outliers)
    return outliers

#
# file_path = input("Enter file path: ")
# percent = float(input("Enter percent of records to read: "))
# k = int(input("Enter number of clusters: "))
# # threshold = float(input("Enter outlier detection threshold: "))
#
# data = read_csv(file_path, percent)
#
# clusters, centroids = k_means(list(data.items()), k)
#
# # Detect outliers
# outliers = detect_outliers(clusters)
#
# # Print clusters
# for i, cluster in enumerate(clusters):
#     print(f"Cluster {i + 1}:*****************************************************************")
#     for movie, rating_list in cluster:
#         print("-", movie, "(Ratings:", rating_list, ")")
#
# # Print centroids
# print("\nCentroids:")
# for i, centroid in enumerate(centroids):
#     print(f"Centroid {i + 1}: {centroid}")
#
# # Print outliers
# print("\nOutliers:")
# for outlier in outliers:
#     print(f"- {outlier[0]} (Rating: {outlier[1]},  cluster index:{outlier[2]} )")



class KMeansOutlierDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means Clustering and Outlier Detection")

        self.create_widgets()

    def create_widgets(self):
        # File path selection
        self.label_file_path = tk.Label(self.root, text="Select CSV file:")
        self.label_file_path.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.entry_file_path = tk.Entry(self.root, width=50)
        self.entry_file_path.grid(row=0, column=1, columnspan=2, padx=5, pady=5)

        self.button_browse = tk.Button(self.root, text="Browse", command=self.select_file)
        self.button_browse.grid(row=0, column=3, padx=5, pady=5)

        # Percentage of records
        self.label_percent = tk.Label(self.root, text="Percentage of records to read:")
        self.label_percent.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.entry_percent = tk.Entry(self.root)
        self.entry_percent.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

        # Number of clusters (k)
        self.label_k = tk.Label(self.root, text="Number of clusters (k):")
        self.label_k.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.entry_k = tk.Entry(self.root)
        self.entry_k.grid(row=2, column=1, columnspan=2, padx=5, pady=5)

        # Run button
        self.button_run = tk.Button(self.root, text="Run", command=self.run_analysis)
        self.button_run.grid(row=3, column=1, columnspan=2, padx=5, pady=10)

        # Output text widget
        self.output_text = tk.Text(self.root, height=15, width=60)
        self.output_text.grid(row=4, column=0, columnspan=4, padx=5, pady=5)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.entry_file_path.delete(0, tk.END)
        self.entry_file_path.insert(0, file_path)

    def run_analysis(self):
        file_path = self.entry_file_path.get()
        percent = float(self.entry_percent.get())
        k = int(self.entry_k.get())

        try:
            data = read_csv(file_path, percent)
            clusters, centroids = k_means(list(data.items()), k)
            outliers = detect_outliers(clusters)

            self.display_results(clusters, centroids, outliers)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_results(self, clusters, centroids, outliers):
        self.output_text.delete(1.0, tk.END)

        self.output_text.insert(tk.END, "Clusters:\n")
        for i, cluster in enumerate(clusters):
            self.output_text.insert(tk.END, f"Cluster {i + 1}:\n")
            for movie, rating_list in cluster:
                self.output_text.insert(tk.END, f"- {movie} (Ratings: {rating_list})\n")

        self.output_text.insert(tk.END, "\nCentroids:\n")
        for i, centroid in enumerate(centroids):
            self.output_text.insert(tk.END, f"Centroid {i + 1}: {centroid}\n")

        self.output_text.insert(tk.END, "\nOutliers:\n")
        for outlier in outliers:
            self.output_text.insert(tk.END, f"- {outlier[0]} (Rating: {outlier[1]}, Cluster Index: {outlier[2]})\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansOutlierDetectionGUI(root)
    root.mainloop()


