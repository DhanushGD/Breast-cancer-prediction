import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset = pd.read_csv("data.csv")  # Replace with your actual dataset file path

# Encode the 'diagnosis' column (string 'M' and 'B' into numeric 0 and 1)
label_encoder = LabelEncoder()
dataset['diagnosis'] = label_encoder.fit_transform(dataset['diagnosis'])

# Function to generate the Correlation Graph and save it in static
def generate_correlation_graph():
    sns.set_style('darkgrid')

    # Selecting specific columns for correlation analysis
    cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
            'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

    plt.figure(figsize=(10, 8))
    plt.title("Correlation Graph")

    # Creating a diverging color palette
    cmap = sns.diverging_palette(1000, 120, as_cmap=True)

    # Generate the heatmap with correlation matrix
    sns.heatmap(dataset[cols].corr(), annot=True, fmt='.2f', linewidths=0.05, cmap=cmap)

    # Save the graph as an image in the static folder
    image_path = 'static/correlation_graph.png'
    plt.savefig(image_path)  # Save image
    plt.close()  # Close the plot
    print("Correlation graph saved at:", image_path)


# Function to generate Dataset Features plot and save it in static
def generate_dataset_features():
    sns.set_style('darkgrid')

    # Creating a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting histogram on the first subplot
    plt.sca(axes[0])
    plt.hist(dataset['diagnosis'])
    plt.title("Counts of Diagnosis")
    plt.xlabel("Diagnosis")

    # Plotting count plot on the second subplot
    plt.sca(axes[1])
    sns.countplot(x='diagnosis', data=dataset)

    # Save the plot as an image in the static folder
    image_path = 'static/dataset_features.png'
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    print("Dataset features plot saved at:", image_path)


# Run the functions to generate and save the images
generate_correlation_graph()
generate_dataset_features()
