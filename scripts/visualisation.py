from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import data_preparation
import classification

# Load dataset
labels, embeddings = data_preparation.load_dataset()

# Load model
model, label_encoder = classification.load_model()

def _visualise_3d_feature_space():
    encoded_labels = label_encoder.fit_transform(labels)

   # Reduce dimensionality to 3D
    pca = PCA(n_components=3)
    embeddings_pca = pca.fit_transform(embeddings)

    # Plot data points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original data points
    for label in np.unique(encoded_labels):
        ax.scatter(embeddings_pca[encoded_labels == label, 0],
                embeddings_pca[encoded_labels == label, 1],
                embeddings_pca[encoded_labels == label, 2],
                label=label_encoder.inverse_transform([label])[0])

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('Feature Space Visualisation in PCA-reduced Space')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    _visualise_3d_feature_space()