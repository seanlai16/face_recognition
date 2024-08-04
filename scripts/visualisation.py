from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import data_preparation
import classification

# Load dataset
labels, embeddings = data_preparation.load_dataset()

# Load model
model, label_encoder = classification.load_model()

def _visualise_3d_decision_boundary():
    encoded_labels = label_encoder.fit_transform(labels)

   # Reduce dimensionality to 3D
    pca = PCA(n_components=3)
    embeddings_pca = pca.fit_transform(embeddings)

    # Train SVM in 3D PCA-reduced space
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(embeddings_pca, encoded_labels)

    # Create a 3D meshgrid
    x_min, x_max = embeddings_pca[:, 0].min() - 1, embeddings_pca[:, 0].max() + 1
    y_min, y_max = embeddings_pca[:, 1].min() - 1, embeddings_pca[:, 1].max() + 1
    z_min, z_max = embeddings_pca[:, 2].min() - 1, embeddings_pca[:, 2].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    zz = (-svm_model.coef_[0][0] * xx - svm_model.coef_[0][1] * yy - svm_model.intercept_[0]) / svm_model.coef_[0][2]

    # Plot decision boundaries and data points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot decision boundary
    ax.plot_surface(xx, yy, zz, alpha=0.5, rstride=100, cstride=100, color='blue', edgecolor='none')

    # Plot original data points
    for label in np.unique(encoded_labels):
        ax.scatter(embeddings_pca[encoded_labels == label, 0],
                embeddings_pca[encoded_labels == label, 1],
                embeddings_pca[encoded_labels == label, 2],
                label=label_encoder.inverse_transform([label])[0])

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D Decision Boundary in PCA-reduced Space')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    _visualise_3d_decision_boundary()