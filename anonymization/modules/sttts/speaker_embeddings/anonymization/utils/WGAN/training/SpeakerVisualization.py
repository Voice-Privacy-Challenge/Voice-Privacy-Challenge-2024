import matplotlib
import torch
from matplotlib import cm
from matplotlib import pyplot as plt

# matplotlib.use("tkAgg")
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
import numpy
from tqdm import tqdm


class Visualizer:
    def __init__(self, algorithm='pca'):
        self.algorithm = algorithm

        if algorithm == 'pca+tsne':
            self.pca = PCA(n_components=100)
            self.algo = TSNE(n_jobs=-1, n_iter_without_progress=4000, n_iter=20000, verbose=1)
        elif algorithm == 'tsne':
            self.algo = TSNE(n_jobs=-1, n_iter_without_progress=4000, n_iter=20000, verbose=1)
        elif algorithm == 'pca':
            self.algo = PCA(n_components=100)
        elif algorithm == 'kernelpca':
            self.algo = KernelPCA(
                n_components=2, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.5
            )

    def visualize_speaker_embeddings(self, list_of_generated_embeddings, list_of_original_embeddings, title_of_plot,
                                     save_file_path=None, legend=True, colors=None):
        label_list = list()
        embedding_list = list()
        for emb in list_of_original_embeddings:
            embedding_list.append(emb.squeeze().numpy())
            label_list.append("Human")
        for emb in list_of_generated_embeddings:
            embedding_list.append(emb.squeeze().numpy())
            label_list.append("Generated")
        embeddings_as_array = numpy.array(embedding_list)

        if self.algorithm == 'pca+tsne':
            embeddings_as_array = self.pca.fit_transform(embeddings_as_array)

        dimensionality_reduced_embeddings_tsne = self.algo.fit_transform(embeddings_as_array)
        return self._plot_embeddings(projected_data=dimensionality_reduced_embeddings_tsne,
                                     labels=label_list,
                                     title=title_of_plot,
                                     save_file_path=save_file_path,
                                     legend=legend,
                                     colors=colors)

    def _plot_embeddings(self, projected_data, labels, title, save_file_path, legend, colors):
        if colors is None:
            colors = cm.PiYG(numpy.linspace(0, 1, len(set(labels))))
        label_to_color = dict()
        for index, label in enumerate(sorted(list(set(labels)))):
            label_to_color[label] = colors[index]

        labels_to_points_x = dict()
        labels_to_points_y = dict()
        for label in labels:
            labels_to_points_x[label] = list()
            labels_to_points_y[label] = list()
        for index, label in enumerate(labels):
            labels_to_points_x[label].append(projected_data[index][0])
            labels_to_points_y[label].append(projected_data[index][1])

        fig, ax = plt.subplots()
        for label in sorted(list(set(labels)), reverse=True):
            x = numpy.array(labels_to_points_x[label])
            y = numpy.array(labels_to_points_y[label])
            ax.scatter(x=x,
                       y=y,
                       c=label_to_color[label],
                       label=label,
                       alpha=0.5)
        if legend:
            ax.legend()
        fig.tight_layout()
        ax.axis('off')
        fig.subplots_adjust(top=0.9, bottom=0.0, right=1.0, left=0.0)
        ax.set_title(title)
        if save_file_path is not None:
            plt.savefig(save_file_path)
        else:
            plt.show()
        plt.close()
        return plt