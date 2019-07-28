
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import Normalizer, MaxAbsScaler, Binarizer, MinMaxScaler, StandardScaler, RobustScaler
from Helper import utilities


'''
This module is used to visualize features using dimensionality reduction and feature selection
'''


# This function visualizes dimensionality reduction plots
'''
Params:
    @:param: plots - List of dimension reduction methods
    @:param: features - dataset input features
    @:param: labels - List of labels / class names
    @:param: title - Title for the plot 
    @:param n_components - Number of components the features should be reduced to
    
'''
def plot_dimension_reduction(plots, features, labels, title = "Ethnicity", n_components=2):

    colors = len(list(set(labels)))
    fig = plt.figure(figsize=(40, 40))
    plt.suptitle(title)
    rows = 1
    cols = 2

    features = Normalizer().fit(features).transform(features)

    if len(plots) > 2:
        rows = 2

        if len(plots) > 4:
            cols = 3
        else:
            cols = 2

    # For all plots in the plot list
    for i in range(0,len(plots)):

        plot1 = plots[i]

        # Creation of the model
        if (plot1 == 'PCA'):
            mod = PCA(n_components=n_components)
            title = "PCA decomposition"

        elif (plot1 == 'TSNE'):
            mod = TSNE(n_components=2, learning_rate=300,perplexity = 30,early_exaggeration = 12,init = 'random',  random_state=22)
            title = "t-SNE decomposition"

        elif plot1 == "FICA":
            mod = FastICA(n_components=n_components, algorithm='parallel', whiten=True, max_iter=100, random_state=22)
            title = "Fast ICA"

        elif plot1 == "MDS":
            mod = MDS(n_components=n_components, n_init=12, max_iter=100, metric=True, n_jobs=-1, random_state=22)
            title = "MDS"

        elif plot1 == "ISO":
            mod = Isomap(n_components=n_components, n_jobs=-1, n_neighbors=100)
            title = "ISO MAP"

        elif plot1 == "LLE":
            mod = LocallyLinearEmbedding(n_components=n_components, n_neighbors=20, method='modified', n_jobs=-1, random_state=22)
            title = "Locally Linear Embedding"


        # Fit and transform the features
        principal_components = mod.fit_transform(features)


        # Put them into a dataframe
        df_features = pd.DataFrame(data=principal_components,
                                   columns=['PC1', 'PC2'])

        # Now we have to paste each row's label and its meaning
        # Convert labels array to df
        df_labels = pd.DataFrame(data=labels,
                                 columns=['label'])




        df_full = pd.concat([df_features, df_labels], axis=1)
        df_full['label'] = df_full['label'].astype(str)
        #df_full.drop(df_full[df_full["label"] == "Indian"].index, inplace=True)



        # Plot
        sns.set_style("dark")

        fig.add_subplot(rows, cols, i + 1)
        sns.scatterplot(x='PC1',
                        y='PC2',
                        hue="label",
                        data=df_full,
                        palette=sns.color_palette("hls", colors),
                        alpha=.7, legend="full",
                        x_jitter=20, y_jitter=0).set_title(title)



    plt.show()


if __name__ == '__main__':

    dataset_embeddings, dataset_labels, dataset_imagepaths = utilities.load_embeddings(embed_filename="embeddings_ethnicity.pkl")
    plot_dimension_reduction(plots=["PCA", "TSNE", "ISO", "FICA", "LLE", "MDS"], features=dataset_embeddings, labels=dataset_labels,
                             title = "Face_Recognition")
