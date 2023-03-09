from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


def plot_features(model, data_loader, num_feats, batch_size, num_samples):
    num_samples = len(data_loader) if not num_samples else num_samples
    feats = np.array([]).reshape((0, num_feats))
    labels = np.array([])
    model.eval()
    model.cuda()

    processed_samples = 0
    with torch.no_grad():
        for (x1, x2, _), label in data_loader:
            print(x1, x2, _)
            if processed_samples >= num_samples:
                break
            x1 = x1.squeeze().cuda()
            out = model(x1)
            out = out.cpu().data.numpy()
            feats = np.append(feats, out, axis=0)
            labels = np.append(labels, label, axis=0)
            processed_samples += batch_size

    tsne = TSNE(n_components=3, perplexity=50, init='pca')
    x_feats = tsne.fit_transform(feats)
    print(x_feats.shape, labels.shape)

    dim_red_df = pd.DataFrame(x_feats)
    dim_red_df['labels'] = pd.Categorical(labels)
    fig = px.scatter_3d(dim_red_df, x=0, y=1, z=2, color='labels')
    fig.show()
