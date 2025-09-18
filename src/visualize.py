# src/visualize.py
import numpy as np
import torch
from torch.cuda.amp import autocast
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

@torch.no_grad()
def plot_tsne_features(
    model, ct_loader, mri_loader, title: str, output_path: str, num_samples: int = 200
):
    """
    Extracts bottleneck features from a model for two domains, computes
    t-SNE, and saves a scatter plot.
    """
    model.eval()
    all_features, domain_labels = [], []
    
    # Collect features for CT domain
    ct_count = 0
    for batch in ct_loader:
        if ct_count >= num_samples: break
        img = batch['image'].to(torch.device('cuda'))
        with autocast(device_type='cuda'):
            _, features_list = model.forward_with_features(img)
            bottleneck = features_list[-1].cpu().numpy()
        all_features.append(bottleneck)
        domain_labels.extend([0] * bottleneck.shape[0])
        ct_count += bottleneck.shape[0]

    # Collect features for MRI domain
    mri_count = 0
    for batch in mri_loader:
        if mri_count >= num_samples: break
        img = batch['image'].to(torch.device('cuda'))
        with autocast(device_type='cuda'):
            _, features_list = model.forward_with_features(img)
            bottleneck = features_list[-1].cpu().numpy()
        all_features.append(bottleneck)
        domain_labels.extend([1] * bottleneck.shape[0])
        mri_count += bottleneck.shape[0]

    # Compute t-SNE
    all_features = np.concatenate(all_features, axis=0)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(all_features)
    
    # Plot results
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    domain_map = {0: 'CT (Source)', 1: 'MRI (Target)'}
    readable_labels = [domain_map[l] for l in domain_labels]
    
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1], hue=readable_labels,
        style=readable_labels, palette='deep', s=80, alpha=0.8
    )
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title='Domain')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"t-SNE plot saved to {output_path}")