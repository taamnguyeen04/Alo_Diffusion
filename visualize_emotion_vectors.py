import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd

from criterions import DAN
from dataset import Affectnet


class EmotionVectorVisualizer:
    """Trích xuất và trực quan hóa emotion vectors từ mô hình DAN"""

    def __init__(self, model_path='affecnet7_epoch6_acc0.6569.pth', device=None):
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

        # Load model
        print("Loading DAN model...")
        self.model = DAN(num_head=4, num_class=7, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

    def extract_features(self, dataloader, max_samples=None):
        """
        Trích xuất feature vectors và labels từ dataset

        Args:
            dataloader: DataLoader của dataset
            max_samples: Số lượng mẫu tối đa (None = toàn bộ dataset)

        Returns:
            features: numpy array của feature vectors
            labels: numpy array của emotion labels
            predictions: numpy array của predicted labels
        """
        features_list = []
        labels_list = []
        predictions_list = []
        valence_list = []
        arousal_list = []

        print("Extracting features from dataset...")
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (images, expr, valence, arousal) in enumerate(tqdm(dataloader)):
                if max_samples and total_samples >= max_samples:
                    break

                images = images.to(self.device)

                # Forward pass - lấy output, features và attention heads
                out, features, heads = self.model(images)

                # Lấy predictions
                _, pred = torch.max(out, 1)

                # Chuyển features về dạng vector (global average pooling)
                # features có shape [batch, 512, H, W]
                feature_vectors = features.mean(dim=[2, 3])  # [batch, 512]

                # Hoặc có thể dùng sum của heads
                # head_features = heads.sum(dim=1)  # [batch, 512]

                features_list.append(feature_vectors.cpu().numpy())
                labels_list.append(expr.numpy())
                predictions_list.append(pred.cpu().numpy())
                valence_list.append(valence.numpy())
                arousal_list.append(arousal.numpy())

                total_samples += images.size(0)

        # Kiểm tra nếu không có dữ liệu
        if len(features_list) == 0:
            raise ValueError("No data found in the dataset! Please check:\n"
                           "1. Dataset path is correct\n"
                           "2. CSV files (training.csv or validation.csv) exist\n"
                           "3. Image files exist at the specified paths")

        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        predictions = np.concatenate(predictions_list)
        valences = np.concatenate(valence_list)
        arousals = np.concatenate(arousal_list)

        print(f"Extracted {len(features)} samples")
        print(f"Feature vector shape: {features.shape}")

        return features, labels, predictions, valences, arousals

    def reduce_dimensions(self, features, method='tsne', n_components=2):
        """
        Giảm chiều của feature vectors

        Args:
            features: Feature vectors
            method: 'tsne' hoặc 'pca'
            n_components: Số chiều output (2 hoặc 3)

        Returns:
            reduced_features: Features sau khi giảm chiều
        """
        print(f"Reducing dimensions using {method.upper()}...")

        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30, max_iter=1000)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError("method phải là 'tsne' hoặc 'pca'")

        reduced = reducer.fit_transform(features)
        print(f"Reduced to {n_components}D space")

        if method == 'pca':
            print(f"Explained variance ratio: {reducer.explained_variance_ratio_}")

        return reduced

    def plot_emotion_distribution(self, labels, save_path='emotion_distribution.png'):
        """Vẽ biểu đồ phân phối các cảm xúc trong dataset"""
        plt.figure(figsize=(12, 6))

        unique, counts = np.unique(labels, return_counts=True)
        emotion_names = [self.labels[i] for i in unique]

        colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))

        plt.subplot(1, 2, 1)
        plt.bar(emotion_names, counts, color=colors)
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Emotion Distribution in Dataset', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)

        # Thêm số lượng lên trên mỗi cột
        for i, (name, count) in enumerate(zip(emotion_names, counts)):
            plt.text(i, count, str(count), ha='center', va='bottom')

        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=emotion_names, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Emotion Distribution (%)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved emotion distribution plot to {save_path}")
        plt.close()

    def plot_2d_scatter(self, reduced_features, labels, predictions=None,
                       save_path='emotion_vectors_2d.png', title='Emotion Vectors Visualization',
                       show_centroids=True):
        """
        Vẽ scatter plot 2D của emotion vectors

        Args:
            reduced_features: Features đã giảm chiều (2D)
            labels: Ground truth labels
            predictions: Predicted labels (optional)
            save_path: Đường dẫn lưu file
            title: Tiêu đề biểu đồ
            show_centroids: Hiển thị tâm của các cụm dữ liệu
        """
        fig = plt.figure(figsize=(16, 6))

        # Plot 1: Ground truth labels
        ax1 = fig.add_subplot(1, 2, 1)
        colors = plt.cm.Set3(np.linspace(0, 1, 7))
        centroids = []

        for i, label_name in enumerate(self.labels):
            mask = labels == i
            if np.sum(mask) > 0:
                ax1.scatter(reduced_features[mask, 0], reduced_features[mask, 1],
                           c=[colors[i]], label=label_name, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

                # Tính và vẽ tâm cụm
                if show_centroids:
                    centroid = reduced_features[mask].mean(axis=0)
                    centroids.append(centroid)
                    # Vẽ tâm cụm với marker lớn hơn và hình sao
                    ax1.scatter(centroid[0], centroid[1], c=[colors[i]], marker='*',
                               s=400, edgecolors='black', linewidth=2, zorder=10)
                    # Thêm text label cho tâm cụm
                    ax1.annotate(label_name, (centroid[0], centroid[1]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=9, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7, edgecolor='black'),
                               zorder=11)

        ax1.set_xlabel('Dimension 1', fontsize=12)
        ax1.set_ylabel('Dimension 2', fontsize=12)
        ax1.set_title('Ground Truth Labels (★ = Centroids)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Predictions (nếu có)
        if predictions is not None:
            ax2 = fig.add_subplot(1, 2, 2)

            for i, label_name in enumerate(self.labels):
                mask = predictions == i
                if np.sum(mask) > 0:
                    ax2.scatter(reduced_features[mask, 0], reduced_features[mask, 1],
                               c=[colors[i]], label=label_name, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

                    # Vẽ tâm cụm cho predictions
                    if show_centroids:
                        centroid = reduced_features[mask].mean(axis=0)
                        ax2.scatter(centroid[0], centroid[1], c=[colors[i]], marker='*',
                                   s=400, edgecolors='black', linewidth=2, zorder=10)
                        ax2.annotate(label_name, (centroid[0], centroid[1]),
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=9, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7, edgecolor='black'),
                                   zorder=11)

            ax2.set_xlabel('Dimension 1', fontsize=12)
            ax2.set_ylabel('Dimension 2', fontsize=12)
            ax2.set_title('Model Predictions (★ = Centroids)', fontsize=14, fontweight='bold')
            ax2.legend(loc='best', framealpha=0.9)
            ax2.grid(True, alpha=0.3)

            # Tính accuracy
            accuracy = np.mean(labels == predictions)
            fig.suptitle(f'{title} (Accuracy: {accuracy:.2%})', fontsize=16, fontweight='bold')
        else:
            fig.suptitle(title, fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D scatter plot to {save_path}")
        plt.close()

    def plot_3d_scatter(self, reduced_features, labels, save_path='emotion_vectors_3d.png',
                       show_centroids=True):
        """Vẽ scatter plot 3D của emotion vectors"""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.Set3(np.linspace(0, 1, 7))

        for i, label_name in enumerate(self.labels):
            mask = labels == i
            if np.sum(mask) > 0:
                ax.scatter(reduced_features[mask, 0], reduced_features[mask, 1], reduced_features[mask, 2],
                          c=[colors[i]], label=label_name, alpha=0.6, s=30)

                # Vẽ tâm cụm
                if show_centroids:
                    centroid = reduced_features[mask].mean(axis=0)
                    ax.scatter(centroid[0], centroid[1], centroid[2],
                             c=[colors[i]], marker='*', s=500, edgecolors='black', linewidth=2)
                    # Thêm label cho centroid
                    ax.text(centroid[0], centroid[1], centroid[2], label_name,
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7, edgecolor='black'))

        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_zlabel('Dimension 3', fontsize=12)
        ax.set_title('3D Emotion Vectors Visualization (★ = Centroids)', fontsize=14, fontweight='bold')
        ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D scatter plot to {save_path}")
        plt.close()

    def plot_confusion_matrix(self, labels, predictions, save_path='confusion_matrix.png'):
        """Vẽ confusion matrix"""
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.labels, yticklabels=self.labels,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
        plt.close()

        # In classification report
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=self.labels))

    def plot_centroids_only(self, reduced_features, labels, save_path='emotion_centroids.png'):
        """Vẽ biểu đồ chỉ hiển thị các tâm cụm và khoảng cách giữa chúng"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        colors = plt.cm.Set3(np.linspace(0, 1, 7))
        centroids = []
        centroid_labels = []

        # Tính centroids
        for i, label_name in enumerate(self.labels):
            mask = labels == i
            if np.sum(mask) > 0:
                centroid = reduced_features[mask].mean(axis=0)
                centroids.append(centroid)
                centroid_labels.append(label_name)

        centroids = np.array(centroids)

        # Plot 1: Centroids với đường nối
        for i, (centroid, label_name) in enumerate(zip(centroids, centroid_labels)):
            # Vẽ centroid
            ax1.scatter(centroid[0], centroid[1], c=[colors[i]], marker='*',
                       s=600, edgecolors='black', linewidth=2, zorder=10)

            # Vẽ đường nối tới các centroids khác
            for j in range(i + 1, len(centroids)):
                ax1.plot([centroid[0], centroids[j, 0]],
                        [centroid[1], centroids[j, 1]],
                        'k--', alpha=0.2, linewidth=1)

                # Tính và hiển thị khoảng cách
                dist = np.linalg.norm(centroid - centroids[j])
                mid_point = (centroid + centroids[j]) / 2
                ax1.text(mid_point[0], mid_point[1], f'{dist:.2f}',
                        fontsize=8, ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

            # Label cho centroid
            ax1.annotate(label_name, (centroid[0], centroid[1]),
                       xytext=(15, 15), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[i], alpha=0.8, edgecolor='black'),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))

        ax1.set_xlabel('Dimension 1', fontsize=12)
        ax1.set_ylabel('Dimension 2', fontsize=12)
        ax1.set_title('Emotion Centroids with Distances', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Distance matrix heatmap
        n = len(centroids)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])

        im = ax2.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.set_xticklabels(centroid_labels, rotation=45, ha='right')
        ax2.set_yticklabels(centroid_labels)
        ax2.set_title('Centroid Distance Matrix', fontsize=14, fontweight='bold')

        # Thêm giá trị vào cells
        for i in range(n):
            for j in range(n):
                text = ax2.text(j, i, f'{dist_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)

        plt.colorbar(im, ax=ax2, label='Euclidean Distance')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved centroids plot to {save_path}")
        plt.close()

    def plot_valence_arousal(self, reduced_features, valences, arousals, labels,
                            save_path='valence_arousal.png'):
        """Vẽ biểu đồ valence-arousal"""
        fig = plt.figure(figsize=(18, 5))

        colors = plt.cm.Set3(np.linspace(0, 1, 7))

        # Plot 1: Valence-Arousal space
        ax1 = fig.add_subplot(1, 3, 1)
        for i, label_name in enumerate(self.labels):
            mask = labels == i
            if np.sum(mask) > 0:
                ax1.scatter(valences[mask], arousals[mask],
                           c=[colors[i]], label=label_name, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

        ax1.set_xlabel('Valence', fontsize=12)
        ax1.set_ylabel('Arousal', fontsize=12)
        ax1.set_title('Valence-Arousal Distribution', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        # Plot 2: Feature space colored by valence
        ax2 = fig.add_subplot(1, 3, 2)
        scatter2 = ax2.scatter(reduced_features[:, 0], reduced_features[:, 1],
                              c=valences, cmap='RdYlGn', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Dimension 1', fontsize=12)
        ax2.set_ylabel('Dimension 2', fontsize=12)
        ax2.set_title('Feature Space (colored by Valence)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter2, ax=ax2, label='Valence')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Feature space colored by arousal
        ax3 = fig.add_subplot(1, 3, 3)
        scatter3 = ax3.scatter(reduced_features[:, 0], reduced_features[:, 1],
                              c=arousals, cmap='YlOrRd', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Dimension 1', fontsize=12)
        ax3.set_ylabel('Dimension 2', fontsize=12)
        ax3.set_title('Feature Space (colored by Arousal)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter3, ax=ax3, label='Arousal')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved valence-arousal plot to {save_path}")
        plt.close()

    def analyze_emotion_clusters(self, reduced_features, labels):
        """Phân tích clusters của các cảm xúc"""
        print("\n" + "="*50)
        print("EMOTION CLUSTER ANALYSIS")
        print("="*50)

        centroids = {}
        for i, label_name in enumerate(self.labels):
            mask = labels == i
            if np.sum(mask) > 0:
                features_i = reduced_features[mask]
                center = features_i.mean(axis=0)
                std = features_i.std(axis=0)
                centroids[label_name] = center

                print(f"\n{label_name}:")
                print(f"  - Số mẫu: {np.sum(mask)}")
                print(f"  - Center (Centroid): [{center[0]:.3f}, {center[1]:.3f}]")
                print(f"  - Std: [{std[0]:.3f}, {std[1]:.3f}]")

        # Tính khoảng cách giữa các centroids
        print("\n" + "="*50)
        print("CENTROID DISTANCES")
        print("="*50)

        emotion_names = list(centroids.keys())
        for i, name1 in enumerate(emotion_names):
            for j in range(i + 1, len(emotion_names)):
                name2 = emotion_names[j]
                dist = np.linalg.norm(centroids[name1] - centroids[name2])
                print(f"{name1:10s} <-> {name2:10s}: {dist:.3f}")

        print("\n" + "="*50)

    def save_features_to_csv(self, reduced_features, labels, predictions,
                            valences, arousals, save_path='emotion_features.csv'):
        """Lưu features và labels vào CSV file"""
        df = pd.DataFrame({
            'dim1': reduced_features[:, 0],
            'dim2': reduced_features[:, 1],
            'true_label': labels,
            'true_emotion': [self.labels[l] for l in labels],
            'pred_label': predictions,
            'pred_emotion': [self.labels[p] for p in predictions],
            'valence': valences,
            'arousal': arousals,
            'correct': labels == predictions
        })

        df.to_csv(save_path, index=False)
        print(f"Saved features to {save_path}")


def main():
    """Hàm chính để chạy visualization"""

    # Cấu hình
    BATCH_SIZE = 64
    MAX_SAMPLES = 5000  # Giới hạn số mẫu để xử lý nhanh hơn (None = toàn bộ)
    USE_TRAIN_SET = False  # True = training set, False = validation set
    OUTPUT_DIR = 'emotion_visualizations'

    # QUAN TRỌNG: Thay đổi đường dẫn này tới thư mục chứa dữ liệu của bạn
    # Thư mục phải chứa training.csv hoặc validation.csv
    DATA_ROOT = "C:/Users/tam/Desktop/Data/FEG"  # Thay đổi đường dẫn này

    # Kiểm tra đường dẫn tồn tại
    if not os.path.exists(DATA_ROOT):
        print(f"ERROR: Data directory not found: {DATA_ROOT}")
        print("Please update DATA_ROOT in the main() function to point to your dataset folder.")
        return

    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup transform
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    print(f"Loading {'training' if USE_TRAIN_SET else 'validation'} dataset...")
    print(f"Data root: {DATA_ROOT}")

    try:
        dataset = Affectnet(is_train=USE_TRAIN_SET, transform=transform, root=DATA_ROOT)
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) == 0:
            print("\nERROR: Dataset is empty! Please check:")
            csv_file = "training.csv" if USE_TRAIN_SET else "validation.csv"
            print(f"1. {csv_file} exists in {DATA_ROOT}")
            print(f"2. CSV file contains valid image paths")
            print(f"3. Image files exist at the specified locations")
            return

    except Exception as e:
        print(f"\nERROR loading dataset: {e}")
        return

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )

    # Initialize visualizer
    visualizer = EmotionVectorVisualizer()

    # Extract features
    features, labels, predictions, valences, arousals = visualizer.extract_features(
        dataloader, max_samples=MAX_SAMPLES
    )

    # Plot emotion distribution
    visualizer.plot_emotion_distribution(
        labels,
        save_path=os.path.join(OUTPUT_DIR, 'emotion_distribution.png')
    )

    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        labels,
        predictions,
        save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    )

    # Reduce dimensions with t-SNE
    print("\nVisualizing with t-SNE...")
    reduced_tsne = visualizer.reduce_dimensions(features, method='tsne', n_components=2)

    visualizer.plot_2d_scatter(
        reduced_tsne,
        labels,
        predictions,
        save_path=os.path.join(OUTPUT_DIR, 'emotion_vectors_tsne_2d.png'),
        title='t-SNE Visualization of Emotion Vectors'
    )

    visualizer.plot_valence_arousal(
        reduced_tsne,
        valences,
        arousals,
        labels,
        save_path=os.path.join(OUTPUT_DIR, 'valence_arousal_tsne.png')
    )

    visualizer.analyze_emotion_clusters(reduced_tsne, labels)

    # Plot centroids only for t-SNE
    visualizer.plot_centroids_only(
        reduced_tsne,
        labels,
        save_path=os.path.join(OUTPUT_DIR, 'emotion_centroids_tsne.png')
    )

    visualizer.save_features_to_csv(
        reduced_tsne,
        labels,
        predictions,
        valences,
        arousals,
        save_path=os.path.join(OUTPUT_DIR, 'emotion_features_tsne.csv')
    )

    # Reduce dimensions with PCA
    print("\nVisualizing with PCA...")
    reduced_pca = visualizer.reduce_dimensions(features, method='pca', n_components=2)

    visualizer.plot_2d_scatter(
        reduced_pca,
        labels,
        predictions,
        save_path=os.path.join(OUTPUT_DIR, 'emotion_vectors_pca_2d.png'),
        title='PCA Visualization of Emotion Vectors'
    )

    visualizer.plot_valence_arousal(
        reduced_pca,
        valences,
        arousals,
        labels,
        save_path=os.path.join(OUTPUT_DIR, 'valence_arousal_pca.png')
    )

    # Plot centroids only for PCA
    visualizer.plot_centroids_only(
        reduced_pca,
        labels,
        save_path=os.path.join(OUTPUT_DIR, 'emotion_centroids_pca.png')
    )

    # 3D visualization with PCA
    print("\nCreating 3D visualization...")
    reduced_pca_3d = visualizer.reduce_dimensions(features, method='pca', n_components=3)
    visualizer.plot_3d_scatter(
        reduced_pca_3d,
        labels,
        save_path=os.path.join(OUTPUT_DIR, 'emotion_vectors_pca_3d.png')
    )

    print(f"\nAll visualizations saved to '{OUTPUT_DIR}' folder!")
    print("\nSummary:")
    print(f"  - Total samples: {len(labels)}")
    print(f"  - Model accuracy: {np.mean(labels == predictions):.2%}")
    print(f"  - Number of emotions: {len(np.unique(labels))}")


if __name__ == "__main__":
    main()
