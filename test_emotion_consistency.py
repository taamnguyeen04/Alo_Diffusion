import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

# Đọc ảnh content.png
img = Image.open('content.png')
img_array = np.array(img)

print(f"Image shape: {img_array.shape}")

# Tách ảnh thành 4 hàng (4 người) và 8 cột (original + 7 emotions)
num_rows = 4  # 4 người
num_cols = 8  # 1 original + 7 emotions

height, width = img_array.shape[:2]
cell_height = height // num_rows
cell_width = width // num_cols

# Tên các cảm xúc theo thứ tự cột
emotion_names = ['Original', 'Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

# Tách từng ảnh nhỏ
images = []
for row in range(num_rows):
    row_images = []
    for col in range(num_cols):
        y1 = row * cell_height
        y2 = (row + 1) * cell_height
        x1 = col * cell_width
        x2 = (col + 1) * cell_width

        cell_img = img_array[y1:y2, x1:x2]
        row_images.append(cell_img)
    images.append(row_images)

print(f"Extracted {num_rows} rows x {num_cols} columns of images")
print(f"Each cell size: {cell_height} x {cell_width}")

# Tính delta (emotion - original) cho mỗi người và mỗi cảm xúc
deltas = []
for person_idx in range(num_rows):
    original = images[person_idx][0].astype(np.float32)
    person_deltas = []

    for emotion_idx in range(1, num_cols):  # Bỏ qua cột original
        emotion_img = images[person_idx][emotion_idx].astype(np.float32)
        delta = emotion_img - original
        person_deltas.append(delta)

    deltas.append(person_deltas)

print(f"\nCalculated deltas for {num_rows} persons x {num_cols-1} emotions")

# So sánh deltas giữa các người cho mỗi cảm xúc
print("\n" + "="*80)
print("PHÂN TÍCH SỰ GIỐNG NHAU CỦA EMOTION VECTORS GIỮA CÁC NGƯỜI")
print("="*80)

# 1. Tính độ tương đồng giữa các delta vectors
print("\n1. Correlation giữa emotion deltas của các cặp người:\n")

for emotion_idx in range(len(emotion_names) - 1):  # Bỏ qua Original
    emotion_name = emotion_names[emotion_idx + 1]
    print(f"\n{emotion_name}:")
    print("-" * 60)

    # Flatten deltas để tính correlation
    flattened_deltas = [deltas[p][emotion_idx].flatten() for p in range(num_rows)]

    # Tính correlation matrix giữa các người
    correlations = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(num_rows):
            corr = np.corrcoef(flattened_deltas[i], flattened_deltas[j])[0, 1]
            correlations[i, j] = corr
            if i < j:  # Chỉ in nửa trên của matrix
                print(f"  Person {i+1} vs Person {j+1}: {corr:.4f}")

    avg_corr = np.mean(correlations[np.triu_indices(num_rows, k=1)])
    print(f"  → Average correlation: {avg_corr:.4f}")

# 2. Tính Mean Squared Difference giữa các delta vectors
print("\n" + "="*80)
print("\n2. Mean Squared Difference (MSD) giữa emotion deltas:\n")

for emotion_idx in range(len(emotion_names) - 1):
    emotion_name = emotion_names[emotion_idx + 1]
    print(f"\n{emotion_name}:")
    print("-" * 60)

    msds = []
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            delta_i = deltas[i][emotion_idx]
            delta_j = deltas[j][emotion_idx]

            msd = np.mean((delta_i - delta_j) ** 2)
            msds.append(msd)
            print(f"  Person {i+1} vs Person {j+1}: {msd:.4f}")

    avg_msd = np.mean(msds)
    print(f"  → Average MSD: {avg_msd:.4f}")

# 3. Visualize: So sánh histogram của delta values
print("\n" + "="*80)
print("\n3. Generating visualization plots...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Distribution of Delta Values for Each Emotion Across Different Persons',
             fontsize=16, fontweight='bold')

for emotion_idx in range(len(emotion_names) - 1):
    ax = axes[emotion_idx // 4, emotion_idx % 4]
    emotion_name = emotion_names[emotion_idx + 1]

    for person_idx in range(num_rows):
        delta_values = deltas[person_idx][emotion_idx].flatten()
        ax.hist(delta_values, bins=50, alpha=0.5, label=f'Person {person_idx+1}')

    ax.set_title(emotion_name, fontweight='bold')
    ax.set_xlabel('Delta Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('emotion_delta_distributions.png', dpi=150, bbox_inches='tight')
print("  → Saved: emotion_delta_distributions.png")

# 4. Visualize: Heatmap của mean delta cho mỗi emotion và person
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Mean Delta Values (RGB channels) for Each Emotion and Person',
             fontsize=16, fontweight='bold')

for emotion_idx in range(len(emotion_names) - 1):
    ax = axes[emotion_idx // 4, emotion_idx % 4]
    emotion_name = emotion_names[emotion_idx + 1]

    # Tính mean delta cho mỗi channel và mỗi person
    mean_deltas = np.zeros((num_rows, 3))  # 4 persons x 3 RGB channels
    for person_idx in range(num_rows):
        mean_deltas[person_idx] = np.mean(deltas[person_idx][emotion_idx], axis=(0, 1))

    # Vẽ heatmap
    im = ax.imshow(mean_deltas.T, cmap='RdBu_r', aspect='auto', vmin=-50, vmax=50)
    ax.set_title(emotion_name, fontweight='bold')
    ax.set_xlabel('Person')
    ax.set_ylabel('RGB Channel')
    ax.set_xticks(range(num_rows))
    ax.set_xticklabels([f'P{i+1}' for i in range(num_rows)])
    ax.set_yticks(range(3))
    ax.set_yticklabels(['R', 'G', 'B'])

    # Thêm giá trị text
    for i in range(num_rows):
        for j in range(3):
            text = ax.text(i, j, f'{mean_deltas[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('emotion_mean_deltas_heatmap.png', dpi=150, bbox_inches='tight')
print("  → Saved: emotion_mean_deltas_heatmap.png")

# 5. Correlation heatmap giữa các emotion deltas
print("\n" + "="*80)
print("\n4. Generating emotion correlation heatmap...")

# Tính correlation matrix cho mỗi cặp (emotion, person)
n_emotions = len(emotion_names) - 1
all_deltas_flat = []
labels = []

for person_idx in range(num_rows):
    for emotion_idx in range(n_emotions):
        delta_flat = deltas[person_idx][emotion_idx].flatten()
        all_deltas_flat.append(delta_flat)
        labels.append(f"P{person_idx+1}-{emotion_names[emotion_idx+1][:3]}")

# Tính correlation matrix
n_total = len(all_deltas_flat)
corr_matrix = np.zeros((n_total, n_total))
for i in range(n_total):
    for j in range(n_total):
        corr_matrix[i, j] = np.corrcoef(all_deltas_flat[i], all_deltas_flat[j])[0, 1]

# Vẽ heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, xticklabels=labels, yticklabels=labels,
            cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Emotion Deltas Across Persons',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Person-Emotion', fontsize=12)
plt.ylabel('Person-Emotion', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('emotion_correlation_matrix.png', dpi=150, bbox_inches='tight')
print("  → Saved: emotion_correlation_matrix.png")

# 6. Summary statistics
# 7. Xuất ảnh delta 4 hàng x 7 cột
print("\n" + "="*80)
print("\n5. Generating delta images visualization...")

# Tạo ảnh delta grid
fig, axes = plt.subplots(num_rows, len(emotion_names)-1, figsize=(21, 12))
fig.suptitle('Delta Images (Emotion - Original) - 4 Persons x 7 Emotions',
             fontsize=16, fontweight='bold')

for person_idx in range(num_rows):
    for emotion_idx in range(len(emotion_names) - 1):
        ax = axes[person_idx, emotion_idx]

        # Lấy delta và normalize để hiển thị
        delta = deltas[person_idx][emotion_idx]

        # Normalize delta về range [0, 255] để hiển thị
        # Delta có thể âm hoặc dương, ta cần scale về [0, 255]
        delta_normalized = delta.copy()
        # Shift về [0, 255] với gray = 128 là không thay đổi
        delta_normalized = delta_normalized + 128
        delta_normalized = np.clip(delta_normalized, 0, 255).astype(np.uint8)

        ax.imshow(delta_normalized)
        ax.axis('off')

        # Chỉ thêm title cho hàng đầu
        if person_idx == 0:
            ax.set_title(emotion_names[emotion_idx + 1], fontsize=12, fontweight='bold')

        # Thêm label cho cột đầu
        if emotion_idx == 0:
            ax.set_ylabel(f'Person {person_idx + 1}', fontsize=12, fontweight='bold', rotation=0,
                         labelpad=40, va='center')

plt.tight_layout()
plt.savefig('delta_images_grid.png', dpi=150, bbox_inches='tight')
print("  → Saved: delta_images_grid.png")
print("    (Gray=128 means no change, brighter=positive delta, darker=negative delta)")

# 8. Xuất ảnh delta dạng heatmap (dễ nhìn hơn)
fig, axes = plt.subplots(num_rows, len(emotion_names)-1, figsize=(21, 12))
fig.suptitle('Delta Heatmaps (Absolute Difference from Original) - 4 Persons x 7 Emotions',
             fontsize=16, fontweight='bold')

# Tìm max delta để scale đồng nhất
max_delta = 0
for person_idx in range(num_rows):
    for emotion_idx in range(len(emotion_names) - 1):
        delta = deltas[person_idx][emotion_idx]
        max_delta = max(max_delta, np.abs(delta).max())

for person_idx in range(num_rows):
    for emotion_idx in range(len(emotion_names) - 1):
        ax = axes[person_idx, emotion_idx]

        # Lấy delta và tính magnitude (độ lớn thay đổi)
        delta = deltas[person_idx][emotion_idx]
        delta_magnitude = np.sqrt(np.sum(delta**2, axis=2))  # L2 norm across RGB channels

        im = ax.imshow(delta_magnitude, cmap='hot', vmin=0, vmax=max_delta)
        ax.axis('off')

        # Chỉ thêm title cho hàng đầu
        if person_idx == 0:
            ax.set_title(emotion_names[emotion_idx + 1], fontsize=12, fontweight='bold')

        # Thêm label cho cột đầu
        if emotion_idx == 0:
            ax.set_ylabel(f'Person {person_idx + 1}', fontsize=12, fontweight='bold', rotation=0,
                         labelpad=40, va='center')

# Thêm colorbar chung
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Delta Magnitude')

plt.savefig('delta_magnitude_heatmap.png', dpi=150, bbox_inches='tight')
print("  → Saved: delta_magnitude_heatmap.png")
print("    (Shows the magnitude of change - brighter = more change)")

print("\n" + "="*80)
print("\nTÓM TẮT KẾT QUẢ:")
print("="*80)

# Tính correlation trung bình cho cùng emotion giữa các người khác nhau
same_emotion_corrs = []
for emotion_idx in range(n_emotions):
    person_deltas = [deltas[p][emotion_idx].flatten() for p in range(num_rows)]

    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            corr = np.corrcoef(person_deltas[i], person_deltas[j])[0, 1]
            same_emotion_corrs.append(corr)

print(f"\nCorrelation trung bình giữa CÙNG cảm xúc của các người khác nhau:")
print(f"  {np.mean(same_emotion_corrs):.4f} ± {np.std(same_emotion_corrs):.4f}")

# Tính correlation trung bình cho khác emotion của cùng người
diff_emotion_corrs = []
for person_idx in range(num_rows):
    for i in range(n_emotions):
        for j in range(i + 1, n_emotions):
            delta_i = deltas[person_idx][i].flatten()
            delta_j = deltas[person_idx][j].flatten()
            corr = np.corrcoef(delta_i, delta_j)[0, 1]
            diff_emotion_corrs.append(corr)

print(f"\nCorrelation trung bình giữa KHÁC cảm xúc của cùng người:")
print(f"  {np.mean(diff_emotion_corrs):.4f} ± {np.std(diff_emotion_corrs):.4f}")

print("\n" + "="*80)
print("\nKẾT LUẬN:")
print("="*80)
if np.mean(same_emotion_corrs) > 0.7:
    print("\n✓ Mô hình áp dụng emotion vectors KHÁ GIỐNG NHAU cho các người khác nhau")
    print("  → Cùng một cảm xúc tạo ra những thay đổi tương tự trên các khuôn mặt khác nhau")
elif np.mean(same_emotion_corrs) > 0.4:
    print("\n~ Mô hình áp dụng emotion vectors TƯƠNG ĐỐI GIỐNG NHAU cho các người khác nhau")
    print("  → Có sự tương đồng nhất định nhưng vẫn có điều chỉnh theo từng người")
else:
    print("\n✗ Mô hình áp dụng emotion vectors KHÁC NHAU cho mỗi người")
    print("  → Mô hình điều chỉnh cảm xúc dựa trên đặc điểm riêng của từng khuôn mặt")

print("\n" + "="*80)
