# ============================================
# STEP 1: Import Libraries
# ============================================
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import time
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

# ============================================
# STEP 2: Load Dataset
# ============================================
(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

# Normalize pixel values (0-255 ko 0-1 mein convert)
x_train_full = x_train_full.astype('float32') / 255.0
x_test_full = x_test_full.astype('float32') / 255.0

print(f"Full Training data shape: {x_train_full.shape}")
print(f"Full Test data shape: {x_test_full.shape}")
print(f"Number of classes: {len(np.unique(y_train_full))}")

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# ============================================
# STEP 3: Define Model Function
# ============================================
def build_model(input_shape):
    """
    CNN Model banata hai -- input shape ke according
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================
# STEP 4: Image Resize Function
# ============================================
def resize_images(images, target_size):
    """
    Images ko resize karta hai target size mein
    """
    resized = tf.image.resize(images, [target_size, target_size])
    return resized.numpy()


# ============================================
# PART A: DIFFERENT IMAGE SIZES COMPARISON
# ============================================
print("\n" + "="*60)
print("PART A: DIFFERENT IMAGE SIZES COMPARISON")
print("="*60)

# Two different image sizes test karenge
image_sizes = [16, 32]  # 16x16 aur 32x32
# (CIFAR-10 original size 32x32 hai)

size_results = {}

for size in image_sizes:
    print(f"\n{'─'*50}")
    print(f"Training with image size: {size}x{size}")
    print(f"{'─'*50}")

    # Resize images
    if size == 32:
        # Original size, no resize needed
        x_train_resized = x_train_full
        x_test_resized = x_test_full
    else:
        print("Resizing images...")
        x_train_resized = resize_images(x_train_full, size)
        x_test_resized = resize_images(x_test_full, size)

    print(f"Train shape: {x_train_resized.shape}")
    print(f"Test shape: {x_test_resized.shape}")

    # Build model
    input_shape = (size, size, 3)
    model = build_model(input_shape)

    # Model summary for first size only
    if size == image_sizes[0]:
        model.summary()

    # Train model and measure time
    start_time = time.time()

    history = model.fit(
        x_train_resized, y_train_full,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test_resized, y_test_full, verbose=0)

    # Save results
    size_results[size] = {
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'history': history.history
    }

    print(f"\nResults for {size}x{size}:")
    print(f"  Test Accuracy:  {test_accuracy*100:.2f}%")
    print(f"  Training Time:  {training_time:.2f} seconds")

# ── Compare Image Size Results ──
print("\n" + "="*60)
print("IMAGE SIZE COMPARISON RESULTS")
print("="*60)
print(f"{'Image Size':<15} {'Accuracy':<15} {'Training Time':<15}")
print(f"{'─'*45}")
for size in image_sizes:
    r = size_results[size]
    print(f"{size}x{size:<10} {r['test_accuracy']*100:.2f}%{'':<8} {r['training_time']:.2f}s")


# ============================================
# PART A: VISUALIZATION - Image Size Comparison
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training Accuracy Curves
for size in image_sizes:
    axes[0].plot(size_results[size]['history']['accuracy'],
                label=f'{size}x{size} Train')
    axes[0].plot(size_results[size]['history']['val_accuracy'],
                '--', label=f'{size}x{size} Val')
axes[0].set_title('Training & Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Test Accuracy Comparison (Bar Chart)
sizes_labels = [f'{s}x{s}' for s in image_sizes]
accuracies = [size_results[s]['test_accuracy']*100 for s in image_sizes]
bars = axes[1].bar(sizes_labels, accuracies, color=['steelblue', 'coral'])
axes[1].set_title('Test Accuracy Comparison')
axes[1].set_ylabel('Accuracy (%)')
for bar, acc in zip(bars, accuracies):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
axes[1].grid(axis='y')

# Plot 3: Training Time Comparison (Bar Chart)
times = [size_results[s]['training_time'] for s in image_sizes]
bars = axes[2].bar(sizes_labels, times, color=['steelblue', 'coral'])
axes[2].set_title('Training Time Comparison')
axes[2].set_ylabel('Time (seconds)')
for bar, t in zip(bars, times):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')
axes[2].grid(axis='y')

plt.tight_layout()
plt.savefig('image_size_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graph saved: image_size_comparison.png")


# ============================================
# PART B: DATASET SIZE COMPARISON
# ============================================
print("\n" + "="*60)
print("PART B: DATASET SIZE COMPARISON")
print("="*60)

# Dataset sizes: Fewer images vs Full dataset
# Fewer = 5000 images (10% of full)
# Full  = 50000 images (100%)
dataset_configs = {
    'Small (5000 images)': 5000,
    'Full (50000 images)': 50000
}

dataset_results = {}

for config_name, num_samples in dataset_configs.items():
    print(f"\n{'─'*50}")
    print(f"Training with: {config_name}")
    print(f"{'─'*50}")

    # Select subset of training data
    if num_samples < len(x_train_full):
        # Randomly select images, balanced across classes
        indices = []
        samples_per_class = num_samples // 10  # 10 classes

        for class_idx in range(10):
            class_indices = np.where(y_train_full.flatten() == class_idx)[0]
            selected = np.random.choice(class_indices, samples_per_class,
                                        replace=False)
            indices.extend(selected)

        np.random.shuffle(indices)
        x_train_subset = x_train_full[indices]
        y_train_subset = y_train_full[indices]
    else:
        x_train_subset = x_train_full
        y_train_subset = y_train_full

    print(f"Training samples: {len(x_train_subset)}")
    print(f"Samples per class: {len(x_train_subset)//10}")

    # Build model (using original 32x32 size)
    model = build_model((32, 32, 3))

    # Train and measure time
    start_time = time.time()

    history = model.fit(
        x_train_subset, y_train_subset,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test_full, y_test_full, verbose=0)

    dataset_results[config_name] = {
        'num_samples': num_samples,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'history': history.history
    }

    print(f"\nResults for {config_name}:")
    print(f"  Test Accuracy:  {test_accuracy*100:.2f}%")
    print(f"  Training Time:  {training_time:.2f} seconds")

# ── Compare Dataset Size Results ──
print("\n" + "="*60)
print("DATASET SIZE COMPARISON RESULTS")
print("="*60)
print(f"{'Dataset':<25} {'Accuracy':<15} {'Training Time':<15}")
print(f"{'─'*55}")
for config_name in dataset_configs:
    r = dataset_results[config_name]
    print(f"{config_name:<25} {r['test_accuracy']*100:.2f}%{'':<8} {r['training_time']:.2f}s")


# ============================================
# PART B: VISUALIZATION - Dataset Size Comparison
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training Curves
colors = ['red', 'green']
for i, config_name in enumerate(dataset_configs):
    axes[0].plot(dataset_results[config_name]['history']['accuracy'],
                 color=colors[i], label=f'{config_name} Train')
    axes[0].plot(dataset_results[config_name]['history']['val_accuracy'],
                 '--', color=colors[i], label=f'{config_name} Val')
axes[0].set_title('Training & Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=8)
axes[0].grid(True)

# Plot 2: Accuracy Comparison
config_labels = list(dataset_configs.keys())
accuracies = [dataset_results[c]['test_accuracy']*100 for c in config_labels]
bars = axes[1].bar(config_labels, accuracies, color=['salmon', 'lightgreen'])
axes[1].set_title('Test Accuracy Comparison')
axes[1].set_ylabel('Accuracy (%)')
for bar, acc in zip(bars, accuracies):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
axes[1].grid(axis='y')

# Plot 3: Training Time Comparison
times = [dataset_results[c]['training_time'] for c in config_labels]
bars = axes[2].bar(config_labels, times, color=['salmon', 'lightgreen'])
axes[2].set_title('Training Time Comparison')
axes[2].set_ylabel('Time (seconds)')
for bar, t in zip(bars, times):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')
axes[2].grid(axis='y')

plt.tight_layout()
plt.savefig('dataset_size_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graph saved: dataset_size_comparison.png")


# ============================================
# FINAL SUMMARY REPORT
# ============================================
print("\n" + "="*60)
print("           FINAL SUMMARY REPORT")
print("="*60)

print("\n📊 PART A: IMAGE SIZE EFFECT")
print("─"*40)
for size in image_sizes:
    r = size_results[size]
    print(f"  {size}x{size}: Accuracy={r['test_accuracy']*100:.2f}%, "
          f"Time={r['training_time']:.1f}s")

print(f"\n  Accuracy Difference: "
      f"{abs(size_results[32]['test_accuracy'] - size_results[16]['test_accuracy'])*100:.2f}%")

print("\n📊 PART B: DATASET SIZE EFFECT")
print("─"*40)
for config_name in dataset_configs:
    r = dataset_results[config_name]
    print(f"  {config_name}: Accuracy={r['test_accuracy']*100:.2f}%, "
          f"Time={r['training_time']:.1f}s")