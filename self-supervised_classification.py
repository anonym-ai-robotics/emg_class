import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model, constraints
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configure GPU memory to prevent excessive allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # This prevents TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        
# Define class movement mapping
CLASS_MOVEMENTS = {
    0: [8, 20, 1, 5, 2, 7, 22, 10, 12, 17, 18, 19, 23, 4],  # Class1 (14 movements)
    1: [3],  # Class2 (1 movement)
    2: [6, 14, 15],  # Class3 (3 movements)
    3: [11, 13, 16, 9, 21],  # Class4 (5 movements)
    4: [0]           # Class5 (1 movement) Rest 
}
# ============================================================================
# STEP 1: SELF-SUPERVISED PRETRAINING
# ============================================================================

class SelfSupervisedDataGenerator(tf.keras.utils.Sequence):
    """
    Generator for self-supervised pretraining on movement classification
    Uses movement-based dataset structure: pretrain_dataset/p{i}/movement{k}/
    """
    
    def __init__(self, dataframe, img_size=(18, 1000), batch_size=64, 
                 shuffle=True, augment=True, noise_std=0.05):
        """
        Initialize self-supervised generator
        
        Args:
            dataframe: DataFrame with columns: filepath, movement, person
            img_size: Image dimensions (height, width)
            batch_size: Batch size
            shuffle: Whether to shuffle data each epoch
            augment: Whether to apply augmentation
            noise_std: Standard deviation for Gaussian noise augmentation
        """
        self.dataframe = dataframe.copy()
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.noise_std = noise_std
        
        # Store as numpy arrays for efficiency
        self.filepaths = self.dataframe['filepath'].values
        self.movements = self.dataframe['class'].values
        self.persons = self.dataframe['subject'].values
        
        # Get number of unique movements
        self.unique_movements = np.unique(self.movements)
        self.num_classes = len(self.unique_movements)
        
        # Create movement to index mapping
        self.movement_to_idx = {movement: idx for idx, movement in enumerate(sorted(self.unique_movements))}
        
        # Convert movement labels to indices
        self.labels = np.array([self.movement_to_idx[m] for m in self.movements])
        
        # Indices for shuffling
        self.indices = np.arange(len(self.filepaths))
        
        # Statistics
        print(f"Self-supervised generator initialized:")
        print(f"  Total samples: {len(self.filepaths)}")
        print(f"  Number of movements: {self.num_classes}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Augmentation: {'ON' if augment else 'OFF'}")
        
        # Shuffle initially if requested
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.filepaths) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get one batch of data"""
        # Get indices for this batch
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.filepaths))
        
        batch_indices = self.indices[start_idx:end_idx]
        
        batch_images = []
        batch_labels = []
        
        for i in batch_indices:
            # Load image
            img = self._load_image(self.filepaths[i])
            
            # Apply augmentation if enabled
            if self.augment:
                img = self._apply_augmentation(img)
            
            # Get label (one-hot encoded)
            label = self.labels[i]
            label_onehot = tf.keras.utils.to_categorical(label, num_classes=self.num_classes)
            
            batch_images.append(img)
            batch_labels.append(label_onehot)
        
        # Convert to numpy arrays
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        
        return batch_images, batch_labels
    
    def _load_image(self, filepath):
        """Load and normalize image"""
        # Load image
        img = tf.keras.preprocessing.image.load_img(
            filepath,
            color_mode='grayscale',
            target_size=self.img_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Normalize to [0, 1]
        img_normalized = img_array.astype('float32') / 255.0
        
        return img_normalized
    
    def _apply_augmentation(self, image):
        """Apply data augmentation"""
        # Additive Gaussian noise (primary augmentation)
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=image.shape)
        augmented = image + noise
        
        # Clip to valid range
        augmented = np.clip(augmented, 0, 1)

        if np.random.random() > 0.5:
            shift_amount = -1
            if np.random.random() > 0.5:
                shift_amount *= -1
            augmented = np.roll(augmented, shift_amount, axis=0)
        
        return augmented
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_class_distribution(self):
        """Get distribution of movements in dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = dict(zip([self._idx_to_movement(idx) for idx in unique], counts))
        return distribution
    
    def _idx_to_movement(self, idx):
        """Convert label index back to movement number"""
        # Find movement number for given index
        for movement, movement_idx in self.movement_to_idx.items():
            if movement_idx == idx:
                return movement
        return None
    def reset(self):
        "Updates indices after each epoch (e.g., for shuffling)"
        self.on_epoch_end()


# ============================================================================
# PRETRAINING MODEL AND PIPELINE
# ============================================================================

def EEGNet_Pretrain(nb_classes, Chans=18, Samples=1000, 
                    dropoutRate=0.1, kernLength=64, F1=16, 
                    D=2, F2=32, norm_rate=0.25, dropoutType='Dropout'):
    """
    EEGNet model for self-supervised pretraining (movement classification)
    """
    if dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    
    input1 = Input(shape=(Chans, Samples, 1))
    
    block1 = layers.Conv2D(F1, (1, kernLength), padding='same',
                           input_shape=(Chans, Samples, 1),
                           use_bias=False)(input1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.DepthwiseConv2D((Chans, 1), use_bias=False, 
                                    depth_multiplier=D,
                                    depthwise_constraint=constraints.MaxNorm(1.))(block1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('elu')(block1)
    block1 = layers.AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)
    
    block2 = layers.SeparableConv2D(F2, (1, 16),
                                   use_bias=False, padding='same')(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation('elu')(block2)
    block2 = layers.AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)
    
    flatten = layers.Flatten(name='flatten')(block2)
    hidden = layers.Dense(256, name='hidden')(flatten)
    dropout_dense = dropoutType(dropoutRate)(hidden)
    dense = layers.Dense(nb_classes, name='dense', 
                         kernel_constraint=constraints.MaxNorm(norm_rate))(dropout_dense)
    softmax = layers.Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

#====================================================================
# Pre-Training PipeLine
#====================================================================
class SelfSupervisedPretrainer:
    """
    Pipeline for self-supervised pretraining on movement classification
    """

    def __init__(self, dataset_path='pretrain_dataset', img_size=(18, 1000), 
             batch_size=32, num_classes=24):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.data_info = None
        self.subjects = []
        self.results = {}
        self.histories = {}

    def load_dataset_info(self):
        """
        Load dataset information and create DataFrame
        """
        print("Loading pretraining dataset information...")
        
        data = []
        persons = set()
        movements = set()
        
        # Walk through pretrain_dataset directory
        for root, dirs, files in os.walk(self.dataset_path):
            # Skip root directory
            if root == self.dataset_path:
                continue
            
            # Check if we're in a movement directory
            path_parts = root.split(os.sep)
            
            person_id = None
            movement_id = None
            
            for part in path_parts:
                if part.startswith('p') and part[1:].isdigit():
                    person_id = int(part[1:])
                elif part.startswith('movement') and part[8:].isdigit():
                    movement_id = int(part[8:])
                elif part.startswith('rest'):
                    movement_id = 0
                    
            if person_id is not None and movement_id is not None:
                persons.add(person_id)
                movements.add(movement_id)
                
                # Add each image file
                for file in files:
                    if file.endswith('.png'):
                        filepath = os.path.join(root, file)
                        data.append({
                            'filepath': filepath,
                            'subject': person_id,
                            'class': movement_id,
                            'filename': file
                        })
        
        # Create DataFrame
        self.data_info = pd.DataFrame(data)
        self.subjects = persons
        # Print statistics
        print(f"\nPretraining Dataset Summary:")
        print(f"  Total images: {len(self.data_info)}")
        print(f"  Persons: {len(persons)} ({sorted(persons)})")
        print(f"  Movements: {len(movements)} ({sorted(movements)})")
        
        # Print distribution
        print(f"\nDistribution per person:")
        for person in sorted(persons):
            person_data = self.data_info[self.data_info['subject'] == person]
            movement_counts = person_data['class'].value_counts().sort_index()
            print(f"  Person {person}: {len(person_data)} images, {len(movement_counts)} movements")
        
        print(f"\nDistribution per movement:")
        for movement in sorted(movements):
            movement_data = self.data_info[self.data_info['class'] == movement]
            person_counts = movement_data['subject'].value_counts().sort_index()
            print(f"  Movement {movement}: {len(movement_data)} images, {len(person_counts)} persons")
        
        return self.data_info
    
    def create_data_generators(self, train_df, val_df, test_df, fold_info):
        
        # Create generators
        train_gen = SelfSupervisedDataGenerator(
            dataframe=train_df,
            img_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=True,
            augment=True,
            noise_std=0.05
        )
        
        val_gen = SelfSupervisedDataGenerator(
            dataframe=val_df,
            img_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=False,
            augment=False,
            noise_std=0.0
        )
        
        test_gen = SelfSupervisedDataGenerator(
            dataframe=test_df,
            img_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=False,
            augment=False,
            noise_std=0.0
        )
        
        # Log class distribution
        if fold_info:
            print(f"\nFold {fold_info['fold']} - Subject {fold_info['test_subject']}:")
            print(f"  Training: {len(train_df)} images")
            print(f"  Validation: {len(val_df)} images")
            print(f"  Test: {len(test_df)} images")
            
            print("\n  Training class distribution:")
            train_counts = train_df['class'].value_counts().sort_index()
            for class_id, count in train_counts.items():
                print(f"    Class {int(class_id)+1}: {count} samples")
        
        return train_gen, val_gen, test_gen

    def run_loso_cv(self, epochs=100, learning_rate=1e-4, early_stop_patience=20):
        """
        Run LOSO-CV for self-supervised pretraining on 24 movement classes
        """
        if self.data_info is None:
            self.data_info = self.load_dataset_info()
        
        # Prepare data for LOSO
        X_indices = np.arange(len(self.data_info))
        y = self.data_info['class'].values  # Movement labels (0-23)
        groups = self.data_info['subject'].values
        
        # Initialize LOSO
        logo = LeaveOneGroupOut()
        
        # Store results
        self.results = {
            'subject_test': [],
            'test_accuracy': [],
            'test_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'movement_distribution': [],
            'predictions': [],
            'true_labels': [],
            'confusion_matrices': [],
            'groups_confusion_matrices': []
        }
        
        print("\n" + "="*60)
        print("STARTING LOSO-CV FOR SELF-SUPERVISED PRETRAINING")
        print("="*60)
        print(f"Number of movement classes: {self.num_classes}")
        print(f"Number of subjects: {len(self.subjects)}")
        
        fold = 1
        for train_idx, test_idx in logo.split(X_indices, y, groups):
            test_subject = groups[test_idx[0]]
            print(f"\n\nFold {fold}: Testing on Subject {test_subject}")
            print("-" * 60)
            
            # Split data
            train_df = self.data_info.iloc[train_idx]
            test_df = self.data_info.iloc[test_idx]

            # Create validation split from training data
            train_sub_df, val_df = train_test_split(
                train_df,
                test_size=0.1,  # 10% for validation
                stratify=train_df['class'],
                random_state=42
            )
            
            # Create data generators
            fold_info = {'fold': fold, 'test_subject': test_subject}
            train_gen, val_gen, test_gen = self.create_data_generators(
                train_sub_df, val_df, test_df, fold_info
            )
            
            # Create EEGNet model for pretraining (24 classes)
            model = EEGNet_Pretrain(
                nb_classes=self.num_classes,
                Chans=self.img_size[0],
                Samples=self.img_size[1],
                dropoutRate=0.1,
                kernLength=64,
                F1=16,
                D=2,
                F2=32,
                norm_rate=0.25,
                dropoutType='Dropout'
            )
            
            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_acc',
                    patience=early_stop_patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath=f'pretrained_subject_{test_subject}_best.h5',
                    monitor='val_acc',
                    save_best_only=True,
                    mode='max',
                    verbose=0
                ),
                ModelCheckpoint(
                    filepath=f'pretrained_subject_{test_subject}_final.h5',
                    save_best_only=False,
                    verbose=0
                )
            ]
            
            # Define metrics
            metrics = [
                tf.keras.metrics.CategoricalAccuracy(name='acc'),
                tf.keras.metrics.AUC(name='auc', from_logits=False),
            ]
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=metrics
            )
            
            # Model summary (first fold only)
            if fold == 1:
                model.summary()
            
            # Train model
            print(f"\nTraining on {len(train_sub_df)} samples, validating on {len(val_df)} samples...")
            
            history = model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
            
            # Store history
            self.histories[test_subject] = history.history
            
            # Evaluate on test set
            print(f"\nEvaluating on Subject {test_subject}...")
            
            # Evaluate
            test_results = model.evaluate(test_gen, verbose=0)
            
            # Get predictions
            test_gen.reset()
            y_pred_proba = model.predict(test_gen, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Get true labels
            test_gen.reset()
            y_true = []
            for i in range(len(test_gen)):
                _, batch_y = test_gen[i]
                y_true.extend(np.argmax(batch_y, axis=1))
            y_true = np.array(y_true)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score
            test_accuracy = accuracy_score(y_true, y_pred)
            
            # Calculate per-class accuracy
            per_class_acc = []
            for class_id in range(self.num_classes):
                mask = y_true == class_id
                if np.any(mask):
                    class_acc = np.mean(y_pred[mask] == class_id)
                    per_class_acc.append(class_acc)
                else:
                    per_class_acc.append(0.0)
            
            # Store results
            self.results['subject_test'].append(test_subject)
            self.results['test_accuracy'].append(test_accuracy)
            self.results['test_loss'].append(test_results[0])
            self.results['train_accuracy'].append(max(history.history['acc']))
            self.results['val_accuracy'].append(max(history.history.get('val_acc', [0])))
            self.results['movement_distribution'].append({
                'train': dict(train_sub_df['class'].value_counts()),
                'test': dict(test_df['class'].value_counts())
            })
            self.results['predictions'].append(y_pred)
            self.results['true_labels'].append(y_true)
            
            # Create and store confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
            self.results['confusion_matrices'].append(conf_matrix)

            pred_group, true_group = [], []
            for pred, label in zip(y_pred, y_true):
                pred_group.extend([group for group, lst in CLASS_MOVEMENTS.items() if pred in lst])
                true_group.extend([group for group, lst in CLASS_MOVEMENTS.items() if label in lst])
            # Confusion matrix
            cm = confusion_matrix(true_group, pred_group)
            self.results['groups_confusion_matrices'].append(cm)
            
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Test Loss: {test_results[0]:.4f}")
            print(f"  Test AUC: {test_results[2]:.4f}")
            
            # Save final model for this fold
            model.save(f'pretrained_model_fold_{fold}.h5')
            
            # Clean up
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            
            fold += 1

        self._print_comprehensive_results()
        self.plot_pretraining_results()
        return self.results
    
    def _print_comprehensive_results(self):
        """Print detailed LOSO-CV results for pretraining"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PRETRAINING LOSO-CV RESULTS")
        print("="*80)
        
        # Individual subject results
        print("\nSubject-wise Results:")
        print("-" * 80)
        print(f"{'Subject':<10} {'Test Acc':<12} {'Test Loss':<12} {'Val Acc':<12} {'Train Acc':<12}")
        print("-" * 80)
        
        for i, subject in enumerate(self.results['subject_test']):
            print(f"{subject:<10} "
                  f"{self.results['test_accuracy'][i]:<12.4f} "
                  f"{self.results['test_loss'][i]:<12.4f} "
                  f"{self.results['val_accuracy'][i]:<12.4f} "
                  f"{self.results['train_accuracy'][i]:<12.4f}")
        
        # Summary statistics
        print("\n" + "-" * 80)
        print("SUMMARY STATISTICS (24 Movement Classes):")
        print(f"Mean Test Accuracy: {np.mean(self.results['test_accuracy']):.4f} "
              f"(±{np.std(self.results['test_accuracy']):.4f})")
        print(f"Mean Test Loss: {np.mean(self.results['test_loss']):.4f} "
              f"(±{np.std(self.results['test_loss']):.4f})")
        print(f"Best Subject Accuracy: {max(self.results['test_accuracy']):.4f}")
        print(f"Worst Subject Accuracy: {min(self.results['test_accuracy']):.4f}")
        
        # Overall confusion matrix statistics
        all_true = np.concatenate(self.results['true_labels'])
        all_pred = np.concatenate(self.results['predictions'])
        overall_accuracy = np.mean(all_true == all_pred)
        print(f"\nOverall Combined Accuracy: {overall_accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report (Combined across all folds):")
        print(classification_report(all_true, all_pred, 
                                   target_names=[f'Movement{i+1}' for i in range(self.num_classes)]))
    
    def plot_pretraining_results(self, save_path='pretraining_results.png'):
        """Plot comprehensive pretraining results"""
        fig, axes = plt.subplots(2, 4, figsize=(18, 16))
        
        # 1. Test accuracy per subject
        subjects = self.results['subject_test']
        test_acc = self.results['test_accuracy']
        
        axes[0, 0].bar(range(len(subjects)), test_acc, color='skyblue')
        axes[0, 0].axhline(y=np.mean(test_acc), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(test_acc):.3f}')
        axes[0, 0].set_xlabel('Test Subject')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Test Accuracy per Left-Out Subject')
        axes[0, 0].set_xticks(range(len(subjects)))
        axes[0, 0].set_xticklabels(subjects)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training history for all folds (mean validation accuracy)
        max_epochs = 0
        for subject, history in self.histories.items():
            max_epochs = max(max_epochs, len(history.get('val_acc', [])))
        
        # Pad histories to same length
        val_acc_matrix = np.full((len(self.histories), max_epochs), np.nan)
        for i, (subject, history) in enumerate(self.histories.items()):
            val_acc = history.get('val_acc', [])
            val_acc_matrix[i, :len(val_acc)] = val_acc
        
        mean_val_acc = np.nanmean(val_acc_matrix, axis=0)
        std_val_acc = np.nanstd(val_acc_matrix, axis=0)
        
        epochs_range = np.arange(len(mean_val_acc))
        axes[0, 1].plot(epochs_range, mean_val_acc, 'b-', label='Mean Validation Accuracy')
        axes[0, 1].fill_between(epochs_range, 
                               mean_val_acc - std_val_acc,
                               mean_val_acc + std_val_acc,
                               alpha=0.2, color='blue')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Mean Validation Accuracy Across Folds')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Overall normalized confusion matrix
        all_true = np.concatenate(self.results['true_labels'])
        all_pred = np.concatenate(self.results['predictions'])
        overall_cm = confusion_matrix(all_true, all_pred, labels=range(self.num_classes))
        cm_normalized = overall_cm.astype('float') / overall_cm.sum(axis=1)[:, np.newaxis]    
        
        show_classes = min(25, self.num_classes)
        im = axes[0, 2].imshow(cm_normalized[:show_classes, :show_classes], 
                               cmap='Blues', vmin=0, vmax=1)
        axes[0, 2].set_xlabel('Predicted Movement')
        axes[0, 2].set_ylabel('True Movement')
        axes[0, 2].set_title(f'Normalized Confusion Matrix (First {show_classes} movements)')
        axes[0, 2].set_xticks(range(show_classes))
        axes[0, 2].set_yticks(range(show_classes))
        axes[0, 2].set_xticklabels([f'M{i+1}' for i in range(show_classes)], rotation=45)
        axes[0, 2].set_yticklabels([f'M{i+1}' for i in range(show_classes)])
        plt.colorbar(im, ax=axes[0, 2])

        # 4. groups confusion matrix
        group_confusion = np.mean(self.results['groups_confusion_matrices'], axis=0)
        group_confusion = group_confusion/np.sum(group_confusion, axis=1)[:, np.newaxis]
        sns.heatmap(group_confusion, annot=True, fmt='.2f', cmap='Blues',
           xticklabels=range(1, 6),
           yticklabels=range(1, 6), cbar=True)
        axes[0, 3].set_xlabel('Predicted Group')
        axes[0, 3].set_ylabel('True Group')
        axes[0, 3].set_title(f'Normalized Confusion Matrix (First {show_classes} Groups)')
        axes[0, 3].set_xticks(range(show_classes))
        axes[0, 3].set_yticks(range(show_classes))
        axes[0, 3].set_xticklabels([f'M{i+1}' for i in range(show_classes)], rotation=45)
        axes[0, 3].set_yticklabels([f'M{i+1}' for i in range(show_classes)])
        
        # 5. Training vs Validation accuracy for each fold
        for i, (subject, history) in enumerate(self.histories.items()):
            if 'acc' in history and 'val_acc' in history:
                min_len = min(len(history['acc']), len(history['val_acc']))
                axes[1, 0].plot(history['acc'][:min_len], alpha=0.3, color='blue')
                axes[1, 0].plot(history['val_acc'][:min_len], alpha=0.3, color='orange')
        
        # Plot mean
        train_acc_matrix = np.full((len(self.histories), max_epochs), np.nan)
        for i, (subject, history) in enumerate(self.histories.items()):
            train_acc = history.get('acc', [])
            train_acc_matrix[i, :len(train_acc)] = train_acc
        
        mean_train_acc = np.nanmean(train_acc_matrix, axis=0)
        mean_val_acc = np.nanmean(val_acc_matrix, axis=0)
        
        axes[1, 0].plot(mean_train_acc, 'b-', linewidth=2, label='Mean Train')
        axes[1, 0].plot(mean_val_acc, 'orange', linewidth=2, label='Mean Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Training vs Validation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 6. Loss curves
        for i, (subject, history) in enumerate(self.histories.items()):
            if 'loss' in history and 'val_loss' in history:
                min_len = min(len(history['loss']), len(history['val_loss']))
                axes[1, 1].plot(history['loss'][:min_len], alpha=0.3, color='blue')
                axes[1, 1].plot(history['val_loss'][:min_len], alpha=0.3, color='orange')
        
        # Plot mean loss
        train_loss_matrix = np.full((len(self.histories), max_epochs), np.nan)
        val_loss_matrix = np.full((len(self.histories), max_epochs), np.nan)
        for i, (subject, history) in enumerate(self.histories.items()):
            train_loss = history.get('loss', [])
            val_loss = history.get('val_loss', [])
            train_loss_matrix[i, :len(train_loss)] = train_loss
            val_loss_matrix[i, :len(val_loss)] = val_loss
        
        mean_train_loss = np.nanmean(train_loss_matrix, axis=0)
        mean_val_loss = np.nanmean(val_loss_matrix, axis=0)
        
        axes[1, 1].plot(mean_train_loss, 'b-', linewidth=2, label='Mean Train Loss')
        axes[1, 1].plot(mean_val_loss, 'orange', linewidth=2, label='Mean Val Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training and Validation Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 7. Per-movement accuracy heatmap (across subjects)
        # Calculate per-movement accuracy for each fold
        per_movement_acc = np.zeros((self.num_classes, len(subjects)))
        
        for i, (true_labels, pred_labels) in enumerate(zip(self.results['true_labels'], 
                                                          self.results['predictions'])):
            for movement_id in range(self.num_classes):
                mask = true_labels == movement_id
                if np.any(mask):
                    acc = np.mean(pred_labels[mask] == movement_id)
                    per_movement_acc[movement_id, i] = acc
        
        # Show heatmap for first 15 movements
        show_movements = min(15, self.num_classes)
        im = axes[1, 2].imshow(per_movement_acc[:show_movements, :], 
                               aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 2].set_xlabel('Subject (Test)')
        axes[1, 2].set_ylabel('Movement Class')
        axes[1, 2].set_title('Per-Movement Accuracy Heatmap')
        axes[1, 2].set_xticks(range(len(subjects)))
        axes[1, 2].set_xticklabels(subjects, rotation=45)
        axes[1, 2].set_yticks(range(show_movements))
        axes[1, 2].set_yticklabels([f'M{i+1}' for i in range(show_movements)])
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.suptitle('Self-Supervised Pretraining Results (24 Movement Classes)', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nAnalysis plots saved to: {save_path}")
        
        # Save overall model (average of all folds or best fold)
        best_idx = np.argmax(self.results['test_accuracy'])
        best_subject = self.results['subject_test'][best_idx]
        print(f"\nBest performing model: Subject {best_subject} with accuracy {self.results['test_accuracy'][best_idx]:.4f}")
        print(f"Pretrained model saved as: pretrained_model_fold_{best_idx+1}.h5")
    
    def save_final_pretrained_model(self, method='best', filepath='pretrained_eegnet_final.h5'):
        """
        Save final pretrained model
        method: 'best' (best fold), 'average' (average weights), or 'last' (last fold)
        """
        if method == 'best':
            best_idx = np.argmax(self.results['test_accuracy'])
            best_subject = self.results['subject_test'][best_idx]
            model_path = f'pretrained_subject_{best_subject}_best.h5'
            model = tf.keras.models.load_model(model_path)
            model.save(filepath)
            print(f"Saved best model (Subject {best_subject}) to: {filepath}")
        elif method == 'last':
            # Load and save last model
            last_subject = self.results['subject_test'][-1]
            model_path = f'pretrained_subject_{last_subject}_final.h5'
            model = tf.keras.models.load_model(model_path)
            model.save(filepath)
            print(f"Saved last model to: {filepath}")
        else:
            print(f"Method '{method}' not implemented. Please use 'best' or 'last'.")
        
        return filepath

def create_transfer_model(pretrained_model, num_classes=5, freeze_layers=True):
    """
    Create transfer learning model for 4-class classification
    
    Args:
        pretrained_model: Pretrained EEGNet model
        num_classes: Number of output classes (4 for your task)
        freeze_layers: Whether to freeze pretrained layers
    """
    # Create a new model with the same architecture
    transfer_model = tf.keras.models.clone_model(pretrained_model)
    
    # Copy weights from pretrained model
    transfer_model.set_weights(pretrained_model.get_weights())
    
    # Freeze layers if requested
    if freeze_layers:
        for layer in transfer_model.layers[:-1]:  # Keep last layer trainable
            layer.trainable = False
    
    x = transfer_model.layers[-1].output  # Get output from the last layer
    
    # Add new output layer
    new_output = layers.Dense(num_classes, activation='softmax', name='new_output')(x)
    
    # Create new model
    new_model = Model(inputs=transfer_model.input, outputs=new_output)
    
    # Compile with lower learning rate for fine-tuning
    new_model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['acc']
    )
    
    print("Transfer learning model created:")
    print(f"  Trainable layers: {sum([layer.trainable for layer in new_model.layers])}/{len(new_model.layers)}")
    
    return new_model

#==================================================================
# Step 2: 4-Class Data Generator
#==================================================================

class BalancedEMGDataGenerator(tf.keras.utils.Sequence):
    """
    Custom balanced data generator with class-aware augmentation
    Returns one-hot encoded labels for categorical_crossentropy
    """
    
    def __init__(self, dataframe, img_size=(18, 1000), batch_size=32, 
                 augmentation_factor=3.0, max_augmentation=10, class_weights=[1., 1., 1., 1., 1.]):
        """
        Initialize balanced generator
        
        Args:
            dataframe: DataFrame with filepaths and labels (integers 0-3)
            img_size: Target image size
            batch_size: Batch size
            augmentation_factor: How much to augment minority classes (max multiplier)
            max_augmentation: Maximum number of augmented samples per original image
        """
        self.dataframe = dataframe.copy()
        self.img_size = img_size
        self.batch_size = batch_size
        self.augmentation_factor = augmentation_factor
        self.max_augmentation = max_augmentation
        
        # Convert class column to integer if it's string
        if self.dataframe['class'].dtype == object:
            self.dataframe['class'] = self.dataframe['class'].astype(int)
        
        # Get class distribution
        self.class_distribution = self._compute_class_distribution()
        self.classes = sorted(self.dataframe['class'].unique())

        self.num_classes = len(self.classes)
        
        # Create balanced sampler
        self.sampler = self._create_balanced_sampler()
        
        # Pre-calculate class weights for loss function
        self.class_weights = class_weights
        print("Class distribution in dataset:")
        for class_id, count in sorted(self.class_distribution.items()):
            print(f"  Class {class_id}: {count} samples")
        
        print(f"\nClass weights for loss function: {self.class_weights}")
        
    def _compute_class_distribution(self):
        """Compute number of samples per class"""
        return dict(self.dataframe['class'].value_counts().sort_index())
    
    def _create_balanced_sampler(self):
        """
        Create a balanced sampler that oversamples minority classes
        and applies more augmentation to minority classes
        """
        sampler_data = []
        
        for class_id in self.classes:
            class_samples = self.dataframe[self.dataframe['class'] == class_id]
            class_count = len(class_samples)
            avg_class_count = np.mean(list(self.class_distribution.values()))
            
            # Calculate how much to augment this class
            # Minority classes get more augmentation
            if class_count < avg_class_count:
                # Calculate augmentation factor based on how rare the class is
                rarity_factor = avg_class_count / max(class_count, 1)
                augmentation_multiplier = min(self.augmentation_factor, rarity_factor)
                num_augmented = min(int(class_count * augmentation_multiplier - class_count), 
                                  class_count * self.max_augmentation)
            else:
                # Majority class gets less augmentation
                augmentation_multiplier = 1.0
                num_augmented = 0
            
            # Store sampler info
            sampler_data.append({
                'class_id': class_id,
                'original_samples': class_samples,
                'augmentation_multiplier': augmentation_multiplier,
                'num_augmented': num_augmented,
                'total_needed': class_count + num_augmented
            })
        
        return sampler_data
    
    def _apply_augmentation(self, image, augmentation_level):
        """
        Apply augmentation based on class rarity
        
        Args:
            image: Input image (18x1000)
            augmentation_level: How much augmentation to apply (0-1, higher for minority classes)
        
        Returns:
            Augmented image
        """
        # Base normalization
        img_normalized = image.astype('float32') / 255.0
        
        # Apply additive Gaussian noise - intensity based on augmentation_level
        if augmentation_level > 0:
            # Higher augmentation level = more noise for minority classes
            noise_std = 0.05
            noise = np.random.normal(loc=0.0, scale=noise_std, size=img_normalized.shape)
            img_noisy = img_normalized + noise
            
            # Clip to valid range
            img_noisy = np.clip(img_noisy, 0, 1)

            # Shift the image downwards and rotate
            if np.random.random() > 0.5:
                shift_amount = -1
                if np.random.random() > 0.5:
                    shift_amount *= -1
                img_noisy = np.roll(img_noisy, shift_amount, axis=0)

            return img_noisy

        
        
        return img_normalized
    
    def _load_and_augment_image(self, filepath, class_id, is_augmented=False):
        """Load image and apply augmentation if needed"""
        # Load image
        img = tf.keras.preprocessing.image.load_img(
            filepath, 
            color_mode='grayscale', 
            target_size=self.img_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Determine augmentation level based on class rarity
        class_info = next(item for item in self.sampler if item['class_id'] == class_id)
        avg_count = np.mean([item['total_needed'] for item in self.sampler])
        class_count = class_info['total_needed']
        
        # Rarer classes get more augmentation
        if class_count < avg_count:
            augmentation_level = 1.0 - (class_count / avg_count)
        else:
            augmentation_level = 0.0
        
        # Apply augmentation (more for augmented samples)
        if is_augmented:
            augmentation_level = min(1.0, augmentation_level * 1.5)
        
        img_augmented = self._apply_augmentation(img_array, augmentation_level)
        
        return img_augmented
    
    def __len__(self):
        """Number of batches per epoch"""
        total_samples = sum(item['total_needed'] for item in self.sampler)
        return int(np.ceil(total_samples / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_images = []
        batch_labels_int = []  # Store integer labels first
        
        # Calculate samples needed for this batch
        samples_per_class = max(1, self.batch_size // self.num_classes)
        
        for _ in range(samples_per_class):
            for class_info in self.sampler:
                if len(batch_images) >= self.batch_size:
                    break
                
                class_id = class_info['class_id']
                class_samples = class_info['original_samples']
                
                # Decide whether to use original or augmented sample
                use_augmented = np.random.random() < (class_info['num_augmented'] / class_info['total_needed'])
                
                if use_augmented and class_info['num_augmented'] > 0:
                    # Use augmented version of a random sample
                    random_sample = class_samples.sample(n=1).iloc[0]
                    img = self._load_and_augment_image(
                        random_sample['filepath'], 
                        class_id, 
                        is_augmented=True
                    )
                else:
                    # Use original sample
                    random_sample = class_samples.sample(n=1).iloc[0]
                    img = self._load_and_augment_image(
                        random_sample['filepath'], 
                        class_id, 
                        is_augmented=False
                    )
                
                # Store image and integer label
                batch_images.append(img)
                batch_labels_int.append(class_id)
        
        # Convert integer labels to one-hot encoding
        batch_labels = tf.keras.utils.to_categorical(batch_labels_int, num_classes=self.num_classes)
        
        # Shuffle the batch
        indices = np.arange(len(batch_images))
        np.random.shuffle(indices)
        
        batch_images = np.array([batch_images[i] for i in indices])
        batch_labels = np.array([batch_labels[i] for i in indices])

        return batch_images, batch_labels
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        pass

class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    """Weighted categorical crossentropy for imbalanced classes"""
    
    def __init__(self, class_weights, name='weighted_categorical_crossentropy'):
        super().__init__(name=name)
        self.class_weights = class_weights
    
    def call(self, y_true, y_pred):
        # Standard categorical crossentropy
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        
        # Apply class weights: convert one-hot y_true to class indices
        y_true_indices = tf.argmax(y_true, axis=1)
        weights = tf.gather(self.class_weights, tf.cast(y_true_indices, tf.int32))
        
        return ce * weights
        
# ============================================================================
# 4-Class Training PipeLine
# ============================================================================

class EEGNetImbalancedLOSOCVPipeline:
    """
    LOSO-CV Pipeline with imbalance handling for EMG dataset
    """
    
    def __init__(self, dataset_path='emg_dataset', img_size=(18, 1000), 
                 batch_size=32, class_weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.data_info = None
        self.subjects = []
        self.num_classes = 5 
        self.results = {}
        self.histories = {}
        self.class_weights = class_weights
        
    def load_dataset_info(self):
        """Load dataset and extract movement information"""
        print("Loading dataset with movement information...")
        
        data = []
        subjects = set()
        
        # Walk through dataset
        for root, dirs, files in os.walk(self.dataset_path):
            png_files = [f for f in files if f.endswith('.png') and (f.startswith('m') or (f.startswith('r')))]
            if png_files:
                path_parts = root.split(os.sep)
                
                subject_id = None
                class_id = None
                
                for part in path_parts:
                    if part.startswith('p') and part[1:].isdigit():
                        subject_id = int(part[1:])
                    elif part.startswith('class') and part[5:].isdigit():
                        class_id = int(part[5:])  # 1-indexed
                
                if subject_id is not None and class_id is not None:
                    subjects.add(subject_id)
                    
                    # Extract movement ID from filename
                    for file in png_files:
                        movement_id = None
                        if file.startswith('m'):
                            try:
                                # Extract number after 'm' and before 'p'
                                movement_part = file.split('p')[0][1:]
                                movement_id = int(movement_part)
                            except:
                                pass
                        
                        filepath = os.path.join(root, file)
                        data.append({
                            'filepath': filepath,
                            'subject': subject_id,
                            'class': class_id - 1,  # 0-indexed integer
                            'class_str': str(class_id - 1),  # String version for ImageDataGenerator
                            'movement_id': movement_id,
                            'filename': file
                        })
        
        self.data_info = pd.DataFrame(data)
        self.subjects = sorted(list(subjects))
        
        # Analyze class distribution
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*60)
        
        for class_idx in range(self.num_classes):
            class_data = self.data_info[self.data_info['class'] == class_idx]
            unique_movements = class_data['movement_id'].nunique()
            total_images = len(class_data)            
            expected_movements = len(CLASS_MOVEMENTS[class_idx])
            print(f"Class {class_idx+1}: {total_images} images, {unique_movements}/{expected_movements} movements")
        
        print(f"\nTotal images: {len(self.data_info)}")
        print(f"Subjects: {len(self.subjects)} ({self.subjects})")
        
        return self.data_info
    
    def create_data_generators(self, train_df, val_df, test_df, fold_info=None):
        """
        Create balanced generators for training, validation, and testing
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe  
            test_df: Testing dataframe
            fold_info: Information about current fold (for logging)
        """
        # Training generator with balancing (returns one-hot encoded labels)
        print("\nCreating balanced training generator...")
        train_generator = BalancedEMGDataGenerator(
            dataframe=train_df,
            img_size=self.img_size,
            batch_size=self.batch_size,
            augmentation_factor=5.0,  # Higher for more imbalance
            max_augmentation=20,
            class_weights = self.class_weights
        )
        
        # Validation generator - Use string labels for ImageDataGenerator
        val_df_str = val_df.copy()
        
        # Create ImageDataGenerator for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        val_generator = val_datagen.flow_from_dataframe(
            val_df_str,
            x_col='filepath',
            y_col='class_str',  # Use string version for categorical mode
            target_size=self.img_size,
            color_mode='grayscale',
            class_mode='categorical',  # Returns one-hot encoded labels
            batch_size=self.batch_size,
            shuffle=False,
            classes=[str(i) for i in range(self.num_classes)]  # Specify class order
        )
        
        # Test generator - Use string labels for ImageDataGenerator
        test_df_str = test_df.copy()
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_dataframe(
            test_df_str,
            x_col='filepath',
            y_col='class_str',  # Use string version for categorical mode
            target_size=self.img_size,
            color_mode='grayscale',
            class_mode='categorical',  # Returns one-hot encoded labels
            batch_size=self.batch_size,
            shuffle=False,
            classes=[str(i) for i in range(self.num_classes)]  # Specify class order
        )
        
        # Log class distribution
        if fold_info:
            print(f"\nFold {fold_info['fold']} - Subject {fold_info['test_subject']}:")
            print(f"  Training: {len(train_df)} images")
            print(f"  Validation: {len(val_df)} images")
            print(f"  Test: {len(test_df)} images")
            
            print("\n  Training class distribution:")
            train_counts = train_df['class'].value_counts().sort_index()
            for class_id, count in train_counts.items():
                print(f"    Class {int(class_id)+1}: {count} samples")
        
        return train_generator, val_generator, test_generator, train_generator.class_weights
    
    def run_loso_cv(self, epochs=100, learning_rate=1e-5, early_stop_patience=5):
        """
        Run LOSO-CV with imbalance handling
        """
        if self.data_info is None:
            self.load_dataset_info()
        
        # Prepare data for LOSO
        X_indices = np.arange(len(self.data_info))
        y = self.data_info['class'].values  # Integer labels
        groups = self.data_info['subject'].values
        
        # Initialize LOSO
        logo = LeaveOneGroupOut()
        
        # Store results
        self.results = {
            'subject_test': [],
            'test_accuracy': [],
            'test_accuracy_per_class': [],
            'test_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'class_distribution': [],
            'predictions': [],
            'true_labels': [],
            'confusion_matrices': []
        }
        
        print("\n" + "="*60)
        print("STARTING LOSO-CV WITH IMBALANCE HANDLING")
        print("="*60)
        
        fold = 1
        for train_idx, test_idx in logo.split(X_indices, y, groups):
            test_subject = groups[test_idx[0]]
            print(f"\n\nFold {fold}: Testing on Subject {test_subject}")
            print("-" * 60)
            
            # Split data
            train_df = self.data_info.iloc[train_idx]
            test_df = self.data_info.iloc[test_idx]

            # Create validation split
            from sklearn.model_selection import train_test_split
            train_sub_df, val_df = train_test_split(
                train_df,
                test_size=0.2,
                stratify=train_df['class'],
                random_state=42
            )
            
            # Create balanced generators
            fold_info = {'fold': fold, 'test_subject': test_subject}
            train_gen, val_gen, test_gen, class_weights = self.create_data_generators(
                train_sub_df, val_df, test_df, fold_info
            )
            
            # Create EEGNet model
            model = tf.keras.models.load_model('transfer_model_5class.h5')
            
            # Custom callback with class-aware metrics
            class ImbalanceAwareMetrics(Callback):
                def __init__(self, val_generator, class_names):
                    super().__init__()
                    self.val_generator = val_generator
                    self.class_names = class_names
                
                def on_epoch_end(self, epoch, logs=None):
                    # Calculate per-class accuracy on validation set
                    self.val_generator.reset()
                    y_true = []
                    y_pred = []
                    
                    for i in range(len(self.val_generator)):
                        batch_x, batch_y = self.val_generator[i]
                        preds = self.model.predict(batch_x, verbose=0)
                        # Convert one-hot to class indices
                        y_true.extend(np.argmax(batch_y, axis=1))
                        y_pred.extend(np.argmax(preds, axis=1))
                    
                    # Calculate per-class accuracy
                    class_accuracies = []
                    for class_id in range(len(self.class_names)):
                        mask = np.array(y_true) == class_id
                        if np.any(mask):
                            class_acc = np.mean(np.array(y_pred)[mask] == class_id)
                            class_accuracies.append(class_acc)
                            logs[f'val_acc_class_{class_id}'] = class_acc
                    
                    # Calculate balanced accuracy
                    balanced_acc = np.mean(class_accuracies) if class_accuracies else 0
                    logs['val_acc'] = balanced_acc
            
            # Define callbacks
            CALLBACKS = [
                EarlyStopping(
                    monitor='val_acc',
                    patience=early_stop_patience,
                    restore_best_weights=True,
                    mode='max'
                ),
                ModelCheckpoint(
                    filepath=f'eegnet_subject_{test_subject}_best.h5',
                    monitor='val_acc',
                    save_best_only=True,
                    mode='max',
                    verbose=0
                ),
                ImbalanceAwareMetrics(val_gen, [f'Class{i+1}' for i in range(self.num_classes)])
            ]
            
            # Define weighted loss function for categorical labels
            weighted_loss = WeightedCategoricalCrossentropy(class_weights)
            
            # Define metrics for categorical labels
            METRICS = [
                tf.keras.metrics.CategoricalAccuracy(name='acc'),
                tf.keras.metrics.AUC(name='auc', from_logits=False),
            ]

             # Compile model
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=weighted_loss,
                metrics=METRICS
            )
            
            # Model summary (first fold only)
            if fold == 1:
                model.summary()
            
            # Train model
            print(f"\nTraining with class weights: {class_weights}")
            
            # Train with validation labels
            history = model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=CALLBACKS,
                verbose=1
            )
            
            # Store history
            self.histories[test_subject] = history.history
            
            # Evaluate on test set
            print(f"\nEvaluating on Subject {test_subject}...")
            
            # Evaluate
            test_results = model.evaluate(test_gen, verbose=0)
            
            # Get predictions
            test_gen.reset()
            y_pred_proba = model.predict(test_gen, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Get true labels from test generator
            test_gen.reset()
            y_true = []
            for i in range(len(test_gen)):
                _, batch_y = test_gen[i]
                # Convert one-hot to class indices
                y_true.extend(np.argmax(batch_y, axis=1))
            y_true = np.array(y_true)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, balanced_accuracy_score
            test_accuracy = accuracy_score(y_true, y_pred)
            test_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
            
            # Calculate per-class accuracy
            per_class_acc = []
            for class_id in range(self.num_classes):
                mask = y_true == class_id
                if np.any(mask):
                    class_acc = np.mean(y_pred[mask] == class_id)
                    per_class_acc.append(class_acc)
                else:
                    per_class_acc.append(0.0)
            
            # Store results
            self.results['subject_test'].append(test_subject)
            self.results['test_accuracy'].append(test_accuracy)
            self.results['test_accuracy_per_class'].append(per_class_acc)
            self.results['test_loss'].append(test_results[0])
            self.results['train_accuracy'].append(max(history.history['acc']))
            self.results['val_accuracy'].append(max(history.history.get('val_acc', [0])))
            self.results['class_distribution'].append({
                'train': dict(train_sub_df['class'].value_counts()),
                'test': dict(test_df['class'].value_counts())
            })
            self.results['predictions'].append(y_pred)
            self.results['true_labels'].append(y_true)
            
            # Create and store confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
            self.results['confusion_matrices'].append(conf_matrix)
            
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Balanced Accuracy: {test_balanced_accuracy:.4f}")
            print(f"  Per-class Accuracy: {[f'{acc:.3f}' for acc in per_class_acc]}")
            
            # Clean up
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            
            fold += 1
        
        self._print_comprehensive_results()
        return self.results
    
    def _print_comprehensive_results(self):
        """Print detailed LOSO-CV results with imbalance analysis"""
        print("\n" + "="*80)
        print("COMPREHENSIVE LOSO-CV RESULTS WITH IMBALANCE HANDLING")
        print("="*80)
        
        # Individual subject results
        print("\nSubject-wise Results:")
        print("-" * 80)
        print(f"{'Subject':<10} {'Test Acc':<12} {'Balanced Acc':<12} {'Class Accuracies'}")
        print("-" * 80)
        
        for i, subject in enumerate(self.results['subject_test']):
            per_class_acc = self.results['test_accuracy_per_class'][i]
            balanced_acc = np.mean([acc for acc in per_class_acc if acc > 0])
            per_class_str = ' '.join([f'C{j+1}:{acc:.2f}' for j, acc in enumerate(per_class_acc)])
            
            print(f"{subject:<10} {self.results['test_accuracy'][i]:<12.4f} "
                  f"{balanced_acc:<12.4f} {per_class_str}")
        
        # Summary statistics
        print("\n" + "-" * 80)
        print("SUMMARY STATISTICS:")
        print(f"Mean Test Accuracy: {np.mean(self.results['test_accuracy']):.4f} "
              f"(±{np.std(self.results['test_accuracy']):.4f})")
        
        # Calculate per-class average accuracy across all folds
        avg_per_class_acc = np.mean(self.results['test_accuracy_per_class'], axis=0)
        print("\nAverage Per-Class Accuracy:")
        for class_id, acc in enumerate(avg_per_class_acc):
            movements = CLASS_MOVEMENTS[class_id]
            print(f"  Class {class_id+1} ({len(movements)} movements): {acc:.4f}")
        
        # Calculate balanced accuracy
        balanced_accuracies = [np.mean([acc for acc in accs if acc > 0]) 
                             for accs in self.results['test_accuracy_per_class']]
        print(f"\nMean Balanced Accuracy: {np.mean(balanced_accuracies):.4f}")
    
    def plot_imbalance_analysis(self, save_path='imbalance_analysis.png'):
        """Plot comprehensive imbalance analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Original class distribution
        class_counts = self.data_info['class'].value_counts().sort_index()
        axes[0, 0].bar(range(len(class_counts)), class_counts.values, 
                      color=['red', 'orange', 'yellow', 'green'])
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].set_title('Original Class Distribution')
        axes[0, 0].set_xticks(range(len(class_counts)))
        axes[0, 0].set_xticklabels([f'Class{i+1}' for i in class_counts.index])
        
        # 2. Test accuracy per subject
        subjects = self.results['subject_test']
        standard_acc = self.results['test_accuracy']
        balanced_acc = [np.mean([acc for acc in accs if acc > 0]) 
                       for accs in self.results['test_accuracy_per_class']]
        
        x = np.arange(len(subjects))
        width = 0.35
        axes[0, 1].bar(x - width/2, standard_acc, width, label='Standard Acc', alpha=0.7)
        axes[0, 1].bar(x + width/2, balanced_acc, width, label='Balanced Acc', alpha=0.7)
        axes[0, 1].set_xlabel('Left-Out Subject')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Standard vs Balanced Accuracy')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(subjects)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Per-class accuracy heatmap
        per_class_matrix = np.array(self.results['test_accuracy_per_class']).T
        im = axes[0, 2].imshow(per_class_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        axes[0, 2].set_xlabel('Subject (Test)')
        axes[0, 2].set_ylabel('Class')
        axes[0, 2].set_title('Per-Class Accuracy Heatmap')
        axes[0, 2].set_xticks(range(len(subjects)))
        axes[0, 2].set_xticklabels(subjects, rotation=45)
        axes[0, 2].set_yticks(range(self.num_classes))
        axes[0, 2].set_yticklabels([f'Class{i+1}' for i in range(self.num_classes)])
        plt.colorbar(im, ax=axes[0, 2])
        
        # 4. Per-class accuracy comparison (box plot)
        data_to_plot = [per_class_matrix[i] for i in range(self.num_classes)]
        bp = axes[1, 0].boxplot(data_to_plot, labels=[f'C{i+1}' for i in range(self.num_classes)])
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Per-Class Accuracy Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Confusion matrix
        all_true = np.concatenate(self.results['true_labels'])
        all_pred = np.concatenate(self.results['predictions'])
        overall_cm = confusion_matrix(all_true, all_pred, labels=range(self.num_classes))
        cm_normalized = overall_cm.astype('float') / overall_cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
           xticklabels=range(1, 6),
           yticklabels=range(1, 6), cbar=True)
        axes[1, 1].set_xlabel('Predicted Class')
        axes[1, 1].set_ylabel('True Class')
        axes[1, 1].set_title('Normalized Confusion Matrix')
        axes[1, 1].set_xticks(range(self.num_classes))
        axes[1, 1].set_yticks(range(self.num_classes))
        axes[1, 1].set_xticklabels([f'C{i+1}' for i in range(self.num_classes)])
        axes[1, 1].set_yticklabels([f'C{i+1}' for i in range(self.num_classes)])
        
        # 6. Minority class performance
        minority_acc = [accs[1] for accs in self.results['test_accuracy_per_class']]  # Class2
        axes[1, 2].plot(subjects, minority_acc, 'o-', linewidth=2, markersize=8)
        axes[1, 2].axhline(y=np.mean(minority_acc), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(minority_acc):.3f}')
        axes[1, 2].set_xlabel('Left-Out Subject')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].set_title('Minority Class (Class2) Performance')
        axes[1, 2].set_xticks(subjects)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Imbalance Analysis - EEGNet LOSO-CV Results', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nAnalysis plots saved to: {save_path}")

#===================================================================
# Main Program
#===================================================================
def main():
    """Main execution pipeline"""
    print("="*80)
    print("SELF-SUPERVISED PRETRAINING PIPELINE FOR EMG DATASET")
    print("="*80)
    
    ############ Step 1: Self-supervised pretraining ###################
    print("\nSTEP 1: Starting self-supervised pretraining...")
    pretrainer = SelfSupervisedPretrainer(
        dataset_path='pretrain_dataset',
        img_size=(18, 1000),
        batch_size=32,
        num_classes=24  # Adjust based on your actual number of movements
    )
    '''
    # Run pretraining
    results = pretrainer.run_loso_cv(
        epochs=200,  # Adjust as needed
        learning_rate=1e-4,
        early_stop_patience=20
    )    
    '''
    model = tf.keras.models.load_model('pretrained_subject_2_best.h5')        # best test accuracy model

    # Step 5: Create transfer learning model
    print("\nCreating transfer learning model for 5-class classification...")
    transfer_model = create_transfer_model(
        pretrained_model=model,
        num_classes=5,
        freeze_layers=True
    )
    
    # Save transfer model
    transfer_model.save('transfer_model_5class.h5')
    print("Transfer learning model saved to: transfer_model_5class.h5")
    
    ################# Step 2: 4-Class Training ####################
    print("\nSTEP 2: Starting 5-Class Training...")
    # Initialize pipeline
    pipeline = EEGNetImbalancedLOSOCVPipeline(
        dataset_path='emg_dataset',
        img_size=(18, 1000),
        batch_size=32,
        class_weights=[2.0, 2.0, 1.0, 1.5, 1.0]            # Best Class Loss Weights combination from grid search
    )
    
    # Load dataset
    pipeline.load_dataset_info()
    
    # Run LOSO-CV with imbalance handling
    results = pipeline.run_loso_cv(
        epochs=200,
        learning_rate=4e-4,
        early_stop_patience=20
    )
    
    # Plot comprehensive analysis
    save_path=f'Analysis.png'
    pipeline.plot_imbalance_analysis(save_path=save_path)
    #print('Test Accuracies per person: ', test_accuracies)'''
    return

if __name__ == "__main__":
    # Run full pipeline
    results = main()