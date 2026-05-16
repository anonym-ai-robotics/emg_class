import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model, constraints
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Define class movement mapping
CLASS_MOVEMENTS = {
    0: [8, 20, 1, 5, 2, 7, 22, 10, 12, 17, 18, 19, 23, 4],  # Class1 (14 movements)
    1: [3],  # Class2 (1 movement)
    2: [6, 14, 15],  # Class3 (3 movements)
    3: [11, 13, 16, 9, 21]  # Class4 (5 movements)
}

def EEGNet(nb_classes, Chans=18, Samples=1000, 
           dropoutRate=0.1, kernLength=64, F1=16, 
           D=2, F2=32, norm_rate=0.25, dropoutType='Dropout'):
    """
    EEGNet model for EMG image classification (18x1000)
    """
    if dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    
    input1 = Input(shape=(Chans, Samples, 1))
    
    ##################################################################
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
    #hidden = layers.Dense(256, name='hidden')(flatten)
    #dropout_dense = dropoutType(dropoutRate)(hidden)
    dense = layers.Dense(nb_classes, name='dense', 
                         kernel_constraint=constraints.MaxNorm(norm_rate))(flatten)
    softmax = layers.Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

class BalancedEMGDataGenerator(tf.keras.utils.Sequence):
    """
    Custom balanced data generator with class-aware augmentation
    Returns one-hot encoded labels for categorical_crossentropy
    """
    
    def __init__(self, dataframe, img_size=(18, 1000), batch_size=32, 
                 augmentation_factor=3.0, max_augmentation=10):
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
        self.class_weights = self._compute_class_weights()
        
        print("Class distribution in dataset:")
        for class_id, count in sorted(self.class_distribution.items()):
            print(f"  Class {class_id}: {count} samples")
        
        print(f"\nClass weights for loss function: {self.class_weights}")
        
    def _compute_class_distribution(self):
        """Compute number of samples per class"""
        return dict(self.dataframe['class'].value_counts().sort_index())
    
    def _compute_class_weights(self):
        """Compute class weights for weighted loss function"""
        # Get class counts
        class_counts = list(self.class_distribution.values())
        
        # Compute weights: inverse of class frequency
        total_samples = sum(class_counts)
        weights = total_samples / (self.num_classes * np.array(class_counts))
        
        # Normalize weights so max weight is not too high
        weights = weights / np.max(weights) * 1.0  # Scale to max weight of 2.0

        weights[0] *= 3.0
        
        return {class_id: float(weight) for class_id, weight in zip(sorted(self.class_distribution.keys()), weights)}
    
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
            noise_std = 0.05#0.02 + 0.08 * augmentation_level  # 0.02 to 0.10 std
            noise = np.random.normal(loc=0.0, scale=noise_std, size=img_normalized.shape)
            img_noisy = img_normalized + noise
            
            # Clip to valid range
            img_noisy = np.clip(img_noisy, 0, 1)

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
        weights = tf.gather(list(self.class_weights.values()), tf.cast(y_true_indices, tf.int32))
        
        return ce * weights

class EEGNetImbalancedLOSOCVPipeline:
    """
    LOSO-CV Pipeline with imbalance handling for EMG dataset
    """
    
    def __init__(self, dataset_path='emg_dataset', img_size=(18, 1000), 
                 batch_size=32):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.data_info = None
        self.subjects = []
        self.num_classes = 4  
        self.results = {}
        self.histories = {}
        
    def load_dataset_info(self):
        """Load dataset and extract movement information"""
        print("Loading dataset with movement information...")
        
        data = []
        subjects = set()
        
        # Walk through dataset
        for root, dirs, files in os.walk(self.dataset_path):
            png_files = [f for f in files if f.endswith('.png') and (f.startswith('m'))]
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
            max_augmentation=20
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
    
    def run_loso_cv(self, epochs=100, learning_rate=5e-5, early_stop_patience=5):
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
            model = EEGNet(
                nb_classes=self.num_classes,
                Chans=self.img_size[0],
                Samples=self.img_size[1],
                dropoutRate=0.0,
                kernLength=64,
                F1=16,
                D=2,
                F2=32,
                norm_rate=0.25,
                dropoutType='Dropout'
            )
            
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
            
            # Compile model with weighted loss
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
        
        im = axes[1, 1].imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
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

def main():
    """Main function to run imbalanced LOSO-CV pipeline"""
    print("="*80)
    print("EEGNet LOSO-CV with Imbalance Handling - CORRECTED VERSION")
    print("="*80)
    
    # Initialize pipeline
    pipeline = EEGNetImbalancedLOSOCVPipeline(
        dataset_path='emg_dataset',
        img_size=(18, 1000),
        batch_size=64
    )
    
    # Load dataset
    pipeline.load_dataset_info()
    
    # Run LOSO-CV with imbalance handling
    results = pipeline.run_loso_cv(
        epochs=100,
        learning_rate=5e-5,
        early_stop_patience=20
    )
    
    # Plot comprehensive analysis
    pipeline.plot_imbalance_analysis()
    
    return pipeline

if __name__ == "__main__":
    # Run full pipeline
    pipeline = main()