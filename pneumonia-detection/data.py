import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DATASET_PATH = os.path.join(os.path.expanduser("~"), ".kaggle", "datasets", "paultimothymooney", "chest-xray-pneumonia", "chest_xray")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def download_and_extract_dataset():
    """
    Downloads and extracts the dataset from Kaggle.
    Requires kaggle.json to be placed in ~/.kaggle/
    """
    import kaggle
    try:
        logger.info("Downloading dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', path=os.path.dirname(DATASET_PATH), unzip=True)
        logger.info("Dataset downloaded and extracted successfully.")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.info("Please ensure your kaggle.json is correctly set up.")
        # As a fallback, we assume the dataset is already in the expected path.
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH} and download failed.")

def create_data_generators():
    """
    Creates data generators for training, validation, and testing.
    Applies data augmentation to the training data.
    """
    if not os.path.exists(DATASET_PATH):
        download_and_extract_dataset()

    # Data augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescale for validation and test sets
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_dir = os.path.join(DATASET_PATH, 'train')
    val_dir = os.path.join(DATASET_PATH, 'val')
    test_dir = os.path.join(DATASET_PATH, 'test')

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED
    )

    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False, # Important for evaluation
        seed=SEED
    )

    return train_generator, validation_generator, test_generator

if __name__ == '__main__':
    logger.info("Creating data generators...")
    train_gen, val_gen, test_gen = create_data_generators()
    logger.info(f"Class indices: {train_gen.class_indices}")
    logger.info("Data generators created successfully.")
    # Example of how to get one batch
    x_batch, y_batch = next(train_gen)
    logger.info(f"Sample batch shape: {x_batch.shape}")
    logger.info(f"Sample labels shape: {y_batch.shape}")
