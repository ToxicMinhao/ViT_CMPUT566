import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess_image(image, label, image_size):
    """
    Preprocesses an image for training or evaluation.
    
    Args:
        image: Input image.
        label: Corresponding label.
        image_size: Target image size (height, width).
        
    Returns:
        A tuple of preprocessed image and one-hot label.
    """
    # Resize the image to the target size
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    image = (image - 0.5) * 2.0  # Scale to [-1, 1]

    return image, tf.one_hot(label, depth=1000)  # Assuming 1000 classes (ImageNet)

def augment_image(image, label, image_size):
    """
    Augments the image with random transformations for training.
    
    Args:
        image: Input image.
        label: Corresponding label.
        image_size: Target image size (height, width).
        
    Returns:
        A tuple of augmented image and label.
    """
    # Random crop and resize
    image = tf.image.random_crop(image, [image_size, image_size, 3])
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.random_flip_left_right(image)

    # Normalize and scale
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - 0.5) * 2.0  # Scale to [-1, 1]

    return image, label

def get_dataset(dataset_name, split, image_size, batch_size, is_training):
    """
    Loads and preprocesses a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'imagenet2012').
        split: Dataset split ('train' or 'validation').
        image_size: Target image size.
        batch_size: Batch size for training or evaluation.
        is_training: Whether the dataset is for training or evaluation.
        
    Returns:
        A tf.data.Dataset object.
    """
    # Load dataset
    ds = tfds.load(dataset_name, split=split, as_supervised=True)

    # Apply preprocessing and augmentations
    if is_training:
        ds = ds.shuffle(1000).repeat()  # Shuffle and repeat for training
        ds = ds.map(
            lambda img, lbl: augment_image(img, lbl, image_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda img, lbl: preprocess_image(img, lbl, image_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    # Batch and prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds