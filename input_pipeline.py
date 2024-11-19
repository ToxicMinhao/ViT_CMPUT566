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