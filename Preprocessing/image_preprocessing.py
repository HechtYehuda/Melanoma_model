from preprocessing_functions import denoise, rotate_melanoma_images
import os

# Denoising
denoise('train')
denoise('test')

# Augmentation
if 'raw_augmented_metadata.csv' in os.listdir('../processed_data/'):
            os.remove('../processed_data/raw_augmented_metadata.csv') # fresh start metadata augmentation
rotate_melanoma_images('90')
rotate_melanoma_images('180')
rotate_melanoma_images('270')