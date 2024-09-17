import pygame
import os

def scale_image(image_path, target_size):
    # Load the image
    original_image = pygame.image.load(image_path)
    
    # Scale the image
    scaled_image = pygame.transform.scale(original_image, target_size)
    
    # Save the scaled image
    pygame.image.save(scaled_image, image_path)
    
    print(f"Scaled {image_path} to {target_size}")

def preprocess_images(width, height):
    # Ensure the assets directory exists
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    # Define image paths and target sizes
    images = {
        'assets/bird.png': (50, 50),
        'assets/pipe.png': (70, height),  # Changed to full height
        'assets/background.png': (width, height)
    }
    
    # Scale each image
    for image_path, target_size in images.items():
        if os.path.exists(image_path):
            scale_image(image_path, target_size)
        else:
            print(f"Warning: {image_path} not found. Skipping.")

if __name__ == "__main__":
    pygame.init()
    preprocess_images(400, 600)  # Use the same WIDTH and HEIGHT as in flappy_bird.py
    pygame.quit()