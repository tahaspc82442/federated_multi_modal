import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate_caption(image_path):
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_directory(root_dir):
    images_root = os.path.join(root_dir, '2750')
    captions_root = os.path.join(root_dir, 'captions')
    
    for dirpath, _, filenames in os.walk(images_root):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Process image
                image_path = os.path.join(dirpath, filename)
                caption = generate_caption(image_path)
                
                if caption:
                    # Create corresponding captions path
                    relative_path = os.path.relpath(dirpath, images_root)
                    caption_dir = os.path.join(captions_root, relative_path)
                    os.makedirs(caption_dir, exist_ok=True)
                    
                    # Save caption as text file
                    caption_filename = os.path.splitext(filename)[0] + '.txt'
                    caption_path = os.path.join(caption_dir, caption_filename)
                    
                    with open(caption_path, 'w') as f:
                        f.write(caption)

if __name__ == "__main__":
    root_directory = '/raid/biplab/taha/eurosat'  # Change this to your root directory if needed
    process_directory(root_directory)
    print("Caption generation complete!")