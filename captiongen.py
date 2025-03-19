import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import multiprocessing


# Initialize model once (will be reused in workers)
global_processor = None
global_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def init_worker():
    """Initialize model and processor for each worker"""
    global global_processor, global_model
    if global_processor is None:
        global_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    if global_model is None:
        global_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def process_image(image_path, batch_size=8):
    """Process images in batches for GPU efficiency"""
    try:
        # Load batch of images
        images = [Image.open(path).convert('RGB') for path in image_path]
        
        # Process batch
        inputs = global_processor(images=images, return_tensors="pt", padding=True).to(device)
        outputs = global_model.generate(**inputs)
        
        # Decode all captions in batch
        return [global_processor.decode(output, skip_special_tokens=True) for output in outputs]
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return [None] * len(image_path)

def process_directory(root_dir, batch_size=8, num_workers=None):
    images_root = os.path.join(root_dir, 'images')
    captions_root = os.path.join(root_dir, 'captions')
    
    # Collect all image paths
    image_paths = []
    for dirpath, _, filenames in os.walk(images_root):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(dirpath, filename))

    # Create batches of paths
    batches = [image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]
    
    # Initialize parallel processing
    num_workers = num_workers or min(cpu_count(), 8)  # Limit to 8 workers by default
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        # Process batches with progress bar
        pbar = tqdm(total=len(image_paths), desc="Generating captions", unit="img")
        
        for batch_idx, batch_paths in enumerate(batches):
            captions = pool.apply_async(process_image, (batch_paths, batch_size)).get()
            
            for path, caption in zip(batch_paths, captions):
                if caption:
                    # Create output directory structure
                    relative_dir = os.path.relpath(os.path.dirname(path), images_root)
                    caption_dir = os.path.join(captions_root, relative_dir)
                    os.makedirs(caption_dir, exist_ok=True)
                    
                    # Save caption
                    caption_filename = os.path.splitext(os.path.basename(path))[0] + '.txt'
                    caption_path = os.path.join(caption_dir, caption_filename)
                    
                    with open(caption_path, 'w') as f:
                        f.write(caption)
                
            pbar.update(len(batch_paths))
            pbar.set_postfix({
                "batches": f"{batch_idx+1}/{len(batches)}",
                "workers": num_workers,
                "batch_size": batch_size
            })
        
        pbar.close()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    root_directory = '/raid/biplab/taha/milaid2'
    
    # Tune these parameters based on your hardware
    process_directory(
        root_directory,
        batch_size=16,  # Reduce if you get GPU OOM errors
        num_workers=12  # Match your CPU core count
    )
    
    print("Caption generation complete!")