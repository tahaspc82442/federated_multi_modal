from PIL import Image

# Load the image
image_path = "/raid/biplab/taha/Mlrs/images/airplane/airplane_8.jpg"
img = Image.open(image_path)

# Check dimensions: returns (width, height)
width, height = img.size
print("Width:", width)
print("Height:", height)

# Check the color mode to infer channels
print("Color mode:", img.mode)
# Common modes: "RGB" (3 channels), "RGBA" (4 channels), "L" (grayscale, 1 channel), etc.
