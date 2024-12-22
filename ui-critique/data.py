from io import BytesIO
from datasets import load_dataset


class DataLoader:
    def __init__(self, rows=1000):
        ds = load_dataset("mrtoy/mobile-ui-design")
        self.images = []
        for image in ds['train'][:rows]['image']:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")  # You can also use "JPEG" or other formats
            image_bytes = buffered.getvalue()  # Get the byte data from the buffer
            self.images.append(image_bytes)
