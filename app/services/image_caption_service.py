from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageCaptionService:
    def __init__(self) -> None:
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

    def analyze_image(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=50)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption