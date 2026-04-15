from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageCaptionService:
    def __init__(self) -> None:
        self.model_id = "Salesforce/blip-image-captioning-base"
        self.processor = None
        self.model = None

    def _ensure_loaded(self) -> None:
        try:
            if self.processor is None:
                self.processor = BlipProcessor.from_pretrained(self.model_id)

            if self.model is None:
                self.model = BlipForConditionalGeneration.from_pretrained(self.model_id)
        except Exception as ex:
            raise RuntimeError(
                f"Failed to load image caption model '{self.model_id}'. "
                f"Check torch/transformers compatibility. Original error: {ex}"
            ) from ex

    def analyze_image(self, image_path: str) -> str:
        self._ensure_loaded()

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=50)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption