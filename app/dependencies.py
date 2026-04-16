from app.services.image_caption_service import ImageCaptionService
from app.services.caption_parser_service import CaptionParserService
from app.services.prompt_builder_service import PromptBuilderService
from app.services.translation_service import TranslationService

image_caption_service = ImageCaptionService()
caption_parser_service = CaptionParserService()
prompt_builder_service = PromptBuilderService()
translation_service = TranslationService()
