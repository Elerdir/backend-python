from __future__ import annotations

from typing import Any

import torch
from langdetect import detect, DetectorFactory
from transformers import MarianMTModel, MarianTokenizer


DetectorFactory.seed = 0


class TranslationService:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Czech -> English
        self.cs_model_id = "Helsinki-NLP/opus-mt-cs-en"
        self.cs_tokenizer = MarianTokenizer.from_pretrained(self.cs_model_id)
        self.cs_model = MarianMTModel.from_pretrained(self.cs_model_id).to(self.device)

        # Czech + Slovak -> English
        self.ces_slk_model_id = "Helsinki-NLP/opus-mt-tc-big-ces_slk-en"
        self.ces_slk_tokenizer = MarianTokenizer.from_pretrained(self.ces_slk_model_id)
        self.ces_slk_model = MarianMTModel.from_pretrained(self.ces_slk_model_id).to(self.device)

    def detect_language(self, text: str) -> str:
        cleaned = (text or "").strip()

        if not cleaned:
            return "unknown"

        try:
            return detect(cleaned)
        except Exception:
            return "unknown"

    def is_english(self, text: str) -> bool:
        return self.detect_language(text) == "en"

    def _translate_with_model(
        self,
        text: str,
        tokenizer: MarianTokenizer,
        model: MarianMTModel,
    ) -> str:
        inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True).to(self.device)

        generated = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
        )

        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def translate_to_english(self, text: str) -> dict[str, Any]:
        cleaned = (text or "").strip()

        if not cleaned:
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": "unknown",
                "translated": False,
                "warning": None,
            }

        source_lang = self.detect_language(cleaned)

        if source_lang == "en":
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": "en",
                "translated": False,
                "warning": None,
            }

        try:
            # Direct Czech model
            if source_lang == "cs":
                translated = self._translate_with_model(
                    cleaned,
                    self.cs_tokenizer,
                    self.cs_model,
                )
                return {
                    "original_text": text,
                    "translated_text": translated,
                    "source_language": source_lang,
                    "translated": True,
                    "warning": None,
                }

            # Short Czech prompts are sometimes misdetected as sk/sl.
            if source_lang in ("sk", "sl"):
                translated = self._translate_with_model(
                    cleaned,
                    self.ces_slk_tokenizer,
                    self.ces_slk_model,
                )
                return {
                    "original_text": text,
                    "translated_text": translated,
                    "source_language": source_lang,
                    "translated": True,
                    "warning": "Language was detected as Slovak/Slovenian; translated using Czech/Slovak→English model.",
                }

            return {
                "original_text": text,
                "translated_text": text,
                "source_language": source_lang,
                "translated": False,
                "warning": f"Unsupported source language for translation: {source_lang}",
            }

        except Exception as ex:
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": source_lang,
                "translated": False,
                "warning": f"Translation failed: {str(ex)}",
            }