from __future__ import annotations

from typing import Any

import torch
from langdetect import detect, DetectorFactory
from transformers import MarianMTModel, MarianTokenizer


DetectorFactory.seed = 0


class TranslationService:
    """Translation service with lazy model loading.

    Models are downloaded/loaded on first use, not at startup.  This allows
    the server to start normally even when HuggingFace Hub is unreachable or
    the model weights are not yet cached locally.
    """

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Czech -> English
        self.cs_model_id = "Helsinki-NLP/opus-mt-cs-en"
        self._cs_tokenizer: MarianTokenizer | None = None
        self._cs_model: MarianMTModel | None = None

        # Czech + Slovak -> English
        self.ces_slk_model_id = "Helsinki-NLP/opus-mt-tc-big-ces_slk-en"
        self._ces_slk_tokenizer: MarianTokenizer | None = None
        self._ces_slk_model: MarianMTModel | None = None

    # --------------------------------------------------
    # Lazy-loaded model properties
    # --------------------------------------------------

    @property
    def cs_tokenizer(self) -> MarianTokenizer:
        if self._cs_tokenizer is None:
            print(f"[TRANSLATION] Loading tokenizer: {self.cs_model_id}")
            self._cs_tokenizer = MarianTokenizer.from_pretrained(self.cs_model_id)
        return self._cs_tokenizer

    @property
    def cs_model(self) -> MarianMTModel:
        if self._cs_model is None:
            print(f"[TRANSLATION] Loading model: {self.cs_model_id}")
            self._cs_model = MarianMTModel.from_pretrained(self.cs_model_id).to(self.device)
        return self._cs_model

    @property
    def ces_slk_tokenizer(self) -> MarianTokenizer:
        if self._ces_slk_tokenizer is None:
            print(f"[TRANSLATION] Loading tokenizer: {self.ces_slk_model_id}")
            self._ces_slk_tokenizer = MarianTokenizer.from_pretrained(self.ces_slk_model_id)
        return self._ces_slk_tokenizer

    @property
    def ces_slk_model(self) -> MarianMTModel:
        if self._ces_slk_model is None:
            print(f"[TRANSLATION] Loading model: {self.ces_slk_model_id}")
            self._ces_slk_model = MarianMTModel.from_pretrained(self.ces_slk_model_id).to(self.device)
        return self._ces_slk_model

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