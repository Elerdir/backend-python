from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class CaptionParserService:
    def __init__(self) -> None:
        model_name = "google/flan-t5-base"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _ask(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=30)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return text if text else "unknown"

    def _normalize(self, value: str) -> str:
        if not value:
            return "unknown"

        value = value.strip().lower()

        prefixes = ("a ", "an ", "the ")
        for prefix in prefixes:
            if value.startswith(prefix):
                value = value[len(prefix):]

        bad_values = {
            "",
            "none",
            "not visible",
            "not specified",
            "not mentioned",
            "unknown hair",
            "unknown clothing",
            "unknown environment",
            "unknown style",
        }

        if value in bad_values:
            return "unknown"

        return value

    def parse_caption(self, caption: str) -> dict:
        subject = self._ask(
            f'From this description, what is the main subject? '
            f'Return only a short answer.\nDescription: "{caption}"'
        )

        hair = self._ask(
            f'From this description, what hair is described? '
            f'If none is described, return only "unknown".\nDescription: "{caption}"'
        )

        clothing = self._ask(
            f'From this description, what clothing is described? '
            f'If none is described, return only "unknown".\nDescription: "{caption}"'
        )

        environment = self._ask(
            f'From this description, what environment or setting is described? '
            f'Return only a short answer.\nDescription: "{caption}"'
        )

        style = self._ask(
            f'From this description, what visual style is described? '
            f'If none is described, return only "unknown".\nDescription: "{caption}"'
        )

        result = {
            "subject": self._normalize(subject),
            "hair": self._normalize(hair),
            "clothing": self._normalize(clothing),
            "environment": self._normalize(environment),
            "style": self._normalize(style),
        }

        print(f"Parsed fields: {result}")
        return result