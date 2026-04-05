from app.models.responses import AnalysisResponse


class PromptBuilderService:
    def build_prompt(self, analysis: AnalysisResponse) -> str:
        parts: list[str] = ["photorealistic"]

        if analysis.subject != "unknown":
            parts.append(analysis.subject)

        if analysis.hair != "unknown":
            parts.append(analysis.hair)

        if analysis.clothing != "unknown":
            parts.append(f"wearing {analysis.clothing}")

        if analysis.environment != "unknown":
            parts.append(f"with {analysis.environment} background")

        if analysis.style != "unknown":
            parts.append(analysis.style)

        return ", ".join(parts)

    def build_negative_prompt(self) -> str:
        return (
            "blurry, low quality, distorted anatomy, extra fingers, extra limbs, "
            "deformed face, bad hands, cropped, duplicate"
        )