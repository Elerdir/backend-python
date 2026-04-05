from app.models.responses import AnalysisResponse


class AnalysisService:
    def analyze_mock(self, filename: str, original_filename: str) -> AnalysisResponse:
        return AnalysisResponse(
            filename=filename,
            original_filename=original_filename,
            subject="woman",
            hair="long blonde hair",
            clothing="light summer dress",
            environment="beach at sunset",
            style="photorealistic",
        )