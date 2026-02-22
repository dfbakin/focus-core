"""Engagement report generation (Phase 7).

Generates post-session reports from inference pipeline output:
    - Session summary (duration, overall engagement distribution)
    - Engagement timeline visualization
    - Key moments (low/high engagement periods with timestamps)
    - Optional: AI-summarized engagement dip analysis
"""


class EngagementReport:
    """Generate engagement timeline reports from prediction results."""

    def __init__(self, predictions: list[dict]):
        raise NotImplementedError

    def generate(self, output_path: str) -> None:
        """Generate and save report to output_path."""
        raise NotImplementedError
