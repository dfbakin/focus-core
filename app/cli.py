"""Command line analysis of a lecture recording; the entry point of the Docker image.

    python -m app.cli lecture.mp4
    python -m app.cli lecture.mp4 --out reports --threshold 45 --interval 2

Writes <name>_report.json with the whole engagement series and the critical
moments, and <name>_report.pdf with the same report as the application.
If the OPENAI_API_KEY environment variable is set and the file has sound,
speech around the deepest drops of every critical moment is transcribed.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from app.core.audio_source import AudioTrackMissingError, read_audio_track
from app.core.config import AnalysisSettings
from app.core.file_analysis import (
    VideoUnreadableError,
    analyze_video_file,
    build_report,
)
from app.core.report import MomentNote, SessionReport, moment_key
from app.core.report_pdf import export_report_pdf
from app.core.transcription import (
    TranscriptionError,
    WhisperClient,
    load_api_key,
    transcribe_moments,
)

EXIT_UNREADABLE = 1
EXIT_NO_DATA = 2


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line; defaults match the analysis settings of the application."""
    defaults = AnalysisSettings()
    parser = argparse.ArgumentParser(
        prog="focuscore", description="Estimate group engagement in a lecture recording."
    )
    parser.add_argument("video", type=Path, help="video file to analyse")
    parser.add_argument("--out", type=Path, default=None,
                        help="folder for the reports (default: next to the video)")
    parser.add_argument("--threshold", type=int, default=defaults.threshold_pct,
                        help="critical engagement threshold, percent")
    parser.add_argument("--interval", type=float, default=defaults.interval_sec,
                        help="seconds of video between two estimates")
    parser.add_argument("--smoothing", type=int, default=defaults.smoothing_window,
                        help="smoothing span, 1 = no smoothing")
    return parser.parse_args(argv)


def report_to_dict(report: SessionReport, video: Path) -> dict:
    """Describe the report as plain data for the JSON file."""
    return {
        "video": video.name,
        "duration_sec": report.duration_sec,
        "threshold_pct": report.threshold_pct,
        "average": report.average,
        "minimum": report.minimum,
        "points": [{"offset_sec": offset, "value": value} for offset, value in report.points],
        "critical_moments": [
            {
                "start_sec": moment.start_sec,
                "end_sec": moment.end_sec,
                "min_value": moment.min_value,
                "transcript": report.note_for(moment).transcript_parts,
            }
            for moment in report.critical_moments
        ],
    }


def add_transcripts(report: SessionReport, video: Path) -> SessionReport:
    """Transcribe the critical moments if a key is set; otherwise return the report as is."""
    api_key = load_api_key()
    moments = report.critical_moments
    if api_key is None or not moments:
        return report
    try:
        audio = read_audio_track(video)
        texts = transcribe_moments(moments, report.points, audio, WhisperClient(api_key))
    except (AudioTrackMissingError, TranscriptionError) as exc:
        print(f"Transcription skipped: {exc}", file=sys.stderr)
        return report
    notes = {
        moment_key(moment.start_sec): MomentNote(transcript=text)
        for moment, text in zip(moments, texts)
        if text
    }
    return replace(report, notes=notes)


def _show_progress(position_sec: float, duration_sec: float) -> None:
    """Print how much of the video is processed, on one line."""
    percent = 100 * position_sec / duration_sec if duration_sec else 0
    print(f"\rAnalysed {percent:5.1f} %", end="", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Run the analysis and write the reports; return the exit code."""
    args = parse_arguments(argv)
    if not args.video.is_file():
        print(f"File not found: {args.video}", file=sys.stderr)
        return EXIT_UNREADABLE
    from app.core.engagement import EngagementModel

    model = EngagementModel()
    print(f"Model runs on {model.device}", file=sys.stderr)
    try:
        analysis = analyze_video_file(
            args.video, model.engagement_score, args.interval, args.smoothing, _show_progress
        )
    except VideoUnreadableError as exc:
        print(exc, file=sys.stderr)
        return EXIT_UNREADABLE
    print(file=sys.stderr)

    report = build_report(analysis, args.threshold)
    if not report.has_data:
        print("The video is too short for an estimate.", file=sys.stderr)
        return EXIT_NO_DATA
    report = add_transcripts(report, args.video)

    out_dir = args.out or args.video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{args.video.stem}_report"
    json_path = stem.with_suffix(".json")
    json_path.write_text(
        json.dumps(report_to_dict(report, args.video), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_report_pdf(report, stem.with_suffix(".pdf"))
    print(f"Saved {json_path} and {stem.with_suffix('.pdf')}", file=sys.stderr)
    print(f"Average engagement {report.average:.0f} %, "
          f"critical moments: {len(report.critical_moments)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
