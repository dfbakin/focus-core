"""Add a sound track to a silent demo video.

The dataset clips have no sound, so a demo of transcription needs a voice
recorded separately, for example a Voice Memos recording of a lecture
fragment. The video stream is copied as is, the sound is converted to AAC
and cut to the length of the video.

Run from the repository root:
    python -m scripts.add_audio_track demo_lecture.mp4 voice.m4a demo_lecture_audio.mp4
"""

import sys
from fractions import Fraction

import av

AUDIO_RATE = 44100


def add_audio_track(video_path: str, audio_path: str, output_path: str) -> None:
    """Write `output_path` with the video of `video_path` and the sound of `audio_path`."""
    with av.open(video_path) as video_in, av.open(audio_path) as audio_in, av.open(
        output_path, "w"
    ) as output:
        video_stream = video_in.streams.video[0]
        video_out = output.add_stream_from_template(video_stream)
        audio_out = output.add_stream("aac", rate=AUDIO_RATE, layout="mono")
        video_duration = float(video_stream.duration * video_stream.time_base)

        for packet in video_in.demux(video_stream):
            if packet.dts is None:
                continue
            packet.stream = video_out
            output.mux(packet)

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=AUDIO_RATE)
        written = 0
        limit = int(video_duration * AUDIO_RATE)
        for frame in audio_in.decode(audio_in.streams.audio[0]):
            for converted in resampler.resample(frame):
                if written >= limit:
                    break
                converted.pts = written
                converted.time_base = Fraction(1, AUDIO_RATE)
                written += converted.samples
                for packet in audio_out.encode(converted):
                    output.mux(packet)
        for packet in audio_out.encode(None):
            output.mux(packet)


def main() -> None:
    """Parse the three paths from the command line."""
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    add_audio_track(*sys.argv[1:])
    print("saved:", sys.argv[3])


if __name__ == "__main__":
    main()
