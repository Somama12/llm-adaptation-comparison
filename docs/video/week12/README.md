# Weeks 10–12 closing experiment demo

Demo build source (not yet rendered). [Research report](../../reports/week10_12.pdf)

The presentation shows the matched Colab LoRA experiment, a saved question and answer pair, the new Python ranking harness, and its execution output. The GPU measurements are read from the saved run; the retrieval audit runs locally during the video build. The seven scenes use the Daniel synthetic voice (170 words per minute) and an English subtitle track.

On macOS, run the build with a Python environment containing Pillow, with ffmpeg, ffprobe and say on PATH. The project environment must exist at .venv with the retrieval dependencies.

    python docs/video/week12/build.py /absolute/path/to/render-directory
