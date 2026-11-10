# Week 8–9 project presentation

The **3-minute, 25-second** video focuses on Weeks 8 and 9. It uses a conversational script, the male **macOS Daniel voice at 170 words per minute**, visible captions, and an embedded English subtitle track.

- [Video (MP4)](week8_9_project_presentation.mp4)
- [Narration script](narration.md)
- [Captions](captions.srt)
- [Build script](build_video.py)

## Walkthrough

1. Actual GitHub repository screenshot and Week 8–9 milestones.
2. Actual baseline execution cell from `notebooks/colab_lora.ipynb`.
3. Actual generation and stale-scoring code in `experiments/run.py`.
4. Actual quantization and adapter configuration in `finetune/train.py`.
5. Saved `results/training_v1.json` measurements.
6. Audited results computed from the saved baseline and LoRA files.
7. A real incorrect few-shot answer and the importance of auditing.
8. Completed work, limitations, and the remaining update experiment.

The code is read directly from repository files, with source line numbers for Python excerpts. Long lines are visually wrapped. The notebook line numbers are within its displayed cell. The GitHub image is an actual browser screenshot; other views are rendered code and evidence panels. GPU experiments are saved evidence, not a live rerun. The video explicitly distinguishes existing training implementation from the Week 9 completed execution. Narration is synthetic, not the student's recorded voice.

The previous cumulative video is superseded by this Week 8–9 revision and remains available in Git history.

## Rebuild

On macOS, provide Pillow, `ffmpeg`, `ffprobe`, and `say`. The repository's `.venv/bin/python` must contain the local retrieval dependencies. The GitHub screenshot is included in `assets/github.jpg`.

```sh
python docs/video/build_video.py /absolute/path/to/output-directory
```

Narration cache entries are keyed by text, voice, and speed. Intermediate images and audio stay in the chosen build directory. The output is H.264/AAC MP4.
