# Project presentation: work completed through Week 9

The approximately **4-minute** video introduces the project and walks through actual repository evidence. It uses **synthetic narration (macOS Samantha)**, visible captions, and an embedded English subtitle track. The transcript is included for an optional personal voice recording.

- [Video (MP4)](week1_9_project_presentation.mp4)
- [Narration script](narration.md)
- [Captions](captions.srt)
- [Fresh retrieval output used in the demonstration](retrieval_demo.json)
- [Build script](build_video.py)

## Demonstrated work

1. Research question and the Nimbus handbook versions.
2. Implemented retrieval, few-shot, and LoRA architecture.
3. A fresh execution of `rag.retriever.TfidfRetriever` against both versions, showing the refund policy changing from 30 to 45 days. This demonstrates retrieval, not a new GPU generation.
4. A table computed directly from the completed, audited baseline and LoRA result files.
5. A real incorrect few-shot answer from the saved results.
6. The completed training log, study limitations, and Week 10–11 plan.

GPU training and model generation are **saved experiments**, not recorded live during video creation. The video does not claim that the Version 2 LoRA update is complete or that the pilot establishes a universal ranking. It is a rendered evidence walkthrough, not a camera recording or desktop screen capture.

## Rebuild

On macOS, install Pillow in the Python used to render the video and provide `ffmpeg`, `ffprobe`, and `say` on PATH. The repository's `.venv/bin/python` must have the local retrieval dependencies installed.

```sh
python docs/video/build_video.py /absolute/path/to/new-output-directory
```

Use a new output directory after changing narration, because existing audio clips are reused. The build reads completed JSON results, executes retrieval, synthesizes narration, renders captions, and exports H.264/AAC MP4. Intermediate images and audio stay in the chosen build directory.
