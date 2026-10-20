# Week 8–9 project video narration

Synthetic narration: macOS Daniel, 170 words per minute. GPU results are saved evidence; displayed code is read directly from the repository.

## 1. From experiment code to measured results

Hi, here's my Week eight and nine update. My project compares three ways to keep a language model current: retrieval, few shot prompting, and LoRA fine tuning. Over these two weeks, I ran the GPU baselines, completed the first LoRA training run, and audited the answers. Let's look at the actual repository, the code behind those runs, and what the results tell us.

## 2. Run every baseline on the same model

This is the baseline cell from my Colab notebook. It calls the experiment runner with the Hugging Face backend and the same Qwen model for retrieval, few shot, and zero shot. That matters because changing the model between methods would make the comparison harder to interpret. The cell also checks that the run is complete before downloading the results. These are saved GPU experiments, not a new training session running during this video.

## 3. Generate once, then check both versions

Here's the evaluation loop. First, it retrieves a passage when the method needs one. Then it builds the prompt and generates an answer. The important part is below: the same answer gets checked against both handbook versions. So I can measure what happens when yesterday's knowledge meets today's policy, without generating a different answer and changing the comparison. Each answer, expected value, and score is saved for inspection.

## 4. The actual QLoRA configuration

For Week nine, I ran this training code. It loads the model in four bit precision and trains small LoRA adapters on the query and value projections. The rank is sixteen, with twenty epochs and thirteen examples. The training setup was already implemented; this week's milestone was completing the run and evaluating its output. This is a deliberately small training budget, so a weak result here doesn't mean fine tuning can never work.

## 5. The run finished, and the measurements are saved

Here's the training log produced by that run. It used a Tesla T four and took about one hundred twenty four seconds. Peak allocated GPU memory was about three point three six gibibytes. That's memory allocated by the framework, not the entire GPU footprint. The log and evaluation answers are in the repository. The adapter itself still needs a durable backup check before I continue training it.

## 6. What actually worked?

Now for the results. Both retrieval methods got ten out of eleven answers right initially, and eleven out of eleven after the handbook was refreshed. Few shot got ten out of eleven on both matching versions. The first LoRA adapter got five out of eleven. The middle column shows the drop when old answers are judged against the new policy. The LoRA update is still pending, so that last cell isn't a completed result.

## 7. A real answer that stayed out of date

Here's one example from the saved answer file. With the updated examples, few shot should answer forty five days for the refund window. It still says thirty. Looking at individual answers helped catch scoring issues too, including an old answer that only matched a new policy by coincidence. That's why I kept the original outputs and a separate audited result file. An accuracy number alone doesn't explain what went wrong.

## 8. Completed runs, clear findings, one remaining update

So the progress for these two weeks is concrete: comparable GPU baselines, a completed LoRA run, and an audited comparison backed by code and saved answers. Retrieval updated cleanly in this small test; the first LoRA recipe learned fewer facts. With only eleven synthetic facts, I can't claim a universal winner. Next, I'll update LoRA on version two and check both recovery and retention, then finish the project in Week eleven.