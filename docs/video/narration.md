# Week 1–9 project video narration

Synthetic narration: macOS Samantha. GPU results are saved evidence; retrieval is executed afresh.

## 1. LLM Adaptation Comparison

This presentation introduces Somama Siddiqui’s Senior Seminar project: comparing ways to keep a language model accurate when factual information changes. The project compares retrieval augmented generation, few shot prompting, and low rank adaptation, or LoRA. The goal is to measure initial accuracy, errors caused by outdated knowledge, and recovery after an update. This walkthrough shows the implemented system and completed work through Week nine.

## 2. A controlled, changing handbook

The test domain is a synthetic cloud storage company called Nimbus. Its handbook has two versions. Eleven evaluated facts include five stable facts and six changed facts. For example, the refund window changes from thirty to forty five days. The implementation also creates thirteen adaptation examples. Evaluation questions use different wording, but deliberately test the same facts. This measures factual learning and updating, rather than generalization to new facts.

## 3. Three ways to supply knowledge

The repository contains two retrieval implementations, a few shot prompt builder, a shared evaluation pipeline, and CUDA LoRA training. Retrieval supplies a relevant handbook passage at inference time. Few shot prompting supplies question and answer examples. LoRA updates a small set of model parameters. The Week eight and nine comparison uses the same Qwen two point five, three billion parameter model on a Tesla T four, keeping the earlier local Llama pilot separate.

## 4. Run the retriever on both versions

Here is an actual retrieval run using the repository’s T F I D F retriever. The question asks how many days after purchase a customer can request a refund. With version one, the retrieved section contains thirty days. Rebuilding retrieval with version two returns the updated forty five day policy. These passages were produced by running the code for this video. This demonstrates retrieval; the language model results shown next come from the saved GPU experiments.

## 5. What the completed experiments show

The audited results show both retrieval methods answering ten of eleven questions correctly with version one, and all eleven after refreshing to version two. Few shot prompting scores ten of eleven on each matching version. The version one LoRA adapter scores five of eleven. Its version two update is not yet complete. The stale column re scores the original answers against changed truth, without generating new answers. These are measurements from one small, controlled pilot.

## 6. An update can fail despite new examples

One concrete failure appears in the saved version two few shot output. The expected refund window is forty five days, but the answer still says thirty days. Auditing individual answers also revealed that a stale answer can accidentally match a new policy. That is not evidence of learning an update. The project therefore retains raw outputs and separately audited scores, rather than relying only on an overall percentage.

## 7. Measured execution, preserved evidence

The saved training log confirms a completed version one LoRA run: thirteen examples, twenty epochs, and about one hundred twenty four seconds of training on a Tesla T four. Peak allocated CUDA memory was about three point three six gibibytes. This is framework allocation, not total device memory. The repository preserves the training measurements, answers, scoring records, and report. Durable adapter backup still needs verification or recovery before continued training.

## 8. Finish the update experiment by Week 11

The main finding is that refreshing retrieval recovered all changed facts in this setup, while the small LoRA training recipe learned fewer facts initially. These broad patterns have precedents in research; this project contributes its own controlled measurements and error analysis, not a proven new general discovery. Only eleven synthetic facts were evaluated. Week ten focuses on the version two LoRA update and retention of stable facts. Week eleven completes the comparison, limitations, and final report.