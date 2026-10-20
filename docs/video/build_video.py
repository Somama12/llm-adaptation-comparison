"""Build an evidence-based narrated video (macOS say, ffmpeg, Pillow).
Run with a Python containing Pillow. The repo .venv supplies sklearn for the
fresh retrieval demonstration. GPU answers are replayed from saved results.
"""
from pathlib import Path
import json, subprocess, textwrap, re, sys
from PIL import Image, ImageDraw, ImageFont
REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else REPO/'docs/video/rendered'
OUT.mkdir(parents=True, exist_ok=True)
def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)
def read(name): return json.loads((REPO/name).read_text())
base=read('results/colab_baselines_audited.json'); lora=read('results/lora_v1.json'); train=read('results/training_v1.json')
assert base['metadata']['status']==lora['metadata']['status']=='complete'
# Run actual repository retrieval code; retain the exact returned passages.
code='''import json
from rag.retriever import chunk_markdown, TfidfRetriever
q="How many days after purchase can a customer request a refund?"
out=[]
for v in ("v1","v2"):
 c,s=TfidfRetriever(chunk_markdown(f"data/handbook_{v}.md")).retrieve(q)[0]
 out.append(dict(version=v, question=q, section=c.section_id, passage=c.text, similarity=s))
print(json.dumps(out))'''
retrieval=json.loads(run([str(REPO/'.venv/bin/python'),'-c',code],cwd=REPO,capture_output=True,text=True).stdout)
(OUT/'retrieval_demo.json').write_text(json.dumps(retrieval,indent=2))
def score(data,method,k,t):
 rows=[r for r in data['summary'] if (r['method'],r['knowledge_version'],r['truth_version'])==(method,k,t)]
 return f"{sum(r['correct'] for r in rows)}/{sum(r['n'] for r in rows)}"
refund=[r for r in base['rows'] if r['method']=='fewshot' and r['knowledge_version']=='v2' and r['truth_version']=='v2' and 'refund' in r['question'].lower() and 'window' in str(r.get('id','')).lower()]
if not refund:
 refund=[r for r in base['rows'] if r['method']=='fewshot' and r['knowledge_version']=='v2' and r['truth_version']=='v2' and not r['correct']]
assert len(refund)==1
scenes=[
('The project question', 'LLM Adaptation Comparison',
 'This presentation introduces Somama Siddiqui’s Senior Seminar project: comparing ways to keep a language model accurate when factual information changes. The project compares retrieval augmented generation, few shot prompting, and low rank adaptation, or LoRA. The goal is to measure initial accuracy, errors caused by outdated knowledge, and recovery after an update. This walkthrough shows the implemented system and completed work through Week nine.', 'intro'),
('Experimental design', 'A controlled, changing handbook',
 'The test domain is a synthetic cloud storage company called Nimbus. Its handbook has two versions. Eleven evaluated facts include five stable facts and six changed facts. For example, the refund window changes from thirty to forty five days. The implementation also creates thirteen adaptation examples. Evaluation questions use different wording, but deliberately test the same facts. This measures factual learning and updating, rather than generalization to new facts.', 'design'),
('Implementation', 'Three ways to supply knowledge',
 'The repository contains two retrieval implementations, a few shot prompt builder, a shared evaluation pipeline, and CUDA LoRA training. Retrieval supplies a relevant handbook passage at inference time. Few shot prompting supplies question and answer examples. LoRA updates a small set of model parameters. The Week eight and nine comparison uses the same Qwen two point five, three billion parameter model on a Tesla T four, keeping the earlier local Llama pilot separate.', 'architecture'),
('Actual program demonstration', 'Run the retriever on both versions',
 'Here is an actual retrieval run using the repository’s T F I D F retriever. The question asks how many days after purchase a customer can request a refund. With version one, the retrieved section contains thirty days. Rebuilding retrieval with version two returns the updated forty five day policy. These passages were produced by running the code for this video. This demonstrates retrieval; the language model results shown next come from the saved GPU experiments.', 'demo'),
('Saved GPU results', 'What the completed experiments show',
 'The audited results show both retrieval methods answering ten of eleven questions correctly with version one, and all eleven after refreshing to version two. Few shot prompting scores ten of eleven on each matching version. The version one LoRA adapter scores five of eleven. Its version two update is not yet complete. The stale column re scores the original answers against changed truth, without generating new answers. These are measurements from one small, controlled pilot.', 'results'),
('Inspect an actual answer', 'An update can fail despite new examples',
 'One concrete failure appears in the saved version two few shot output. The expected refund window is forty five days, but the answer still says thirty days. Auditing individual answers also revealed that a stale answer can accidentally match a new policy. That is not evidence of learning an update. The project therefore retains raw outputs and separately audited scores, rather than relying only on an overall percentage.', 'answer'),
('Completed LoRA training', 'Measured execution, preserved evidence',
 'The saved training log confirms a completed version one LoRA run: thirteen examples, twenty epochs, and about one hundred twenty four seconds of training on a Tesla T four. Peak allocated CUDA memory was about three point three six gibibytes. This is framework allocation, not total device memory. The repository preserves the training measurements, answers, scoring records, and report. Durable adapter backup still needs verification or recovery before continued training.', 'training'),
('Interpretation and next steps', 'Finish the update experiment by Week 11',
 'The main finding is that refreshing retrieval recovered all changed facts in this setup, while the small LoRA training recipe learned fewer facts initially. These broad patterns have precedents in research; this project contributes its own controlled measurements and error analysis, not a proven new general discovery. Only eleven synthetic facts were evaluated. Week ten focuses on the version two LoRA update and retention of stable facts. Week eleven completes the comparison, limitations, and final report.', 'closing')]
fontdir=Path('/System/Library/Fonts/Supplemental')
def font(n,b=False): return ImageFont.truetype(str(fontdir/('Arial Bold.ttf' if b else 'Arial.ttf')),n)
NAVY='#101d32'; MUTED='#a7b9cf'; WHITE='#f3f7fc'; TEAL='#59dac4'; PANEL='#1a2b43'
def wrap(draw,text,x,y,width=1400,size=32,color=WHITE,b=False,gap=13):
 f=font(size,b); lines=[]
 for para in text.split('\n'):
  line=''
  for word in para.split():
   test=(line+' '+word).strip()
   if draw.textlength(test,font=f)>width and line: lines.append(line);line=word
   else: line=test
  lines.append(line)
 for line in lines: draw.text((x,y),line,font=f,fill=color); y+=size+gap
 return y

def frame(index,kind):
 im=Image.new('RGB',(1600,900),NAVY);d=ImageDraw.Draw(im)
 d.rectangle((0,0,1600,9),fill=TEAL)
 d.text((70,40),'SOMAMA SIDDIQUI  /  SENIOR SEMINAR I  /  WEEKS 1–9',font=font(22),fill=MUTED)
 d.text((70,93),scenes[index][1],font=font(47,True),fill=WHITE)
 d.text((70,160),scenes[index][0].upper(),font=font(23,True),fill=TEAL)
 if kind=='intro':
  wrap(d,'How well do language models keep up\nwhen the facts change?',70,270,size=55,b=True)
  wrap(d,'RAG     •     Few-shot prompting     •     LoRA',70,470,size=35,color=TEAL)
  wrap(d,'Actual retrieval execution + saved GPU experiment walkthrough',70,565,size=28,color=MUTED)
 elif kind=='design':
  for x,v in [(70,'v1'),(820,'v2')]:
   d.rounded_rectangle((x,235,x+700,490),20,fill=PANEL)
   wrap(d,'data/handbook_'+v+'.md',x+25,260,width=650,size=27,color=TEAL)
   passage=next(s for s in (REPO/f'data/handbook_{v}.md').read_text().split('\n') if 'Customers may request' in s)
   wrap(d,passage.replace('**',''),x+25,320,width=645,size=29)
  wrap(d,'11 evaluated facts  =  5 stable + 6 changed',70,545,size=36,b=True)
  wrap(d,'13 adaptation examples • Held-out wording, overlapping facts',70,610,size=29,color=MUTED)
 elif kind=='architecture':
  for x,name,body,path in [(70,'RAG','Handbook → retrieve passage','rag/retriever.py\nrag/embedding.py'),(575,'Few-shot','Examples → prompt context','experiments/run.py'),(1080,'LoRA','Examples → train adapter','finetune/train.py')]:
   d.rounded_rectangle((x,245,x+450,480),18,fill=PANEL)
   wrap(d,name,x+22,267,width=405,size=36,color=TEAL,b=True)
   wrap(d,body,x+22,332,width=405,size=27)
   wrap(d,path,x+22,410,width=405,size=23,color=MUTED)
  
  for x in (295,800,1305): d.text((x,500),'↓',font=font(50),fill=TEAL)
  wrap(d,'Same Qwen 2.5 3B → generated answers → scoring → audited JSON',95,580,size=33,b=True)
  wrap(d,'One NF4 inference backend • Tesla T4 • Local Llama pilot reported separately',95,640,size=26,color=MUTED)
 elif kind=='demo':
  wrap(d,'Fresh execution: TfidfRetriever(...).retrieve(question)',70,220,size=29,color=TEAL)
  wrap(d,'Q: '+retrieval[0]['question'],70,275,size=31,b=True)
  y=350
  for r in retrieval:
   d.rounded_rectangle((70,y,1530,y+150),15,fill=PANEL)
   wrap(d,r['version'].upper()+'  →  '+r['section'],90,y+15,size=27,color=TEAL)
   wrap(d,r['passage'].replace('**',''),90,y+60,width=1390,size=27)
   y+=170
 elif kind=='results':
  cols=[80,640,940,1250]
  for x,t in zip(cols,['Method','Original v1','Stale v1 → v2','Updated v2']): d.text((x,240),t,font=font(26,True),fill=TEAL)
  for i,(method,name,ds) in enumerate([('tfidf','TF-IDF RAG',base),('embedding','Embedding RAG',base),('fewshot','Few-shot',base),('lora','LoRA',lora)]):
   y=300+i*76
   vals=[name,score(ds,method,'v1','v1'),score(ds,method,'v1','v2'),'Pending' if method=='lora' else score(ds,method,'v2','v2')]
   for x,t in zip(cols,vals): d.text((x,y),t,font=font(32),fill=WHITE)
  wrap(d,'Correct answers / 11 facts • Audited scores • One retained evaluation per condition',80,635,size=25,color=MUTED)
 elif kind=='answer':
  r=refund[0]
  wrap(d,'results/colab_baselines_audited.json',70,230,size=27,color=TEAL)
  y=wrap(d,'Question: '+r['question'],70,300,size=33,b=True)
  y=wrap(d,'Expected: '+r['expected'],70,y+27,size=33,color=TEAL)
  y=wrap(d,'Saved answer: '+r['answer'],70,y+25,size=33)
  wrap(d,'Method: fewshot  |  Knowledge: v2  |  Truth: v2  |  Correct: false',70,635,size=26,color=MUTED)
 elif kind=='training':
  wrap(d,'results/training_v1.json — actual saved log',70,230,size=29,color=TEAL)
  text=json.dumps({k:train[k] for k in ['gpu','training_examples','epochs','train_seconds','peak_allocated_cuda_bytes']},indent=2)
  wrap(d,text,95,290,size=31,gap=12)
  wrap(d,'≈ 124.2 seconds  |  ≈ 3.36 GiB peak allocated CUDA memory',70,625,size=30,color=TEAL)
 elif kind=='closing':
  for y,title,desc in [(240,'Completed','Retrieval + prompting baselines; v1 LoRA; saved answers and audit.'),(355,'Week 10','Update LoRA to v2; measure changed-fact recovery and stable-fact retention.'),(470,'Week 11','Finalize comparisons, error analysis, reproducibility, and final report.')]:
   wrap(d,title,70,y,size=32,color=TEAL,b=True)
   wrap(d,desc,70,y+48,size=29)
  wrap(d,'Pilot evidence: 11 synthetic facts; no claim of universal method superiority.',70,625,size=27,color=MUTED)
 d.text((70,730),'github.com/Somama12/llm-adaptation-comparison',font=font(22),fill=MUTED)
 d.text((1000,730),'Synthetic narration',font=font(22),fill=MUTED)
 d.text((1390,730),f'{index+1:02d} / 08',font=font(22),fill=MUTED)
 return im

def probe(p): return float(run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(p)],capture_output=True,text=True).stdout.strip())
def stamp(t):
 ms=round(t*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
clips=[];image_lines=[];subtitles=[];elapsed=0;counter=1;manifest=[]
for idx,(label,title,narration,kind) in enumerate(scenes):
 audio=OUT/f'scene_{idx+1:02}.aiff'
 if not audio.exists() or audio.stat().st_size <= 4096:
  run(['say','-v','Samantha','-r','145','-o',str(audio),narration])
 duration=probe(audio);clips.append(audio);baseframe=frame(idx,kind)
 baseframe.save(OUT/f'preview_{idx+1:02}.png')
 sentences=re.split(r'(?<=[.!?])\s+',narration)
 total=sum(len(s.split()) for s in sentences)
 for sn,s in enumerate(sentences):
  dur=duration*len(s.split())/total
  im=baseframe.copy();d=ImageDraw.Draw(im);d.rectangle((0,780,1600,900),fill='#07111e')
  wrap(d,s,70,799,width=1460,size=27,gap=9)
  path=OUT/f'frame_{idx+1:02}_{sn:02}.png';im.save(path)
  image_lines += [f"file '{path.name}'",f'duration {dur:.6f}']
  subtitles.append(f'{counter}\n{stamp(elapsed)} --> {stamp(elapsed+dur)}\n{s}\n')
  elapsed+=dur;counter+=1
 manifest.append({'scene':idx+1,'title':title,'duration_seconds':duration,'narration':narration})
 print(f'Scene {idx+1}: {duration:.1f}s',flush=True)
image_lines.append(f"file '{path.name}'")
(OUT/'frames.txt').write_text('\n'.join(image_lines)+'\n')
(OUT/'audio.txt').write_text('\n'.join(f"file '{p.name}'" for p in clips)+'\n')
(OUT/'captions.srt').write_text('\n'.join(subtitles))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
(OUT/'narration.md').write_text('# Week 1–9 project video narration\n\nSynthetic narration: macOS Samantha. GPU results are saved evidence; retrieval is executed afresh.\n\n'+'\n\n'.join(f'## {i+1}. {s[1]}\n\n{s[2]}' for i,s in enumerate(scenes)))
run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(OUT/'audio.txt'),'-c:a','aac','-b:a','128k',str(OUT/'narration.m4a')])
run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(OUT/'frames.txt'),'-i',str(OUT/'narration.m4a'),'-i',str(OUT/'captions.srt'),'-map','0:v','-map','1:a','-map','2:s','-vf','fps=15','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-c:a','copy','-c:s','mov_text','-metadata:s:s:0','language=eng','-movflags','+faststart','-t',str(elapsed),str(OUT/'week1_9_project_presentation.mp4')])
assert 120<=probe(OUT/'week1_9_project_presentation.mp4')<=300
print(f'Finished: {elapsed:.1f} seconds',flush=True)
