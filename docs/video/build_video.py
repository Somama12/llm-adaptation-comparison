"""Build an evidence-based narrated video (macOS say, ffmpeg, Pillow).
Run with a Python containing Pillow. The repo .venv supplies sklearn for the
fresh retrieval demonstration. GPU answers are replayed from saved results.
"""
from pathlib import Path
import json, subprocess, textwrap, re, sys, hashlib
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
('Week 8–9 update', 'From experiment code to measured results',
 "Hi, here's my Week eight and nine update. My project compares three ways to keep a language model current: retrieval, few shot prompting, and LoRA fine tuning. Over these two weeks, I ran the GPU baselines, completed the first LoRA training run, and audited the answers. Let's look at the actual repository, the code behind those runs, and what the results tell us.", 'github'),
('Week 8 • Colab execution', 'Run every baseline on the same model',
 "This is the baseline cell from my Colab notebook. It calls the experiment runner with the Hugging Face backend and the same Qwen model for retrieval, few shot, and zero shot. That matters because changing the model between methods would make the comparison harder to interpret. The cell also checks that the run is complete before downloading the results. These are saved GPU experiments, not a new training session running during this video.", 'notebook'),
('Week 8 • Evaluation code', 'Generate once, then check both versions',
 "Here's the evaluation loop. First, it retrieves a passage when the method needs one. Then it builds the prompt and generates an answer. The important part is below: the same answer gets checked against both handbook versions. So I can measure what happens when yesterday's knowledge meets today's policy, without generating a different answer and changing the comparison. Each answer, expected value, and score is saved for inspection.", 'runner'),
('Week 9 • Training code', 'The actual QLoRA configuration',
 "For Week nine, I ran this training code. It loads the model in four bit precision and trains small LoRA adapters on the query and value projections. The rank is sixteen, with twenty epochs and thirteen examples. The training setup was already implemented; this week's milestone was completing the run and evaluating its output. This is a deliberately small training budget, so a weak result here doesn't mean fine tuning can never work.", 'traincode'),
('Week 9 • Saved training log', 'The run finished, and the measurements are saved',
 "Here's the training log produced by that run. It used a Tesla T four and took about one hundred twenty four seconds. Peak allocated GPU memory was about three point three six gibibytes. That's memory allocated by the framework, not the entire GPU footprint. The log and evaluation answers are in the repository. The adapter itself still needs a durable backup check before I continue training it.", 'training'),
('Week 8–9 • Audited results', 'What actually worked?',
 "Now for the results. Both retrieval methods got ten out of eleven answers right initially, and eleven out of eleven after the handbook was refreshed. Few shot got ten out of eleven on both matching versions. The first LoRA adapter got five out of eleven. The middle column shows the drop when old answers are judged against the new policy. The LoRA update is still pending, so that last cell isn't a completed result.", 'results'),
('Week 8–9 • Inspect the evidence', 'A real answer that stayed out of date',
 "Here's one example from the saved answer file. With the updated examples, few shot should answer forty five days for the refund window. It still says thirty. Looking at individual answers helped catch scoring issues too, including an old answer that only matched a new policy by coincidence. That's why I kept the original outputs and a separate audited result file. An accuracy number alone doesn't explain what went wrong.", 'answer'),
('Week 8–9 • Takeaway', 'Completed runs, clear findings, one remaining update',
 "So the progress for these two weeks is concrete: comparable GPU baselines, a completed LoRA run, and an audited comparison backed by code and saved answers. Retrieval updated cleanly in this small test; the first LoRA recipe learned fewer facts. With only eleven synthetic facts, I can't claim a universal winner. Next, I'll update LoRA on version two and check both recovery and retention, then finish the project in Week eleven.", 'closing')]
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

def code_panel(draw, path, lines, startline=1):
 draw.rounded_rectangle((55,215,1545,710),16,fill='#0b1422')
 draw.text((80,230),path,font=font(25,True),fill=TEAL)
 mono=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',21)
 y=280
 for number,line in enumerate(lines,startline):
  parts=textwrap.wrap(line, width=105, replace_whitespace=False, drop_whitespace=False) or ['']
  for j,part in enumerate(parts):
   assert y+25<=708, (path,'code panel overflow')
   draw.text((78,y),str(number) if j==0 else '↳',font=mono,fill='#75869c')
   color='#8ce0cb' if part.lstrip().startswith('#') else '#e1e7ef'
   draw.text((135,y),part,font=mono,fill=color)
   y+=25

def frame(index,kind):
 im=Image.new('RGB',(1600,900),NAVY);d=ImageDraw.Draw(im)
 d.rectangle((0,0,1600,9),fill=TEAL)
 d.text((70,40),'SOMAMA SIDDIQUI  /  SENIOR SEMINAR I  /  WEEKS 8–9',font=font(22),fill=MUTED)
 d.text((70,93),scenes[index][1],font=font(47,True),fill=WHITE)
 d.text((70,160),scenes[index][0].upper(),font=font(23,True),fill=TEAL)
 if kind=='github':
  shot=Image.open(REPO/'docs/video/assets/github.jpg').convert('RGB')
  shot.thumbnail((790,500))
  im.paste(shot,(70,210))
  wrap(d,'WEEK 8',800,240,size=32,color=TEAL,b=True)
  wrap(d,'Same-model GPU baselines\nSaved and audited answers',800,295,width=700,size=32)
  wrap(d,'WEEK 9',800,420,size=32,color=TEAL,b=True)
  wrap(d,'Version 1 LoRA training\nInitial and stale evaluation',800,475,width=700,size=32)
  wrap(d,'Code + notebook + raw evidence on GitHub',800,625,width=700,size=25,color=MUTED)
 elif kind=='notebook':
  notebook=read('notebooks/colab_lora.ipynb')
  cell=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and "# Preserve the completed Week 8" in ''.join(c['source']))
  code_panel(d,'notebooks/colab_lora.ipynb • Week 8 baseline cell',cell.splitlines())
 elif kind=='runner':
  lines=(REPO/'experiments/run.py').read_text().splitlines()
  start=next(i for i,l in enumerate(lines) if 'retrieval_seconds = time.perf_counter()-start' in l)
  code_panel(d,'experiments/run.py • generation and stale scoring',textwrap.dedent('\n'.join(lines[start:start+14])).splitlines(),start+1)
 elif kind=='traincode':
  lines=(REPO/'finetune/train.py').read_text().splitlines()
  start=next(i for i,l in enumerate(lines) if 'model = AutoModelForCausalLM.from_pretrained' in l)
  code_panel(d,'finetune/train.py • quantization and LoRA setup',textwrap.dedent('\n'.join(lines[start:start+10])).splitlines(),start+1)
 elif kind=='intro':
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
 cache=audio.with_suffix('.sha256')
 digest=hashlib.sha256(('Daniel|170|'+narration).encode()).hexdigest()
 if not audio.exists() or audio.stat().st_size <= 4096 or not cache.exists() or cache.read_text()!=digest:
  run(['say','-v','Daniel','-r','170','-o',str(audio),narration])
  cache.write_text(digest)
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
(OUT/'narration.md').write_text('# Week 8–9 project video narration\n\nSynthetic narration: macOS Daniel, 170 words per minute. GPU results are saved evidence; displayed code is read directly from the repository.\n\n'+'\n\n'.join(f'## {i+1}. {s[1]}\n\n{s[2]}' for i,s in enumerate(scenes)))
run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(OUT/'audio.txt'),'-c:a','aac','-b:a','128k',str(OUT/'narration.m4a')])
run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(OUT/'frames.txt'),'-i',str(OUT/'narration.m4a'),'-i',str(OUT/'captions.srt'),'-map','0:v','-map','1:a','-map','2:s','-vf','fps=15','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-c:a','copy','-c:s','mov_text','-metadata:s:s:0','language=eng','-movflags','+faststart','-t',str(elapsed),str(OUT/'week8_9_project_presentation.mp4')])
assert 120<=probe(OUT/'week8_9_project_presentation.mp4')<=300
print(f'Finished: {elapsed:.1f} seconds',flush=True)
