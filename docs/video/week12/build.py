"""Render a narrated code demonstration from measured local and GPU results."""
import json, subprocess, sys, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
R=Path(__file__).resolve().parents[3]
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
def cmd(*args, **kw):return subprocess.run(args,check=True,**kw)
def read(path):return json.loads((R/path).read_text())
folder=R/'results/week12'
pairs=[dict(r,stale=r['audited_stale'],updated=r['audited_updated']) for r in json.loads((folder/'pairs_audited.json').read_text())]
def count(group,key):return sum(r[key] for r in pairs if r['category']==group)
cs,cu,ss,su=[count(g,k) for g,k in [('changed','stale'),('changed','updated'),('stable','stale'),('stable','updated')]]
training=read('results/week12/training_v2.json')
ranklog=cmd(str(R/'.venv/bin/python'),'-m','experiments.rank_audit',cwd=R,capture_output=True,text=True).stdout
code=(R/'experiments/rank_audit.py').read_text().splitlines()
start=next(i for i,l in enumerate(code) if 'scores = ' in l)
ex=next((r for r in pairs if r['updated']!=r['stale']),pairs[0])
scenes=[
('01  /  RESEARCH QUESTION','Can an assistant keep up with a policy change?',
['Nimbus Cloud Storage  /  two handbook versions','6 policies change     •     5 policies stay the same','Experiments: adapter update and passage retrieval'],
"Imagine asking a support assistant how long you have to get a refund. The policy changed, but the question didn't. Does the assistant learn the new answer? And when you ask about something that hasn't changed, does it still remember? This is my Weeks ten through twelve project update. I'll show two experiments: what happened after updating a small language model, and what happened when I reworded the questions used to find handbook passages."),
('02  /  GPU TRAINING','Continue training the same adapter',
['Qwen/Qwen2.5-3B-Instruct','v1: train → evaluate → keep checkpoint','v2: resume v1 → train → evaluate',f"Update: {training['training_examples']} examples  /  {training['epochs']} epochs",f"GPU: {training['gpu']}   •   Training: {training['train_seconds']:.1f} seconds"],
"Here's how the update worked. The old training scores were saved, but there wasn't a verified local copy of the old adapter. So the Colab run rebuilt version one, evaluated it, then continued from that checkpoint using version two examples. That gives a matched before and after comparison. Both sets of answers come from the same training sequence. The older submission stays separate. The figures on screen come from the completed update log."),
('03  /  MEASURED OUTCOME','Compare answers against the same new policy',
['                              Stale     Updated',f'Changed facts                  {cs}/6        {cu}/6',f'Stable facts                   {ss}/5        {su}/5','Metric: core fact marked correct after audit','Source: results/week12/pairs_audited.json'],
f"Here's the result. On the six changed policies, the stale adapter scored {cs} and the updated adapter scored {cu}. On the five unchanged policies, the counts were {ss} before the update and {su} afterward. These are counts of correct core facts. The automated scorer counted five correct after the update. One answer called two factor authentication voluntary even though it was already required. That brought the audited score back to four. That's why the repository keeps the full text for each question, and why I wouldn't rank methods on a single percentage alone."),
('04  /  RAW OUTPUT','Inspect one matched question',
[ex['question'],'Policy target: '+ex['expected'],'Before: '+textwrap.shorten(ex['before'],width=155,placeholder='…'),'After: '+textwrap.shorten(ex['after'],width=155,placeholder='…')],
"This is an actual pair of responses from the run. The question and the target are identical on both sides. Only the adapter changed. The analysis joins by question identifier, so changing the order of the saved answers can't accidentally pair two different facts. It also refuses incomplete runs or duplicate identifiers. That matters on a small benchmark: a single mispaired answer could change the apparent result quite a bit."),
('05  /  CODE WALKTHROUGH','How the retrieval audit works',
[l.strip() for l in code[start:start+5]]+['experiments/rank_audit.py'],
"Now let's look at the retrieval code. It scores each passage against the question, sorts the scores, and looks up the rank of the section that actually contains the policy. I ran this on the same questions in their original and paraphrased forms. I compared whole words with short sequences of characters, then repeated the comparison with section titles included. No language model generates an answer here. This isolates the part of the system that finds evidence."),
('06  /  EXECUTION OUTPUT','What changed when the questions were reworded?',
['$ python -m experiments.rank_audit','Version | wording | index | top 1 | top 3']+[line for line in ranklog.splitlines() if line.startswith('v2 | paraphrase')],
"Here is the output from running that code. On the second handbook, whole words put the right section in the top three for seven of eleven paraphrased questions. Character matching brought that up to ten. Adding the title didn't change those counts. All four indexes looked perfect on the original wording, which is why the paraphrases mattered. These are still the same eleven facts, and finding a passage doesn't guarantee that a model will read it correctly."),
('07  /  CONCLUSION','Two different reasons an answer can go wrong',
['Update failure: the adapter can retain stale facts.','Retrieval failure: wording can hide the right passage.','Evidence: source, raw responses, timings, and ranked passages.','Scope: 11 synthetic facts; one matched update run.'],
"These experiments answer two different questions. Updating an adapter tests whether its answers change. Rewording a query tests whether the system can even find the relevant evidence. Keeping those measurements separate made the failures easier to explain. The core comparison is now measured across the original, stale, and updated conditions. A useful follow up would be a larger handbook with new policies and independent training runs, to see whether the same patterns hold.")]
def font(n,b=False):return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if b else '')+'.ttf',n)
segments=[]; captions=[]; clock=0.0
def stamp(t):
    ms=round(1000*t);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
for i,(label,title,lines,speech) in enumerate(scenes):
    im=Image.new('RGB',(1280,720),'#101d32');d=ImageDraw.Draw(im)
    d.text((50,35),'SOMAMA SIDDIQUI     /     SENIOR SEMINAR I     /     WEEKS 10–12',font=font(20),fill='#afc5d6')
    d.text((50,105),label,font=font(24,True),fill='#66ddc9')
    d.text((50,155),title,font=font(38,True),fill='white')
    y=245
    for line in lines:
        for l in textwrap.wrap(line,width=85,replace_whitespace=False) or ['']:
            d.text((62,y),l,font=font(24),fill='#dae8f3');y+=36
        y+=8
    assert y<665,(i,y)
    d.text((50,670),'github.com/Somama12/llm-adaptation-comparison    •    Synthetic narration',font=font(19),fill='#a6b9cb')
    if i==0:
        thumb=Image.open(R/'docs/video/assets/github.jpg').convert('RGB')
        thumb=thumb.crop((250,425,1160,790));thumb.thumbnail((500,220));im.paste(thumb,(700,450))
    if i in (2,4,5):
        d.rounded_rectangle((45,215,1230,626),12,fill='#0b1523')
        y=250
        mono=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',22)
        for line in lines:
            for fragment in textwrap.wrap(line, width=84,replace_whitespace=False) or ['']:
                d.text((68,y),fragment,font=mono,fill='#a9eadb');y+=33
            y+=7
    pic=out/f'{i}.png';im.save(pic)
    audio=out/f'{i}.aiff';cmd('say','-v','Daniel','-r','170','-o',str(audio),speech)
    seg=out/f'{i}.mp4';cmd('ffmpeg','-y','-v','error','-loop','1','-framerate','10','-i',str(pic),'-i',str(audio),'-shortest','-c:v','libx264','-preset','veryfast','-pix_fmt','yuv420p','-c:a','aac',str(seg))
    duration=float(cmd('ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(seg),capture_output=True,text=True).stdout)
    sentences=speech.split('. ');total=sum(len(x) for x in sentences)
    for sentence in sentences:
        end=clock+duration*len(sentence)/total
        captions.append(f'{len(captions)+1}\n{stamp(clock)} --> {stamp(end)}\n{sentence.rstrip(chr(46))}.\n')
        clock=end
    segments.append("file '"+seg.name+"'")
(out/'parts.txt').write_text('\n'.join(segments))
(out/'captions.srt').write_text('\n'.join(captions))
cmd('ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(out/'parts.txt'),'-i',str(out/'captions.srt'),'-map','0:v','-map','0:a','-map','1:s','-c','copy','-c:s','mov_text','-metadata:s:s:0','language=eng','-movflags','+faststart',str(out/'week10_12_demo.mp4'))
(out/'transcript.txt').write_text('\n\n'.join(s[1]+'\n'+s[3] for s in scenes))
