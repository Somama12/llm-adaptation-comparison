"""Compare outputs for the same eleven facts before and after a LoRA update."""
from pathlib import Path
import json, sys


def read(folder, version):
    data=json.loads((folder/f'lora_{version}.json').read_text())
    if data['metadata']['status'] != 'complete':
        raise ValueError('incomplete evaluation')
    rows=[r for r in data['rows'] if r['truth_version']=='v2' and
          r['knowledge_version']==version and r['method']=='lora']
    by_id={r['id']:r for r in rows}
    if len(rows)!=11 or len(by_id)!=11:
        raise ValueError('expected eleven unique question IDs')
    return by_id


def compare(folder):
    old,new=read(folder,'v1'),read(folder,'v2')
    if old.keys()!=new.keys():
        raise ValueError('different question IDs')
    pairs=[]
    for key,a in old.items():
        b=new[key]
        if any(a[k]!=b[k] for k in ('category','question','expected')):
            raise ValueError('different question or target')
        pairs.append(dict(id=key, category=a['category'], question=a['question'],
                          expected=b['expected'], before=a['answer'], after=b['answer'],
                          stale=a['correct'], updated=b['correct']))
    return pairs


if __name__=='__main__':
    folder=Path(sys.argv[1])
    pairs=compare(folder)
    (folder/'pairs.json').write_text(json.dumps(pairs,indent=2)+'\n')
    for group in ('changed','stable'):
        subset=[r for r in pairs if r['category']==group]
        print(group, {k:sum((r['stale'],r['updated'])==v for r in subset) for k,v in
                      [('gained',(False,True)),('lost',(True,False)),
                       ('kept_correct',(True,True)),('still_wrong',(False,False))]})
    for r in pairs:
        print(r['id'], int(r['stale']), '->', int(r['updated']), repr(r['after']))
