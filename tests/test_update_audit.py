import json
import pytest
from experiments.update_audit import compare


def make(folder,v,ids):
    rows=[dict(id=str(i),truth_version='v2',knowledge_version=v,method='lora',
               question=str(i),category='stable',expected='yes',answer='yes',correct=True) for i in ids]
    (folder/f'lora_{v}.json').write_text(json.dumps(dict(metadata={'status':'complete'},rows=rows)))


def test_pairs_by_id_even_if_output_order_changes(tmp_path):
    make(tmp_path,'v1',range(11))
    make(tmp_path,'v2',reversed(range(11)))
    assert len(compare(tmp_path))==11


def test_rejects_duplicate_that_masks_missing_fact(tmp_path):
    make(tmp_path,'v1',range(11))
    make(tmp_path,'v2',[0,*range(10)])
    with pytest.raises(ValueError,match='unique'):
        compare(tmp_path)
