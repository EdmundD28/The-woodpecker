"""Fixed synthetic perturbation audit, not a measurement in a new venue."""
import json
from pathlib import Path
import sys
import numpy as np


def main():
    repo=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(repo))
    from cnn_analysis.quality import predict
    from cnn_analysis.export_quality import venue_model
    run=repo/'firmware_b_demo/test/fixtures/quality'
    model=json.loads((run/'quality_model.json').read_text(encoding='utf-8'))
    venue_profile=venue_model(run/'quality_model.json')
    normal=[json.loads(s) for s in (run/'normal_test.jsonl').read_text(encoding='utf-8').splitlines()]
    configs=[('unchanged',1,0,0),('offset_plus_150',1,150,0),('offset_minus_150',1,-150,0),
             ('gain_0.7',.7,0,0),('gain_1.3',1.3,0,0),('noise_10adc',1,0,10),
             ('noise_30adc',1,0,30),('noise_80adc',1,0,80),('gain_0.7_noise_30adc',.7,0,30)]
    results=[]
    for name,gain,offset,noise in configs:
        rng=np.random.default_rng(420)
        strict=venue=hard=0
        for r in normal:
            a=np.array(r['raw'],dtype=float); mean=a.mean()
            values=np.rint(mean+(a-mean)*gain+offset+rng.normal(0,noise,len(a)))
            adjusted={**r,'raw':np.clip(values,0,4095).astype(int).tolist()}
            p=predict(adjusted,model)
            v=predict(adjusted,venue_profile)
            hard_reject=p['reason']=='INVALID_CAPTURE'
            hard+=hard_reject; strict+=p['retest']
            venue+=int(hard_reject or v['score']>v['threshold']*1.5)
        results.append({'scenario':name,'records':len(normal),'strict_retests':strict,'venue_retests':venue,'basic_retests':hard})
    out=repo/'firmware_b_demo/.host_tests'; out.mkdir(exist_ok=True)
    (out/'venue_sensitivity.json').write_text(json.dumps({'note':'synthetic perturbations only; no threshold tuning; no proof of new-venue accuracy','results':results},indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
