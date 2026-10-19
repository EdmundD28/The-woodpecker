"""Run the approved broad quality gate and retained point-median frequency model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import preprocess, validate_config
from .network import CNN
from .quality import predict as quality_predict
from .spectrum_preparation import rms_log


ROOT=Path(__file__).resolve().parent


class ReleasedPredictor:
    def __init__(self,seed=None):
        self.manifest=json.loads((ROOT/'current_release.json').read_text(encoding='utf-8'))
        folder=(ROOT/self.manifest['release_directory']).resolve()
        if not folder.is_relative_to(ROOT/'releases'):raise ValueError('invalid release directory')
        self.release_directory=folder
        name=self.manifest['default_frequency'] if seed is None else f'frequency_seed{seed}.json'
        if name not in self.manifest['frequency_models']:raise ValueError('unknown released seed')
        def verified(filename,digest):
            data=(folder/filename).read_bytes()
            if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('model checksum mismatch: '+filename)
            return json.loads(data)
        self.model=verified(name,self.manifest['frequency_models'][name]['sha256'])
        self.quality=verified(self.manifest['quality_file'],self.manifest['quality_sha256'])
        if self.model['format_version']!='WOOD_POINT_MEDIAN_FREQUENCY_V1' or self.model['aggregation']!='median':raise ValueError('unsupported frequency model')
        self.config=self.model['config'];validate_config(self.config)
        if len(self.quality['feature_names'])!=25:raise ValueError('approved broad quality only')
        self.nets={name:CNN({**self.config,'channels':1},weights) for name,weights in self.model['point_networks'].items()}
        self.model_file=name

    @staticmethod
    def _ordered(records):
        if len(records)!=3 or sorted(r.get('impact_point_id',0) for r in records)!=[1,2,3]:raise ValueError('exactly one record for each of points1,2,3 required')
        return sorted(records,key=lambda r:r['impact_point_id'])

    def _frequency_scores(self,records):
        rows=self._ordered(records)
        x=np.stack([preprocess(r,self.config)[:,0] for r in rows],axis=1)[None,:,:]
        spectrum=rms_log(x,self.config,self.config['spectrum_preparation'])
        pp=[]
        for point in range(3):
            net=self.nets['middle' if point==1 else 'endpoint']
            pp.append(net.forward(spectrum[:,:,point:point+1])[0])
        result=np.median(np.stack(pp),axis=0)
        return result/result.sum()

    def predict(self,records):
        rows=self._ordered(records)
        quality=[quality_predict(r,self.quality) for r in rows]
        base={'release_id':self.manifest['release_id'],'model_file':self.model_file,'quality':quality}
        if any(p['retest'] for p in quality):
            return {**base,'status':'RETEST','retest_points':[r['impact_point_id'] for r,p in zip(rows,quality) if p['retest']]}
        p=self._frequency_scores(rows);index=int(p.argmax())
        return {**base,'status':'CLASSIFIED','class':self.config['classes'][index],
                'class_display_name':self.config['class_display_names'][index],'scores':p.tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',required=True,help='JSONL file containing one new raw capture per point')
    p.add_argument('--seed',type=int,choices=[42,123,2026],default=None)
    args=p.parse_args();path=Path(args.records)
    records=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    print(json.dumps(ReleasedPredictor(args.seed).predict(records),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
