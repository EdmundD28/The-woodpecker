"""Compile actual quality gate and compare every saved record to Python oracle."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]; repo=root.parent
    sys.path.insert(0,str(repo))
    from luo_analysis.quality import features, predict
    from luo_analysis.export_quality import venue_model
    parser=argparse.ArgumentParser()
    parser.add_argument('--compiler',type=Path,default=repo.parent/'tmp/model-cpp/ziglang/zig.exe')
    parser.add_argument('--run',type=Path,default=root/'test/fixtures/quality')
    parser.add_argument('--prepare-only',action='store_true')
    a=parser.parse_args(); out=root/'.host_tests'; out.mkdir(exist_ok=True)
    temp=out/'tmp'; temp.mkdir(exist_ok=True)
    model=json.loads((a.run/'quality_model.json').read_text(encoding='utf-8'))
    venue=venue_model(a.run/'quality_model.json')
    records=[]; labels=[]
    for name in ['train','calibration','normal_test','abnormal_test']:
        rows=[json.loads(s) for s in (a.run/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        records+=rows; labels += [name]*len(rows)
    def nested(rows): return ',\n'.join('{'+','.join(map(str,row))+'}' for row in rows)
    fs=[features(r,model['config']) for r in records]
    predictions=[predict(r,model) for r in records]
    venue_predictions=[predict(r,venue) for r in records]
    header=(f'static const unsigned quality_count={len(records)};\n'
            'static const uint16_t quality_raw[][2400]={'+nested([r['raw'] for r in records])+'};\n'
            'static const float quality_expected_features[][27]={'+nested([[f'{v:.9e}f' for v in row] for row in fs])+'};\n'
            'static const float quality_expected_scores[]={'+','.join(f'{p["score"]:.9e}f' for p in predictions)+'};\n'
            'static const float quality_venue_scores[]={'+','.join(f'{p["score"]:.9e}f' for p in venue_predictions)+'};\n'
            'static const unsigned quality_slots[]={'+','.join(str(r['impact_point_id']-1) for r in records)+'};\n'
            'static const unsigned quality_partition[]={'+','.join(str(['train','calibration','normal_test','abnormal_test'].index(n)) for n in labels)+'};\n')
    (out/'quality_vectors.h').write_text(header,encoding='utf-8')
    if a.prepare_only: return
    source=r'''#include "signal_quality.h"
#include "quality_vectors.h"
#include <cassert>
#include <cstdio>
int main() {
    static float workspace[signal_quality::workspace_floats];
    float feature[27],maxFeature=0,maxScore=0; unsigned rejects[2][4]={};
    for(unsigned i=0;i<quality_count;++i) {
        assert(signal_quality::features(quality_raw[i],feature,workspace));
        for(unsigned j=0;j<27;++j) {
            float error=std::fabs(feature[j]-quality_expected_features[i][j]);
            if(error>maxFeature) maxFeature=error;
        }
        for(unsigned policy=0;policy<2;++policy) {
            auto p=policy?signal_quality::Policy::Venue:signal_quality::Policy::Strict;
            auto r=signal_quality::evaluate(quality_raw[i],2400,false,quality_slots[i],p,workspace);
            const float expected=policy?quality_venue_scores[i]:quality_expected_scores[i];
            float error=std::fabs(r.score-expected); if(error>maxScore) maxScore=error;
            assert((r.problem!=nullptr)==(expected>r.threshold));
            rejects[policy][quality_partition[i]]+=!r.passed();
            if(quality_slots[i]!=1) {
                auto reverse=signal_quality::evaluate(quality_raw[i],2400,false,2-quality_slots[i],p,workspace);
                assert(reverse.score==r.score && reverse.passed()==r.passed());
            }
        }
    }
    assert(maxFeature<2e-4f && maxScore<2e-4f);
    assert(rejects[0][2]==0 && rejects[1][2]==0 && rejects[0][3]==5 && rejects[1][3]==5);
    std::printf("{\"records\":%u,\"max_feature_error\":%.9g,\"max_score_error\":%.9g,\"strict_normal_retests\":%u,\"venue_normal_retests\":%u,\"strict_abnormal_retests\":%u,\"venue_abnormal_retests\":%u}\n",
        quality_count,double(maxFeature),double(maxScore),rejects[0][2],rejects[1][2],rejects[0][3],rejects[1][3]);
}
'''
    cpp=out/'verify_quality.cpp'; cpp.write_text(source,encoding='utf-8')
    binary=out/'verify_quality.exe'
    env={**os.environ,'ZIG_GLOBAL_CACHE_DIR':str(out/'zig-global'),'ZIG_LOCAL_CACHE_DIR':str(out/'zig-local'),
         'TEMP':str(temp),'TMP':str(temp)}
    command=[str(a.compiler)]+(['c++'] if a.compiler.stem=='zig' else [])
    command+=['-std=c++17','-O2','-UNDEBUG','-I',str(root/'include'),'-I',str(out),str(cpp),'-o',str(binary)]
    subprocess.run(command,check=True,env=env)
    run=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
    report=json.loads(run.stdout)
    (out/'quality_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
