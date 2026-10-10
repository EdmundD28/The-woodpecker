"""Optional host C++ check of exported preprocessing and scores (not hardware QA)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from .core import dump


def verify(run, compiler='c++', zig=False):
    run=Path(run).resolve()
    model=json.loads((run/'model.json').read_text(encoding='utf-8'))
    vectors=json.loads((run/'fixed_test_vectors.json').read_text(encoding='utf-8'))
    budget=json.loads((run/'deployment_budget.json').read_text(encoding='utf-8'))
    if not vectors:
        raise ValueError('no fixed test vectors')
    mode=model['config'].get('input_mode','time')
    n=len(vectors[0]['preprocessed_input'])
    out=run/'cpp_checks'; out.mkdir(exist_ok=True)
    raw=',\n'.join('{'+','.join(str(v) for v in a['record']['raw'])+'}' for a in vectors)
    inp=',\n'.join('{'+','.join(f'{v:.9e}f' for v in a['preprocessed_input'])+'}' for a in vectors)
    scores=',\n'.join('{'+','.join(f'{v:.9e}f' for v in a['expected_scores'])+'}' for a in vectors)
    header=(run/'model_weights.h').as_posix()
    text=f'''#include "{header}"
#include <cstdio>
#include <cmath>
static const uint16_t raw[{len(vectors)}][1024]={{ {raw} }};
static const float expected_input[{len(vectors)}][{n}]={{ {inp} }};
static const float expected_scores[{len(vectors)}][2]={{ {scores} }};
int main() {{
    static float input[{n}], scratch[{budget['scratch_bytes']//4}];
    float max_input=0,max_score=0;
    bool labels_match=true;
    for(int row=0;row<{len(vectors)};++row) {{
        wood_model_{mode}::preprocess(raw[row],input);
        for(int i=0;i<{n};++i) {{
            if(!std::isfinite(input[i])) return 2;
            float e=std::fabs(input[i]-expected_input[row][i]);
            if(e>max_input) max_input=e;
        }}
        float scores[2]; wood_model_{mode}::infer(input,scores,scratch);
        for(int i=0;i<2;++i) {{
            if(!std::isfinite(scores[i])) return 2;
            float e=std::fabs(scores[i]-expected_scores[row][i]);
            if(e>max_score) max_score=e;
        }}
        if((scores[1]>scores[0])!=(expected_scores[row][1]>expected_scores[row][0])) labels_match=false;
    }}
    bool passed=max_input<=1e-7f && max_score<=1e-4f && labels_match;
    std::printf("{{\\\"max_input_error\\\":%.9g,\\\"max_score_error\\\":%.9g,\\\"labels_match\\\":%s,\\\"passed\\\":%s}}\\n",
                 double(max_input),double(max_score),labels_match?"true":"false",passed?"true":"false");
    return passed?0:1;
}}
'''
    source=out/'check.cpp'; source.write_text(text,encoding='utf-8')
    binary=out/('check.exe' if os.name=='nt' else 'check')
    command=[compiler]+(['c++'] if zig else [])+['-std=c++17','-O2',str(source),'-o',str(binary)]
    compiled=subprocess.run(command,capture_output=True,text=True)
    if compiled.returncode:
        (out/'compiler_errors.txt').write_text(compiled.stderr,encoding='utf-8')
        raise RuntimeError('host C++ compilation failed; see cpp_checks/compiler_errors.txt')
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    if not result.stdout.strip():
        raise RuntimeError('host C++ test failed without JSON output: '+result.stderr)
    report=json.loads(result.stdout)
    report.update(input_mode=mode,model_version=model['model_version'],vectors=len(vectors),
                  verification_scope='HOST C++ ONLY; ESP32 pending')
    dump(out/'verification.json',report)
    if result.returncode:
        raise RuntimeError('host C++ mismatch; see cpp_checks/verification.json')
    print(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description='Optional exported C++ reference check')
    p.add_argument('run'); p.add_argument('--compiler',default='c++'); p.add_argument('--zig',action='store_true')
    a=p.parse_args(); verify(a.run,a.compiler,a.zig)


if __name__=='__main__': main()
