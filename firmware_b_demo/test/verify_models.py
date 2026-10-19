"""Verify actual firmware against all six saved training-export oracle triplets."""
import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    workspace = root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=Path, default=workspace/'tmp/model-cpp/ziglang/zig.exe')
    parser.add_argument('--run', type=Path, default=root/'test/fixtures/recognition', help='Directory containing time/ and spectrum/ model.json and fixed_test_vectors.json.')
    args = parser.parse_args()
    subprocess.run([sys.executable,str(root/'test/verify_quality.py'),'--prepare-only'],check=True)
    out = root/'.host_tests'
    out.mkdir(exist_ok=True)
    temp = out/'tmp'
    temp.mkdir(exist_ok=True)
    env = {**os.environ, 'ZIG_GLOBAL_CACHE_DIR': str(out/'zig-global'),
           'ZIG_LOCAL_CACHE_DIR': str(out/'zig-local'), 'TEMP': str(temp), 'TMP': str(temp)}
    reports = []
    manifest=json.loads((root/'include/models/manifest.json').read_text(encoding='utf-8'))
    config=configparser.ConfigParser()
    config.read(root/'platformio.ini',encoding='utf-8')
    default_env=config['platformio']['default_envs'].strip()
    if manifest['default']!='spectrum' or config['env:'+default_env]['build_flags'].strip()!='-DWOOD_USE_SPECTRUM=1':
        raise ValueError('default build must select the adopted spectrum model')
    for build, flag in [('time', '0'), ('spectrum', '1'), ('default', None)]:
        mode=manifest['default'] if build=='default' else build
        source=args.run
        header=root/'include/models'/f'{mode}_weights.h'
        if hashlib.sha256(header.read_bytes()).hexdigest()!=manifest['models'][mode]['header_sha256']:
            raise ValueError('deployed header hash mismatch: '+mode)
        source_model=json.loads((source/mode/'model.json').read_text(encoding='utf-8'))
        if source_model['model_version']!=manifest['models'][mode]['version']:
            raise ValueError('oracle model version mismatch: '+mode)
        vectors = json.loads((source/mode/'fixed_test_vectors.json').read_text(encoding='utf-8'))
        raw = ',\n'.join('{'+','.join('{'+','.join(map(str, r['raw']))+'}' for r in v['records'])+'}' for v in vectors)
        inputs = ',\n'.join('{'+','.join(f'{x:.9e}f' for row in v['preprocessed_input'] for x in row)+'}' for v in vectors)
        scores = ',\n'.join('{'+','.join(f'{x:.9e}f' for x in v['expected_scores'])+'}' for v in vectors)
        length = len(vectors[0]['preprocessed_input'])*3
        (out/'reference_vectors.h').write_text(
            f'constexpr size_t vector_count={len(vectors)};\n'
            f'static const uint16_t test_raw[{len(vectors)}][3][2400]={{{raw}}};\n'
            f'static const float expected_input[{len(vectors)}][{length}]={{{inputs}}};\n'
            f'static const float expected_scores[{len(vectors)}][2]={{{scores}}};\n', encoding='utf-8')
        binary = out/f'test_{build}.exe'
        command = [str(args.compiler)]
        if args.compiler.stem == 'zig':
            command += ['c++']
        if flag is not None:
            command += [f'-DWOOD_USE_SPECTRUM={flag}']
        command += ['-std=c++17', '-O2', '-UNDEBUG',
                    '-I', str(root/'test/host'), '-I', str(root/'include'), '-I', str(out),
                    str(root/'test/host/test_firmware.cpp'), '-o', str(binary)]
        subprocess.run(command, check=True, env=env)
        result = subprocess.run([str(binary)], capture_output=True, text=True)
        print(result.stdout, end='', flush=True)
        if result.returncode:
            raise RuntimeError(f'{mode} firmware test failed ({result.returncode}): {result.stderr}')
        reports.append({'build': build, 'mode': mode, 'passed': True, 'oracle_triplets': len(vectors), 'output': result.stdout})
    (out/'verification.json').write_text(json.dumps(reports, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
