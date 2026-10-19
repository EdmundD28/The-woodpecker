"""Package approved frozen models without retraining or firmware modification."""
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    root=Path(__file__).resolve().parent
    source=root/'runs/20261019_guarded_broad_robust_spectrum_v3'
    follow=root/'runs/20261019_conservative_frequency_v4'
    release=root/'releases/20261019_guarded_broad_mild_point_vote_v1'
    release.mkdir(parents=True,exist_ok=False)
    evidence=release/'evidence';evidence.mkdir()
    models={};results=[]
    for seed in [42,123,2026]:
        parent=follow/f'mixed_seed{seed}';model=read(parent/'model.json');result=read(parent/'result.json')
        validation=read(parent/'validation.json')['mild_response']['clean']
        name=f'frequency_seed{seed}.json'
        exported={'format_version':'WOOD_POINT_MEDIAN_FREQUENCY_V1','config':model['config'],
                  'point_networks':model['point_networks']['mild_response'],'aggregation':'median','joint_model_weight':0,
                  'training_recipe':'mild_response; train-only spectral response log std0.10; no real abnormal training',
                  'source_sha256':hashlib.sha256((parent/'model.json').read_bytes()).hexdigest()}
        write(release/name,exported)
        models[name]={'seed':seed,'validation_loss':validation['cross_entropy'],
                      'sha256':hashlib.sha256((release/name).read_bytes()).hexdigest()}
        results.append(result)
    # Deployment artifact selection uses the existing validation scores only.
    default=min(models,key=lambda n:models[n]['validation_loss'])
    (release/'quality.json').write_bytes((source/'quality_model.json').read_bytes())
    quality_hash=hashlib.sha256((release/'quality.json').read_bytes()).hexdigest()
    assert quality_hash==read(follow/'protocol_locked.json')['quality_preserved_sha256']
    for origin,target in [(source/'quality_summary.json','quality_summary.json'),
                          (source/'verification.json','v3_verification.json'),
                          (source/'quality_runtime_verification.json','quality_runtime_verification.json'),
                          (source/'recognition_comparison.csv','v3_recognition_comparison.csv'),
                          (follow/'summary.json','v4_summary.json'),
                          (follow/'numerical_verification.json','v4_numerical_verification.json'),
                          (follow/'all11_real_abnormal_comparison.csv','all11_abnormal_comparison.csv'),
                          (follow/'generic_stress_comparison.csv','generic_stress_comparison.csv'),
                          (root/'data/task_quality_labels_20261019.csv','task_quality_labels.csv')]:
        (evidence/target).write_bytes(origin.read_bytes())
    write(evidence/'mild_frequency_seed_results.json',results)
    current={'release_id':release.name,'release_directory':'releases/'+release.name,
             'status':'USER_APPROVED_REMOTE_MODEL_DELIVERY','default_frequency':default,'frequency_models':models,
             'quality_file':'quality.json','quality_sha256':quality_hash,
             'default_selection':'lowest existing clean source-validation cross entropy among three optimized mild-response seeds; no retraining or real-abnormal tuning',
             'adopted_frequency_recipe':'mild_response',
             'rejected_recipes':['V2 quality revision','mild_plus_bands'],
             'classification_acceptance':'90% minimum and95% target apply to quality-usable inputs; forced abnormal insertion is a comparative stress test, not a release veto',
             'firmware_updated':False,'hardware_flashed':False,'strict_profile_enabled':False,
             'quality_before_classification':True,
             'limits':'retrospective two-specimen/three-condition evidence; novel real scenes and abnormal families not guaranteed; forced abnormal insertion scores are reported separately'}
    write(release/'manifest.json',current)
    write(root/'current_release.json',current)
    print(json.dumps({'release':str(release),'default_frequency':default,'models':len(models),'quality_preserved':True}))


if __name__=='__main__':main()
