"""Derive paired wall/event statistics from accepted raw Stage3 samples."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
OUT=ROOT/'performance/fasteq_optimization/v7/uniform'
CAND=ROOT/'src/FastEq_v7'
CASES=('o2_f32_warp','o2_f64_warp','o3_f32_warp','o3_f64_warp')


def ratio(samples):
    q=statistics.quantiles(samples,n=4,method='inclusive')
    return {'rounds':samples,'median':statistics.median(samples),'q25':q[0],'q75':q[2]}


def main():
    summary={'status':'awaiting remaining representative cases and root final QA','cases':{},
             'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'current_production_sha256':{name:hashlib.sha256((CAND/name).read_bytes()).hexdigest()
                                        for name in ('fasteq/adapters/tensor_product.py','fasteq/ops/segmented_product.py')}}
    for name in CASES:
        path=OUT/(name+'.json')
        if not path.exists():
            summary['cases'][name]={'status':'no accepted samples yet'};continue
        d=json.loads(path.read_text())
        entry={'file':str(path),'status':d['status'],'dtype':d['dtype'],'metadata':d['metadata'],
               'executed_source_sha256':d['source_sha256'],'precision':d['precision'],
               'wall_medians_ms':d['wall_medians_ms'],'event_medians_ms':d['event_medians_ms'],
               'timer_protocol':d['timer_protocol'],'paired_wall_ratios':d['paired_round_ratios'],
               'paired_event_ratios':{k+'/new':ratio([a/b for a,b in zip(d['cuda_event_ms'][k],d['cuda_event_ms']['new'],strict=True)])
                                      for k in ('original','legacy')},
               'ratio_of_wall_medians':{k+'/new':d['wall_medians_ms'][k]/d['wall_medians_ms']['new']for k in ('original','legacy')},
               'ratio_of_event_medians':{k+'/new':d['event_medians_ms'][k]/d['event_medians_ms']['new']for k in ('original','legacy')}}
        summary['cases'][name]=entry
    if all(case['status']=='ok'for case in summary['cases'].values()):
        summary['status']='four representative normal cases complete; root final affected QA tracked separately'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({name:{'status':case['status'],'wall':case.get('wall_medians_ms'),
                            'event':case.get('event_medians_ms'),'paired_event':case.get('paired_event_ratios')}
                      for name,case in summary['cases'].items()},indent=2))


if __name__=='__main__':main()
