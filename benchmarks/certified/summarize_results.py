"""Derive all report aggregates from recorded per-trial measurements."""
import json,statistics,hashlib,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def summarize(name):
    p=ROOT/'results'/name
    if not p.exists():return None
    trials=json.loads(p.read_text())['trials'];out={}
    for variant in sorted({r['variant'] for r in trials}):
        a=[r for r in trials if r['variant']==variant];times=[r['end_to_end_seconds'] for r in a]
        out[variant]={'trials':len(a),'full_clears':sum(r['full_clear'] for r in a),'proven_optima':sum(r['optimal'] for r in a),'target_144_reached':sum(r['score']>=144 for r in a),'scores':[r['score'] for r in a],'seconds':times,'median_seconds':statistics.median(times),'min_seconds':min(times),'max_seconds':max(times),'mean_score':statistics.mean(r['score'] for r in a)}
    return out

def main():
    result={'warm_heldout':summarize('heldout_primary.json'),'matched_budget':summarize('heldout_budget.json'),'cold':summarize('cold_comparison.json'),'arena_target':summarize('arena_time_to_144.json'),'arena_budget':summarize('arena_budget.json'),'arena_legacy_hydra':summarize('arena_legacy_hydra.json'),'arena_extended':summarize('arena_extended_30s.json')}
    trials=json.loads((ROOT/'results/heldout_primary.json').read_text())['trials'];d={(r['case'],r['seed'],r['variant']):r for r in trials};common=[r for r in trials if r['variant']=='previous' and r['full_clear']]
    result['paired_success_only']={'n':len(common)}
    for v in ('original','previous'):
        ratio=[d[(r['case'],r['seed'],v)]['end_to_end_seconds']/d[(r['case'],r['seed'],'candidate')]['end_to_end_seconds'] for r in common]
        result['paired_success_only'][v]={'median_speedup':statistics.median(ratio),'min_speedup':min(ratio),'max_speedup':max(ratio)}
    result['by_family']={}
    for f in sorted({r['family'] for r in trials}):
        result['by_family'][f]={v:{'runs':sum(r['family']==f and r['variant']==v for r in trials),'full_clear':sum(r['full_clear'] for r in trials if r['family']==f and r['variant']==v),'median_seconds':statistics.median(r['end_to_end_seconds'] for r in trials if r['family']==f and r['variant']==v)} for v in ('original','previous','candidate')}
    import numpy,numba
    result['environment']={'python':platform.python_version(),'numpy':numpy.__version__,'numba':numba.__version__,'platform':platform.platform(),'cpu':'AMD EPYC9V74 (cloud reports9 logical CPUs)','measurement_workers':1,'blas_threads':1,'numba_threads':1}
    result['freeze']=json.loads((ROOT/'evaluation/candidate_freeze.json').read_text())
    (ROOT/'results/summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('freeze','warm_heldout')},indent=2))
if __name__=='__main__':main()
