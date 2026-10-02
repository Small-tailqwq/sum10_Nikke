"""Target JSON ingress to independently validated JSON egress benchmark.
Warm workers finish import/JIT on OTHER boards before target ingress.
No witness or generator metadata is available to solver workers.
"""
import argparse, json, os, sys, time, hashlib, statistics, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path(os.environ.get('SUM10_REPO',str(ROOT.parents[1])))
sys.path.insert(0,str(REPO/'benchmarks'))

def upper_bound(board):
    digits=[v for row in board for v in row if v>0]
    costs=[0]+[10**9]*9
    for v in digits:
        nxt=costs.copy()
        for residue,n in enumerate(costs):
            nr=(residue+v)%10; nxt[nr]=min(nxt[nr],n+1)
        costs=nxt
    return len(digits)-costs[sum(digits)%10]

def verify(board,result):
    rows,cols=len(board),len(board[0]); cells=[list(row) for row in board]
    score=0
    for move in result['path']:
        if not isinstance(move,(list,tuple)) or len(move)!=4: raise ValueError('bad rect')
        r1,c1,r2,c2=map(int,move)
        if not (0<=r1<=r2<rows and 0<=c1<=c2<cols): raise ValueError('bounds')
        selected=[(r,c) for r in range(r1,r2+1) for c in range(c1,c2+1) if cells[r][c]>0]
        if sum(cells[r][c] for r,c in selected)!=10: raise ValueError('illegal sum')
        score+=len(selected)
        for r,c in selected: cells[r][c]=0
    if score!=result['score']: raise ValueError('score mismatch')
    full=all(v==0 for row in cells for v in row)
    bound=upper_bound(board)
    optimal=score==bound
    if result.get('optimal') and not optimal: raise ValueError('unsupported optimum claim')
    result['verified']=True; result['optimal']=optimal; result['full_clear']=full; result['upper_bound']=bound
    result['remaining']=sum(v>0 for row in cells for v in row)
    return result

def worker(args):
    import numpy as np
    from solver_support import load_core,solve_base,CORE_NAMES
    if args.variant in ('original','previous','v4'):
        if args.variant=='v4': CORE_NAMES.update({'_fast_scan_rects_v4','_solve_process_beam_search'})
        source=(ROOT/'frozen'/f'{args.variant}_god_brain.py').read_text()
        module=load_core(source,f'_cert_{args.variant}',args.cache_dir)
        # The original has no RNG seeder; the same Numba global RNG can be seeded
        # by the previous version helper, materialized independently.
        seedmod=load_core((ROOT/'frozen'/'previous_god_brain.py').read_text(),'_cert_seed',args.cache_dir)
        def solve(board,beam,seed,budget):
            t=time.perf_counter(); rows,cols=len(board),len(board[0])
            vals=np.array(board,np.int8).ravel(); mask=(vals>0).astype(np.int8)
            best={'score':0,'path':[]}; attempts=0; seed_history=[]
            if args.legacy_hydra and args.variant in ('original','previous'):
                seedmod._seed_search_rng(seed%(2**32-1))
                out=module._solve_process_hydra((mask.tolist(),vals.tolist(),rows,cols,beam,args.mode,seed,budget,{'w_island':63.0,'w_fragment':1.0}))
                return {'score':int(out['score']),'path':out['path'],'attempts':out['iterations']+1,'legacy_hydra':True}
            while attempts==0 or time.perf_counter()-t<budget:
                runseed=(seed+attempts*104729)%(2**32-1)
                seedmod._seed_search_rng(runseed)
                if args.variant=='v4':
                    module.random.seed(runseed)
                    out=module._solve_process_beam_search((mask.tolist(),vals.tolist(),rows,cols,beam,args.mode,runseed))
                else:
                    out=solve_base(module,mask,vals,rows,cols,beam,args.mode,
                      {'w_island':63.0,'w_fragment':1.0})
                attempts+=1; seed_history.append(runseed)
                if out['score']>best['score']: best={'score':int(out['score']),'path':out['path']}
                if best['score']==upper_bound(board) or (args.target_score is not None and best['score']>=args.target_score): break
            best.update(attempts=attempts,seeds=seed_history)
            return best
    else:
        import importlib.util
        spec=importlib.util.spec_from_file_location('fast_solver',str(ROOT/args.candidate))
        module=importlib.util.module_from_spec(spec); sys.modules['fast_solver']=module; spec.loader.exec_module(module)
        def solve(board,beam,seed,budget):
            kwargs={} if args.target_score is None else {'target_score':args.target_score}
            return module.solve(board,beam=beam,seed=seed,time_limit=budget,**kwargs)
    if not args.cold:
        warm=[[1,9,2,8,3,7,4,6,5,5],[5,5,6,4,7,3,8,2,9,1]]
        saved_target=args.target_score; args.target_score=None
        verify(warm,solve(warm,2,9091,30 if args.variant not in ('original','previous','v4') else 0))
        args.target_score=saved_target
        print('READY',flush=True)
    for line in sys.stdin:
        req=json.loads(line)
        result=verify(req['board'],solve(req['board'],req['beam'],req['seed'],req['budget']))
        print(json.dumps(result,separators=(',',':')),flush=True)
        if args.cold: break

def main(args):
    corpus=json.loads(Path(args.boards).read_text())
    if isinstance(corpus,dict): corpus=corpus.get('boards',corpus.get('cases'))
    if args.limit: corpus=corpus[:args.limit]
    variants=args.variants.split(','); trials=[]
    output=Path(args.output)
    for i,case in enumerate(corpus):
        board=case.get('board') or [[int(c) for c in row] for row in case['grid']]; name=case.get('id',case.get('name',str(i)))
        for seed in args.seeds:
            for variant in variants if (i+seed)%2 else variants[::-1]:
                with tempfile.TemporaryDirectory(prefix='sum10-certified-worker-') as cache:
                    if not args.cold:
                        cache=str(ROOT/'module_cache'/variant); Path(cache).mkdir(parents=True,exist_ok=True)
                    cmd=[sys.executable,__file__,'--worker','--variant',variant,'--cache-dir',cache,'--mode',args.mode,'--candidate',args.candidate]
                    if args.cold: cmd+=['--cold']
                    if args.legacy_hydra: cmd+=['--legacy-hydra']
                    if args.target_score is not None: cmd+=['--target-score',str(args.target_score)]
                    env=os.environ.copy(); env['OPENBLAS_NUM_THREADS']='1'; env['OMP_NUM_THREADS']='1'; env['NUMBA_NUM_THREADS']='1'
                    if args.cold: env['NUMBA_CACHE_DIR']=cache
                    else: env['NUMBA_CACHE_DIR']=str(ROOT/'jit_cache'/variant)
                    budget=(0 if variant in ('original','previous','v4') else None) if args.single_pass else args.budget
                    request=json.dumps({'board':board,'beam':args.beam,'seed':seed,'budget':budget},separators=(',',':'))+'\n'
                    if args.cold: start=time.perf_counter()
                    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
                    if not args.cold:
                        ready=proc.stdout.readline()
                        if ready.strip()!='READY': raise RuntimeError(ready+proc.stderr.read())
                        start=time.perf_counter()
                    proc.stdin.write(request); proc.stdin.flush()
                    line=proc.stdout.readline(); elapsed=time.perf_counter()-start
                    proc.stdin.close(); proc.wait()
                    if proc.returncode or not line: raise RuntimeError(proc.stderr.read())
                    result=json.loads(line); verify(board,result)
                    trial={'case':name,'family':case.get('family'),'variant':variant,'seed':seed,'beam':args.beam,'budget_s':budget,'single_pass':args.single_pass,'end_to_end_seconds':elapsed,'cold':args.cold,**result}
                    trials.append(trial)
                    output.write_text(json.dumps({'timing':'before target JSON write (cold: before worker launch) to validated result JSON read; includes input conversion, search, reconstruction, independent validation, JSON serialization and pipe transfer','mode':args.mode,'trials':trials},indent=2)+'\n')
                    print(name,variant,seed,'score',result['score'],'optimal',result['optimal'],'%.5fs'%elapsed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--worker',action='store_true'); p.add_argument('--legacy-hydra',action='store_true'); p.add_argument('--single-pass',action='store_true'); p.add_argument('--target-score',type=int); p.add_argument('--variant'); p.add_argument('--variants',default='original,previous,candidate'); p.add_argument('--cache-dir'); p.add_argument('--candidate',default='fast_solver.py'); p.add_argument('--mode',default='omni'); p.add_argument('--cold',action='store_true'); p.add_argument('--boards'); p.add_argument('--output'); p.add_argument('--limit',type=int); p.add_argument('--beam',type=int,default=200); p.add_argument('--budget',type=float,default=5.); p.add_argument('--seeds',type=int,nargs='+',default=[17]); a=p.parse_args()
    worker(a) if a.worker else main(a)
