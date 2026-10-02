#!/usr/bin/env python3
"""Independent certificate replay. Pure Python, no generator/solver imports.

Rectangle sums use direct nested row/column loops, never a shared move scanner,
NumPy, prefix sums, or a solver. Also accepts externally returned solver paths.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

class InvalidCertificate(ValueError): pass

def require(condition,message):
    if not condition: raise InvalidCertificate(message)

def canonical_hash(obj):
    raw=json.dumps(obj,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()

def replay(board,path,require_full=True,expected_counts=None,include_trace=False):
    rows=board['rows'];cols=board['cols'];values=board['values']
    require(type(rows) is int and type(cols) is int and rows>0 and cols>0,'invalid dimensions')
    require(len(values)==rows*cols,'value count differs from board size')
    require(all(type(v) is int and 1<=v<=9 for v in values),'digits must be integers 1..9')
    require(board.get('initial_active',[1]*(rows*cols))==[1]*(rows*cols),'corpus must start fully active')
    if 'grid' in board:
        require(board['grid']==[''.join(map(str,values[r*cols:(r+1)*cols])) for r in range(rows)],'grid/values mismatch')
    if expected_counts is not None: require(len(path)==len(expected_counts),'removed-count length mismatch')
    active=[True]*(rows*cols)
    removal_step=[None]*(rows*cols)
    dependencies=[];depths=[];move_stats=[];score=0
    for step,rect in enumerate(path):
        require(isinstance(rect,(list,tuple)) and len(rect)==4,f'move {step}: invalid rectangle')
        require(all(type(x) is int for x in rect),f'move {step}: noninteger coordinate')
        r1,c1,r2,c2=rect
        require(0<=r1<=r2<rows and 0<=c1<=c2<cols,f'move {step}: out of bounds')
        total=0;indices=[];blockers=set();initial_total=0
        for r in range(r1,r2+1):
            for c in range(c1,c2+1):
                i=r*cols+c
                initial_total+=values[i]
                if active[i]:
                    total+=values[i];indices.append(i)
                else:
                    blockers.add(removal_step[i])
        require(total==10,f'move {step}: rectangle sum is {total}, expected 10')
        require(len(indices)>=2,f'move {step}: fewer than two cells')
        if expected_counts is not None:
            require(len(indices)==expected_counts[step],f'move {step}: removed-count mismatch')
        anchored=(active[r1*cols+c1] and active[r2*cols+c2]) or (active[r1*cols+c2] and active[r2*cols+c1])
        depth=1+max((depths[x] for x in blockers),default=0)
        depths.append(depth);dependencies.append(sorted(blockers))
        area=(r2-r1+1)*(c2-c1+1)
        move_stats.append({'step':step,'rectangle':list(rect),'dependent_on':sorted(blockers),'removed':len(indices),'area':area,'empty_cells_inside':area-len(indices),
                          'height':r2-r1+1,'width':c2-c1+1,'anchored':bool(anchored),
                          'initially_legal':initial_total==10,'dependency_depth':depth})
        for i in indices:
            active[i]=False;removal_step[i]=step
        score+=len(indices)
    remaining=[i for i,flag in enumerate(active) if flag]
    if require_full: require(not remaining,f'{len(remaining)} cells remain')
    result={'score':score,'remaining_indices':remaining,'move_count':len(path),
            'digit_counts':dict(sorted(Counter(values).items())),
            'mean_digit':sum(values)/len(values),'total_value':sum(values),
            'removed_count_histogram':dict(sorted(Counter(m['removed'] for m in move_stats).items())),
            'anchored_moves':sum(m['anchored'] for m in move_stats),
            'unanchored_moves':sum(not m['anchored'] for m in move_stats),
            'dependent_moves':sum(not m['initially_legal'] for m in move_stats),
            'dependency_edges':sum(map(len,dependencies)),
            'max_dependency_depth':max(depths,default=0),
            'long_range_moves':sum(max(m['height'],m['width'])>=8 and m['empty_cells_inside']>0 for m in move_stats),
            'two_dimensional_moves':sum(m['height']>1 and m['width']>1 for m in move_stats),
            'max_rectangle_area':max((m['area'] for m in move_stats),default=0),
            'max_span':max((max(m['height'],m['width']) for m in move_stats),default=0),
            'max_empty_cells_inside':max((m['empty_cells_inside'] for m in move_stats),default=0)}
    if include_trace:result['move_annotations']=move_stats
    return result

def summarize(records):
    keys=('score','move_count','mean_digit','anchored_moves','unanchored_moves','dependent_moves',
          'dependency_edges','max_dependency_depth','long_range_moves','two_dimensional_moves',
          'max_rectangle_area','max_span','max_empty_cells_inside')
    summary={'boards':len(records),'verified_full_clear':all(x['score']==160 for x in records)}
    for key in keys:
        values=[x[key] for x in records]
        summary[key]={'min':min(values),'mean':statistics.mean(values),'max':max(values)}
    counts=Counter()
    for record in records: counts.update(record['digit_counts'])
    summary['digit_counts']=dict(sorted(counts.items(),key=lambda x:int(x[0])))
    return summary

def check(root):
    manifest=json.loads((root/'manifest.json').read_text())
    output={'checker':'pure_python_nested_loop_rectangle_replay','splits':{}}
    seen_boards=set();seen_seeds=set()
    for split,meta in manifest['splits'].items():
        boards=json.loads((root/split/'boards.json').read_text())
        certs=json.loads((root/split/'certificates.json').read_text())
        seeds=json.loads((root/split/'seeds.json').read_text())
        require(len(boards)==len(certs)==len(seeds)==meta['count'],'split size mismatch')
        for name,items in (('boards',boards),('certificates',certs),('seeds',seeds)):
            require(canonical_hash(items)==meta[name+'_sha256'],f'{split}: hash mismatch for {name}')
        records=[];annotations=[]
        for board,cert,seed in zip(boards,certs,seeds):
            require(board['id']==cert['id']==seed['id'],'identifier mismatch')
            require(board['rows']==16 and board['cols']==10,'unexpected board dimensions')
            require(board['certified_optimum']==160,'incorrect optimum label')
            require(cert['seed']==seed['seed'],'seed mismatch')
            require(canonical_hash(board)==cert['board_sha256'],'certificate board hash mismatch')
            b=tuple(board['values'])
            require(b not in seen_boards,'duplicate board across corpus');seen_boards.add(b)
            require(seed['seed'] not in seen_seeds,'duplicate seed across corpus');seen_seeds.add(seed['seed'])
            result=replay(board,cert['path'],expected_counts=cert['removed_counts'],include_trace=True)
            annotations.append({'id':board['id'],'moves':result.pop('move_annotations')})
            result.update(id=board['id'],family=board['family'])
            records.append(result)
        by_family={family:summarize([r for r in records if r['family']==family]) for family in manifest['families']}
        require(all(s['boards']==meta['per_family'] for s in by_family.values()),'family balance mismatch')
        output['splits'][split]={'overall':summarize(records),'by_family':by_family}
        # Individual holdout structural records remain sealed alongside its data.
        (root/split/'validation.json').write_text(json.dumps(records,indent=2)+'\n')
        (root/split/'move_annotations.json').write_text(json.dumps(annotations,indent=2)+'\n')
    (root/'validation_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();result=check(args.root)
    for split,info in result['splits'].items():
        s=info['overall']
        print(f"{split}: {s['boards']} boards, full clear={s['verified_full_clear']}; mean moves={s['move_count']['mean']:.1f}, mean dependent moves={s['dependent_moves']['mean']:.1f}, max span={s['max_span']['max']}, unanchored moves={s['unanchored_moves']}")

if __name__=='__main__':main()
