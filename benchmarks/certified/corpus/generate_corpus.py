#!/usr/bin/env python3
"""Geometry-first certified Sum10 instances. Never imports or invokes a solver."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

ROWS, COLS = 16, 10
FAMILIES = ('rectangle_carving', 'orthogonal_weave', 'layered_anchors')
ROOT = Path(__file__).resolve().parent

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()

def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()

def bbox(cells):
    return [min(p[0] for p in cells), min(p[1] for p in cells),
            max(p[0] for p in cells), max(p[1] for p in cells)]

def inside(p, box):
    a,b,c,d = box
    return a <= p[0] <= c and b <= p[1] <= d

def occupied(active, box):
    return sorted(p for p in active if inside(p, box))

def composition(rng, size):
    # Uniform over ordered positive compositions of 10 into size parts.
    # With size >= 2, the maximum is automatically <= 9.
    cuts = [0] + sorted(rng.sample(range(1,10), size-1)) + [10]
    values = [cuts[i+1]-cuts[i] for i in range(size)]
    rng.shuffle(values)
    return values

def choose_pair_rectangle(active, rng, target):
    """All returned rectangles have two live opposite corners and even occupancy."""
    points = sorted(active)
    # Random sampling plus nearest same-row/column candidates guarantees progress:
    # consecutive nonempty rows and consecutive same-row points provide a 2-cell box.
    pairs = [rng.sample(points,2) for _ in range(220)]
    by_row = {}
    for p in points: by_row.setdefault(p[0], []).append(p)
    for row in by_row.values():
        pairs += list(zip(row, row[1:]))
    nonempty = sorted(by_row)
    for r1,r2 in zip(nonempty,nonempty[1:]):
        pairs.append(min(((a,b) for a in by_row[r1] for b in by_row[r2]),
                         key=lambda ab: abs(ab[0][1]-ab[1][1])))
    choices = {}
    for pair in pairs:
        box = bbox(pair)
        cells = occupied(active,box)
        n = len(cells)
        if n in (2,4,6,8,10):
            choices[tuple(box)] = cells
    if not choices: raise RuntimeError('pair geometry unexpectedly stalled')
    nearest = min(abs(len(c)-target) for c in choices.values())
    options = [(box,c) for box,c in choices.items() if abs(len(c)-target)==nearest]
    # Prefer emptier spanning rectangles once cavities exist, but keep stochasticity.
    weights = [math.sqrt(((b[2]-b[0]+1)*(b[3]-b[1]+1))/len(c)) for b,c in options]
    box,cells = rng.choices(options, weights=weights, k=1)[0]
    return list(box),cells

def choose_stripe(active, rng, target, horizontal):
    lines = {}
    for r,c in sorted(active):
        key = r if horizontal else c
        lines.setdefault(key, []).append((r,c))
    choices = []
    for line in lines.values():
        for n in (2,4,6,8):
            if n<=len(line):
                for start in range(len(line)-n+1):
                    cells = line[start:start+n]
                    choices.append((bbox(cells),cells))
    if not choices:
        return choose_pair_rectangle(active,rng,target)
    nearest = min(abs(len(c)-target) for b,c in choices)
    options = [(b,c) for b,c in choices if abs(len(c)-target)==nearest]
    weights = [math.sqrt(((b[2]-b[0]+1)*(b[3]-b[1]+1))/len(c)) for b,c in options]
    return rng.choices(options,weights=weights,k=1)[0]

def choose_general_rectangle(active,rng,target):
    if len(active)<=8:
        box=bbox(active)
        return box,occupied(active,box)
    choices={}
    for _ in range(350):
        r1,r2=sorted(rng.sample(range(ROWS),2))
        c1,c2=sorted(rng.sample(range(COLS),2))
        cells=occupied(active,(r1,c1,r2,c2))
        if 2<=len(cells)<=8 and len(active)-len(cells)!=1:
            choices[tuple(bbox(cells))]=cells
    # Safe fallback: at least one 2-cell bounding rectangle exists on any >2-cell set.
    if not choices:
        return choose_pair_rectangle(active,rng,2)
    nearest=min(abs(len(c)-target) for c in choices.values())
    options=[(b,c) for b,c in choices.items() if abs(len(c)-target)==nearest]
    weights=[((b[2]-b[0]+1)*(b[3]-b[1]+1))/len(c) for b,c in options]
    box,cells=rng.choices(options,weights=weights,k=1)[0]
    return list(box),cells

def generate(family,seed,identifier):
    rng=random.Random(seed)
    active={(r,c) for r in range(ROWS) for c in range(COLS)}
    groups=[]
    def remove(box,cells):
        assert len(cells)>=2 and len(cells)<=10
        assert set(cells)==set(occupied(active,box))
        groups.append((list(box),sorted(cells)))
        active.difference_update(cells)
    if family=='layered_anchors':
        # Independently varying 2-cell survivor blocks in every row; remove outer
        # even-length runs before allowing arbitrary 2-D dependency-spanning boxes.
        strips=[]
        for r in range(ROWS):
            start=rng.choice((0,2,4,6,8))
            for lo,hi in ((0,start),(start+2,COLS)):
                c=lo
                while c<hi:
                    remaining=hi-c
                    n=rng.choice([x for x in (2,2,2,4,6) if x<=remaining])
                    strips.append(([r,c,r,c+n-1],[(r,j) for j in range(c,c+n)]))
                    c+=n
        rng.shuffle(strips)
        for box,cells in strips: remove(box,cells)
    while active:
        if family=='rectangle_carving':
            target=rng.choices((2,4,6,8),weights=(70,24,5,1))[0]
            box,cells=choose_pair_rectangle(active,rng,target)
        elif family=='orthogonal_weave':
            target=rng.choices((2,4,6,8),weights=(68,25,6,1))[0]
            box,cells=choose_stripe(active,rng,target,horizontal=(len(groups)%2==0))
        else:
            target=rng.choices((2,3,4,5,6,7,8),weights=(34,24,18,10,7,4,3))[0]
            box,cells=choose_general_rectangle(active,rng,target)
        remove(box,cells)
    # Values are assigned after the removal geometry is complete: no value-based
    # search, solvability test, candidate solver, or difficulty rejection is used.
    grid=[[0]*COLS for _ in range(ROWS)]
    for _,cells in groups:
        for (r,c),value in zip(cells,composition(rng,len(cells))): grid[r][c]=value
    record={'id':identifier,'family':family,'rows':ROWS,'cols':COLS,
            'values':[v for row in grid for v in row],
            'grid':[''.join(map(str,row)) for row in grid],
            'initial_active':[1]*(ROWS*COLS),'certified_optimum':ROWS*COLS}
    certificate={'id':identifier,'seed':seed,'board_sha256':digest(record),
                 'path':[b for b,c in groups],'removed_counts':[len(c) for b,c in groups]}
    return record,certificate

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args()
    # Domain-separated, fixed, predeclared seed sets. They are never selected by
    # solver performance. Manifest commits hashes before optimizer sees holdout.
    splits={'warmup':1,'development':4,'sealed_holdout':8}
    manifest={'schema_version':1,'rows':ROWS,'cols':COLS,'families':list(FAMILIES),
              'construction':'geometry_first_positive_compositions', 'splits':{}}
    for split,per_family in splits.items():
        boards=[]; certificates=[]; seeds=[]
        for family in FAMILIES:
            for i in range(per_family):
                seed=int.from_bytes(hashlib.sha256(f'sum10-certified-v1:{split}:{family}:{i}'.encode()).digest()[:8],'big')
                identifier=f'{split}_{family}_{i:02d}'
                record,certificate=generate(family,seed,identifier)
                boards.append(record);certificates.append(certificate)
                seeds.append({'id':identifier,'family':family,'seed':seed})
        write_json(args.output/split/'boards.json',boards)
        write_json(args.output/split/'certificates.json',certificates)
        write_json(args.output/split/'seeds.json',seeds)
        manifest['splits'][split]={'count':len(boards),'per_family':per_family,
               'boards_sha256':digest(boards),'certificates_sha256':digest(certificates),
               'seeds_sha256':digest(seeds)}
    write_json(args.output/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
