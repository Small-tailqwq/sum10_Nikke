#!/usr/bin/env python3
"""A separately committed higher-mean stress extension; original corpus unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from generate_corpus import (ROWS,COLS,bbox,occupied,composition,choose_pair_rectangle,
                             choose_stripe,digest,write_json)

ROOT=Path(__file__).resolve().parent/'high_digit_extension'
FAMILY='high_digit_pair_weave'

def generate(seed,identifier):
    rng=random.Random(seed)
    active={(r,c) for r in range(ROWS) for c in range(COLS)}
    groups=[]
    # At most eight 4-cell moves, all remaining moves pairs: the exact board
    # mean is 5-q/16, in [4.5,5.0], without value-based rejection or solving.
    desired_quartets=rng.randint(4,8)
    quartet_steps=set(rng.sample(range(32),desired_quartets))
    while active:
        step=len(groups);target=4 if step in quartet_steps else 2
        if step%3==0:
            box,cells=choose_stripe(active,rng,target,horizontal=(step%2==0))
        else:
            box,cells=choose_pair_rectangle(active,rng,target)
        if len(cells)>4: box,cells=choose_pair_rectangle(active,rng,2)
        assert len(cells) in (2,4)
        assert set(cells)==set(occupied(active,box))
        groups.append((box,sorted(cells)));active.difference_update(cells)
    grid=[[0]*COLS for _ in range(ROWS)]
    for _,cells in groups:
        for (r,c),value in zip(cells,composition(rng,len(cells))):grid[r][c]=value
    board={'id':identifier,'family':FAMILY,'rows':ROWS,'cols':COLS,
           'values':[v for row in grid for v in row],
           'grid':[''.join(map(str,row)) for row in grid],
           'initial_active':[1]*(ROWS*COLS),'certified_optimum':160}
    assert 4.5<=sum(board['values'])/160<=5
    certificate={'id':identifier,'seed':seed,'board_sha256':digest(board),
                 'path':[b for b,c in groups],'removed_counts':[len(c) for b,c in groups]}
    return board,certificate

def main(output=ROOT):
    manifest={'schema_version':1,'rows':ROWS,'cols':COLS,'families':[FAMILY],
              'construction':'geometry_first_positive_compositions_high_digit_stress_extension',
              'original_corpus_unchanged':True,'splits':{}}
    for split,count in (('development',4),('sealed_holdout',8)):
        boards=[];certificates=[];seeds=[]
        for i in range(count):
            seed=int.from_bytes(hashlib.sha256(f'sum10-certified-high-digit-v1:{split}:{FAMILY}:{i}'.encode()).digest()[:8],'big')
            identifier=f'{split}_{FAMILY}_{i:02d}'
            board,cert=generate(seed,identifier)
            boards.append(board);certificates.append(cert)
            seeds.append({'id':identifier,'family':FAMILY,'seed':seed})
        for name,records in (('boards',boards),('certificates',certificates),('seeds',seeds)):
            write_json(output/split/(name+'.json'),records)
        manifest['splits'][split]={'count':count,'per_family':count,
                                  'boards_sha256':digest(boards),'certificates_sha256':digest(certificates),
                                  'seeds_sha256':digest(seeds)}
    write_json(output/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT)
    main(parser.parse_args().output)
