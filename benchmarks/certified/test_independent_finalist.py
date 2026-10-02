"""Independent differential scanner and certificate checks, without witnesses."""
import importlib.util,json,random,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fast_solver_finalist as candidate
from benchmark_certified import verify,upper_bound

def brute_moves(board):
    rows=len(board); cols=len(board[0]); found=set()
    for r1 in range(rows):
        for r2 in range(r1,rows):
            for c1 in range(cols):
                for c2 in range(c1,cols):
                    cells=tuple((r,c) for r in range(r1,r2+1) for c in range(c1,c2+1) if board[r][c]>0)
                    if sum(board[r][c] for r,c in cells)==10:
                        found.add(cells)
    return found

def key(board,rect):
    r1,c1,r2,c2=rect
    return tuple((r,c) for r in range(r1,r2+1) for c in range(c1,c2+1) if board[r][c]>0)

@pytest.mark.parametrize('seed',range(300))
def test_complete_legal_successors(seed):
    rng=random.Random(seed)
    rows=rng.randint(1,6);cols=rng.randint(1,7)
    board=[[rng.choice([0,0,0,1,2,3,4,5,6,7,8,9]) for _ in range(cols)] for _ in range(rows)]
    rects=candidate.legal_moves(board)
    actual=[key(board,rect) for rect in rects]
    assert len(actual)==len(set(actual))
    assert set(actual)==brute_moves(board)

def test_cross_corner_completeness():
    board=[[0,1,0],[2,0,3],[0,4,0]]
    assert ((0,1),(1,0),(1,2),(2,1)) in {key(board,m) for m in candidate.legal_moves(board)}

@pytest.mark.parametrize('seed',range(40))
def test_returned_path_and_input_integrity(seed):
    rng=random.Random(seed)
    board=[[rng.randint(0,9) for _ in range(7)] for _ in range(6)]
    frozen=json.dumps(board)
    result=candidate.solve(board,beam=30,seed=seed,time_limit=None)
    verify(board,result)
    assert json.dumps(board)==frozen
    assert candidate.solve(board,beam=30,seed=seed,time_limit=None)['path']==result['path']

@pytest.mark.parametrize('seed',range(20))
def test_tiny_matches_exhaustive_optimum(seed):
    rng=random.Random(seed)
    board=[[rng.randint(0,9) for _ in range(3)] for _ in range(3)]
    flat=tuple(v for row in board for v in row)
    from functools import lru_cache
    @lru_cache(None)
    def dfs(state):
        b=[list(state[i:i+3]) for i in range(0,9,3)];best=0
        for selected in brute_moves(b):
            nxt=list(state)
            for r,c in selected:nxt[r*3+c]=0
            best=max(best,len(selected)+dfs(tuple(nxt)))
        return best
    expected=dfs(flat)
    out=candidate.solve(board,beam=512,seed=seed,time_limit=None)
    verify(board,out)
    assert out['score']==expected

@pytest.mark.parametrize('board',[[[0]],[[1]],[[1,9]],[[0,5,0,5]],[[1]*64],[[1]*10]*20,[[10]],[[1.2]],[[True]],[],[[1],[2,3]]])
def test_input_edge_cases(board):
    valid=bool(board) and isinstance(board[0],list) and bool(board[0]) and len(board)*len(board[0])<=192 and len(board[0])<=63 and len({len(r) for r in board})==1 and all(type(v)==int and 0<=v<=9 for row in board for v in row)
    if valid:verify(board,candidate.solve(board,beam=2,time_limit=None))
    else:
        with pytest.raises((ValueError,TypeError)):candidate.solve(board)
