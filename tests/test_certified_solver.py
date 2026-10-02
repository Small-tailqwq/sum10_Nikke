"""Complete-move solver properties and opt-in legacy worker integration."""
import json, random, sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Head'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import certified_solver as candidate
from solver_support import load_core

def replay(board,out):
    cells=[r[:] for r in board];removed=0
    for r1,c1,r2,c2 in out['path']:
        assert 0<=r1<=r2<len(cells) and 0<=c1<=c2<len(cells[0])
        live=[(r,c) for r in range(r1,r2+1) for c in range(c1,c2+1) if cells[r][c]]
        assert sum(cells[r][c] for r,c in live)==10
        removed+=len(live)
        for r,c in live:cells[r][c]=0
    assert removed==out['score']
    if out['optimal']:assert out['score']==out['upper_bound']
    return cells

def brute(board):
    out=set()
    for r1 in range(len(board)):
        for r2 in range(r1,len(board)):
            for c1 in range(len(board[0])):
                for c2 in range(c1,len(board[0])):
                    live=tuple((r,c) for r in range(r1,r2+1) for c in range(c1,c2+1) if board[r][c])
                    if sum(board[r][c] for r,c in live)==10:out.add(live)
    return out

@pytest.mark.parametrize('seed',range(80))
def test_scanner_complete_unique(seed):
    rng=random.Random(seed)
    board=[[rng.choice([0,0,1,2,3,4,5,6,7,8,9]) for _ in range(6)] for _ in range(5)]
    moves=candidate.legal_moves(board)
    actual=[tuple((r,c) for r in range(m[0],m[2]+1) for c in range(m[1],m[3]+1) if board[r][c]) for m in moves]
    assert len(actual)==len(set(actual))
    assert set(actual)==brute(board)

@pytest.mark.parametrize('pair_first',[False,True])
@pytest.mark.parametrize('seed',range(10))
def test_tiny_replay_reproducibility(seed,pair_first):
    rng=random.Random(seed);board=[[rng.randint(0,9) for _ in range(5)] for _ in range(4)]
    original=json.dumps(board)
    a=candidate.solve(board,beam=32,seed=seed,time_limit=None,pair_first=pair_first)
    b=candidate.solve(board,beam=32,seed=seed,time_limit=None,pair_first=pair_first)
    replay(board,a);assert a['path']==b['path'];assert original==json.dumps(board)

def test_corner_free_cross_and_proof():
    board=[[0,1,0],[2,0,3],[0,4,0]]
    out=candidate.solve(board,time_limit=None)
    replay(board,out);assert out['score']==4 and out['optimal'] and out['full_clear']

def test_zero_budget_and_impossible_residue():
    assert candidate.solve([[1,9]],time_limit=0)['path']==[]
    out=candidate.solve([[1,9,1]],time_limit=None)
    replay([[1,9,1]],out);assert out['score']==2 and out['optimal'] and out['upper_bound']==2

def test_worker_opt_in(tmp_path):
    source=(Path(__file__).resolve().parents[1]/'Head/god_brain.py').read_text()
    mod=load_core(source,'_cert_worker_test',tmp_path)
    out=mod._solve_process_hydra(([1]*6,[1,9,2,8,5,5],2,3,20,'complete',17,30.,{'name':'test'}))
    replay([[1,9,2],[8,5,5]],out)
    assert out['score']==6 and out['optimal'] and out['worker_id']==17
    assert out['iterations']==out['attempts']-1

@pytest.mark.parametrize('mask,values', [([1],[-1]),([2],[5]),([1],[10]),([1,1],[1])])
def test_worker_rejects_invalid_input(mask,values,tmp_path):
    source=(Path(__file__).resolve().parents[1]/'Head/god_brain.py').read_text()
    mod=load_core(source,'_cert_bad_worker_test',tmp_path)
    with pytest.raises(ValueError):mod._solve_process_hydra((mask,values,1,1,20,'complete',17,1.,{}))


def test_extreme_finite_weights_rejected():
    with pytest.raises(ValueError):
        candidate.solve([[1,9,5,5]],weights=(-1e308,0,1e308,1,0))
    with pytest.raises(ValueError):
        candidate.solve([[1,9,5,5]],cut_penalty=1e308)
