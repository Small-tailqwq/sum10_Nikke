"""Headless JSON-in/JSON-out entry point for opt-in complete-move search.

Example: python benchmarks/solve_certified.py board.json --beam 200 --seconds 3
Input is a 2D integer board (zero=empty), or {"board": [[...], ...]}.
No OCR, WebSocket, screenshot, mouse or keyboard modules are imported.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Head'))
from certified_solver import solve

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('board',type=Path)
    p.add_argument('--beam',type=int,default=200)
    p.add_argument('--seconds',type=float,default=3.)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--target',type=int)
    args=p.parse_args()
    data=json.loads(args.board.read_text())
    if isinstance(data,dict): data=data['board']
    print(json.dumps(solve(data,beam=args.beam,seed=args.seed,time_limit=args.seconds,target_score=args.target)))

if __name__=='__main__':main()
