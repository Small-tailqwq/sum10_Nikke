#!/usr/bin/env python3
"""Freeze candidate/config hashes, then release only board JSON for evaluation.

Operational embargo, not an access-control or anti-cheating boundary. Run exactly
once per predeclared evaluation. Re-tuning after inspecting results requires a
new never-seen holdout, not a fresh destination for these same boards.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent

def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def sha_json(obj):return sha_bytes(json.dumps(obj,sort_keys=True,separators=(',',':')).encode())

def release(candidate_files,configuration_path,destination,corpus_root=ROOT,include_extension=True):
    if destination.exists(): raise ValueError('Destination exists; refuse to overwrite or repeat an evaluation release')
    config=json.loads(configuration_path.read_text())
    candidates=[]
    for path in candidate_files:
        data=path.read_bytes()
        candidates.append({'path':str(path.resolve()),'sha256':sha_bytes(data),'bytes':len(data)})
    if not candidates:raise ValueError('At least one candidate file is required')
    sources=[('primary',corpus_root)]
    extension=corpus_root/'high_digit_extension'
    if include_extension and extension.exists():sources.append(('high_digit_extension',extension))
    committed=[]
    for label,root in sources:
        manifest=json.loads((root/'manifest.json').read_text())
        source=root/'sealed_holdout'/'boards.json'
        data=source.read_bytes();boards=json.loads(data)
        expected=manifest['splits']['sealed_holdout']['boards_sha256']
        if sha_json(boards)!=expected:raise ValueError(f'{label}: holdout commitment mismatch')
        committed.append((label,source,{'name':label,'count':len(boards),
                         'boards_sha256':expected,'file_sha256':sha_bytes(data),
                         'manifest_file_sha256':sha_bytes((root/'manifest.json').read_bytes()),
                         'evaluation_filename':label+'_holdout_boards.json'}))
    freeze={'schema_version':1,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'candidate_files':candidates,'configuration':config,
            'configuration_file_sha256':sha_bytes(configuration_path.read_bytes()),
            'corpus_holdout_commitments':[x[2] for x in committed],
            'protocol':'Freeze candidate and all settings before first holdout read. No certificate or seed release.'}
    destination.mkdir(parents=True,exist_ok=False)
    # The freeze record is persisted before any holdout boards are exported.
    (destination/'candidate_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    for label,source,info in committed:
        target=destination/info['evaluation_filename']
        shutil.copyfile(source,target)
        if sha_bytes(target.read_bytes())!=info['file_sha256']:raise RuntimeError('Copy verification failed')
    return freeze

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,action='append',required=True,help='Repeat for each candidate implementation/dependency file')
    parser.add_argument('--configuration',type=Path,required=True,help='JSON containing all frozen search budgets, modes, seeds and settings')
    parser.add_argument('--destination',type=Path,required=True,help='New evaluation output directory; must not already exist')
    parser.add_argument('--corpus-root',type=Path,default=ROOT)
    parser.add_argument('--exclude-extension',action='store_true')
    args=parser.parse_args()
    freeze=release(args.candidate,args.configuration,args.destination,args.corpus_root,not args.exclude_extension)
    print(json.dumps({'destination':str(args.destination),'frozen_at_utc':freeze['frozen_at_utc'],
                      'candidate_files':len(freeze['candidate_files']),
                      'holdout_boards':sum(x['count'] for x in freeze['corpus_holdout_commitments'])},indent=2))

if __name__=='__main__':main()
