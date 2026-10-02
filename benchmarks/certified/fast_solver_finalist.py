"""Standalone complete-move, unique-state beam search for sum-10 rectangles.

A full-clear path certifies the cell-count upper bound. A partial path reaching
the independently computed residue bound also proves optimal; all other answers
are only best-found heuristic results. Seven necessary multiset inequalities
penalize value distributions that cannot partition into sum-10 groups. They
never remove legal moves or certify a geometric solution. The time limit is cooperative:
conversion, JIT compilation and result reconstruction are included in elapsed
seconds; a compiled expansion cannot be interrupted before a layer ends.

No game, input-control, OCR, corpus, or witness modules are imported.
"""
from __future__ import annotations

import math
import time
import numpy as np
from numba import njit

_U = np.uint64


@njit(cache=True, inline='always')
def _mix(x):
    x ^= x >> _U(30)
    x *= _U(0xbf58476d1ce4e5b9)
    x ^= x >> _U(27)
    x *= _U(0x94d049bb133111eb)
    return x ^ (x >> _U(31))


@njit(cache=True, inline='always')
def _hash(a, b, c):
    return _mix(a ^ _mix(b + _U(0x9e3779b97f4a7c15)) ^ _mix(c + _U(0xd1b54a32d192ed03)))


@njit(cache=True, inline='always')
def _pop(x):
    x -= (x >> _U(1)) & _U(0x5555555555555555)
    x = (x & _U(0x3333333333333333)) + ((x >> _U(2)) & _U(0x3333333333333333))
    x = (x + (x >> _U(4))) & _U(0x0f0f0f0f0f0f0f0f)
    return int((x * _U(0x0101010101010101)) >> _U(56))


@njit(cache=True, inline='always')
def _shiftleft(a,b,c,n):
    return a << _U(n), (b << _U(n)) | (a >> _U(64-n)), (c << _U(n)) | (b >> _U(64-n))


@njit(cache=True, inline='always')
def _shiftright(a,b,c,n):
    return (a >> _U(n)) | (b << _U(64-n)), (b >> _U(n)) | (c << _U(64-n)), c >> _U(n)


@njit(cache=True, inline='always')
def _isolated(a,b,c,cols,leftedge,rightedge):
    # Shift sources after excluding their wrapping row edges.
    l0,l1,l2 = _shiftleft(a & ~rightedge[0], b & ~rightedge[1], c & ~rightedge[2],1)
    r0,r1,r2 = _shiftright(a & ~leftedge[0], b & ~leftedge[1], c & ~leftedge[2],1)
    d0,d1,d2 = _shiftleft(a,b,c,cols)
    u0,u1,u2 = _shiftright(a,b,c,cols)
    return _pop(a & ~(l0|r0|d0|u0)) + _pop(b & ~(l1|r1|d1|u1)) + _pop(c & ~(l2|r2|d2|u2))


@njit(cache=True, inline="always")
def _violation(a,b,c,masks):
    c1=_pop(a&masks[1,0])+_pop(b&masks[1,1])+_pop(c&masks[1,2])
    c2=_pop(a&masks[2,0])+_pop(b&masks[2,1])+_pop(c&masks[2,2])
    c3=_pop(a&masks[3,0])+_pop(b&masks[3,1])+_pop(c&masks[3,2])
    c4=_pop(a&masks[4,0])+_pop(b&masks[4,1])+_pop(c&masks[4,2])
    c5=_pop(a&masks[5,0])+_pop(b&masks[5,1])+_pop(c&masks[5,2])
    c6=_pop(a&masks[6,0])+_pop(b&masks[6,1])+_pop(c&masks[6,2])
    c7=_pop(a&masks[7,0])+_pop(b&masks[7,1])+_pop(c&masks[7,2])
    c8=_pop(a&masks[8,0])+_pop(b&masks[8,1])+_pop(c&masks[8,2])
    c9=_pop(a&masks[9,0])+_pop(b&masks[9,1])+_pop(c&masks[9,2])
    violation=0
    violation+=max(0,(-3)*c1+(-6)*c2+(1)*c3+(-2)*c4+(2)*c6+(-1)*c7+(6)*c8+(3)*c9)
    violation+=max(0,(-3)*c1+(-1)*c2+(1)*c3+(-2)*c4+(2)*c6+(-1)*c7+(1)*c8+(3)*c9)
    violation+=max(0,(-2)*c1+(-4)*c2+(-1)*c3+(2)*c4+(-2)*c6+(1)*c7+(4)*c8+(2)*c9)
    violation+=max(0,(-1)*c1+(-2)*c2+(-3)*c3+(-4)*c4+(4)*c6+(3)*c7+(2)*c8+(1)*c9)
    violation+=max(0,(-1)*c1+(-2)*c2+(-3)*c3+(1)*c4+(-1)*c6+(3)*c7+(2)*c8+(1)*c9)
    violation+=max(0,(-1)*c1+(-1)*c3+(1)*c7+(1)*c9)
    violation+=max(0,(-1)*c1+(1)*c9)
    return violation

@njit(cache=True, inline='always')
def _swap(i,j,states,hs,parents,rects,counts,centers,slots,table):
    for k in range(3):
        states[i,k],states[j,k] = states[j,k],states[i,k]
    for k in range(4):
        rects[i,k],rects[j,k] = rects[j,k],rects[i,k]
    hs[i],hs[j] = hs[j],hs[i]
    parents[i],parents[j] = parents[j],parents[i]
    counts[i],counts[j] = counts[j],counts[i]
    centers[i],centers[j] = centers[j],centers[i]
    slots[i],slots[j] = slots[j],slots[i]
    table[slots[i]],table[slots[j]] = i,j


@njit(cache=True)
def _expand(vals,rows,cols,current,current_counts,current_centers,beam,
            seed,weights,center_weights,leftedge,rightedge,parent_offset,classic=False,cut_penalty=1000.0):
    """Generate the exact legal-successor set, retaining a unique top beam.

    The table holds full state masks; hashes only select buckets. Deleted heap
    entries use tombstones, and rebuilding bounds lookup cost. Because rank is
    a deterministic function of state, an evicted duplicate can never improve
    the monotonic minimum accepted rank.
    """
    digit_masks=np.zeros((10,3),np.uint64)
    for i in range(len(vals)):
        if vals[i]:
            digit_masks[vals[i],i//64] |= _U(1)<<_U(i%64)
    states = np.empty((beam,3),np.uint64)
    hs = np.empty(beam,np.float64)
    parents = np.empty(beam,np.int64)
    rects = np.empty((beam,4),np.int16)
    counts = np.empty(beam,np.int16)
    centers = np.empty(beam,np.int32)
    slots = np.empty(beam,np.int64)
    tsize=16
    while tsize < beam*16:
        tsize*=2
    table=np.full(tsize,-1,np.int64)
    tablemask=tsize-1
    used=0
    size=0
    expanded=0
    generated=0
    duplicates=0
    bestcount=32767
    bestparent=-1
    bestrect=np.zeros(4,np.int16)
    beststate=np.zeros(3,np.uint64)
    board=np.empty((rows,cols),np.int16)
    rowlive=np.empty(rows,np.int16)
    colvals=np.empty(cols,np.int16)
    colmasks=np.empty((cols,3),np.uint64)
    colcounts=np.empty(cols,np.int16)
    colcenters=np.empty(cols,np.int32)
    for p in range(len(current)):
        expanded+=1
        a,b,c=current[p,0],current[p,1],current[p,2]
        for r in range(rows):
            rowlive[r]=0
            for col in range(cols):
                i=r*cols+col
                live=(current[p,i//64] >> _U(i%64)) & _U(1)
                board[r,col]=vals[i] if live else 0
                rowlive[r]+=int(live)
        for r1 in range(rows):
            if rowlive[r1]==0:
                continue
            colvals[:]=0
            colmasks[:,:]=0
            colcounts[:]=0
            colcenters[:]=0
            for r2 in range(r1,rows):
                for col in range(cols):
                    value=board[r2,col]
                    if value:
                        i=r2*cols+col
                        colvals[col]+=value
                        colmasks[col,i//64] |= _U(1) << _U(i%64)
                        colcounts[col]+=1
                        colcenters[col]+=center_weights[i]
                if rowlive[r2]==0:
                    continue
                end=0
                total=0
                m0,m1,m2=_U(0),_U(0),_U(0)
                removed=0
                removedcenter=0
                for start in range(cols):
                    if colvals[start]==0:
                        continue
                    if end < start:
                        end=start
                    while end<cols and total<10:
                        total+=colvals[end]
                        m0|=colmasks[end,0]
                        m1|=colmasks[end,1]
                        m2|=colmasks[end,2]
                        removed+=colcounts[end]
                        removedcenter+=colcenters[end]
                        end+=1
                    if total==10:
                        # end-1 has a nonzero column. Requiring nonempty top
                        # and bottom boundaries gives the unique minimal box.
                        top=False
                        bottom=False
                        for col in range(start,end):
                            top=top or board[r1,col]>0
                            bottom=bottom or board[r2,col]>0
                        if top and bottom and (not classic or removed==2):
                            ca,cb,cc=a ^ m0,b ^ m1,c ^ m2
                            count=int(current_counts[p])-removed
                            center=int(current_centers[p])-removedcenter
                            generated+=1
                            if count<bestcount:
                                bestcount=count
                                bestparent=parent_offset+p
                                bestrect[0],bestrect[1],bestrect[2],bestrect[3]=r1,start,r2,end-1
                                beststate[0],beststate[1],beststate[2]=ca,cb,cc
                            if count==0:
                                return (states[:size],counts[:size],centers[:size],parents[:size],rects[:size],
                                        bestcount,bestparent,bestrect,beststate,expanded,generated,duplicates,True)
                            key=_hash(ca,cb,cc)
                            noise=float(_mix(key ^ seed) >> _U(11))*(1.0/9007199254740992.0)
                            islands=_isolated(ca,cb,cc,cols,leftedge,rightedge)
                            panic=weights[3] if count<30 else 1.0
                            h=-count*weights[0]-islands*weights[1]*panic-center*weights[2]+noise*weights[4]-cut_penalty*_violation(ca,cb,cc,digit_masks)
                            if size<beam or h>hs[0]:
                                if used*2>=tsize:
                                    table[:]=-1
                                    for q in range(size):
                                        bucket=np.int64(_hash(states[q,0],states[q,1],states[q,2]) & _U(tablemask))
                                        while table[bucket]>=0:
                                            bucket=(bucket+1)&tablemask
                                        table[bucket]=q
                                        slots[q]=bucket
                                    used=size
                                bucket=np.int64(key & _U(tablemask))
                                firstdeleted=-1
                                duplicate=False
                                while table[bucket]!=-1:
                                    q=table[bucket]
                                    if q<0:
                                        if firstdeleted<0:
                                            firstdeleted=bucket
                                    elif states[q,0]==ca and states[q,1]==cb and states[q,2]==cc:
                                        duplicate=True
                                        duplicates+=1
                                        break
                                    bucket=(bucket+1)&tablemask
                                if not duplicate:
                                    if firstdeleted>=0:
                                        bucket=firstdeleted
                                    else:
                                        used+=1
                                    if size<beam:
                                        q=size
                                        size+=1
                                    else:
                                        q=0
                                        table[slots[0]]=-2
                                    states[q,0],states[q,1],states[q,2]=ca,cb,cc
                                    hs[q]=h
                                    counts[q]=count
                                    centers[q]=center
                                    parents[q]=parent_offset+p
                                    rects[q,0],rects[q,1],rects[q,2],rects[q,3]=r1,start,r2,end-1
                                    slots[q]=bucket
                                    table[bucket]=q
                                    if q>0:
                                        while q>0:
                                            up=(q-1)//2
                                            if hs[up]<=hs[q]:
                                                break
                                            _swap(q,up,states,hs,parents,rects,counts,centers,slots,table)
                                            q=up
                                    else:
                                        while q*2+1<size:
                                            child=q*2+1
                                            if child+1<size and hs[child+1]<hs[child]:
                                                child+=1
                                            if hs[q]<=hs[child]:
                                                break
                                            _swap(q,child,states,hs,parents,rects,counts,centers,slots,table)
                                            q=child
                    total-=colvals[start]
                    m0 ^= colmasks[start,0]
                    m1 ^= colmasks[start,1]
                    m2 ^= colmasks[start,2]
                    removed-=colcounts[start]
                    removedcenter-=colcenters[start]
    return (states[:size],counts[:size],centers[:size],parents[:size],rects[:size],
            bestcount,bestparent,bestrect,beststate,expanded,generated,duplicates,False)


def _input(board):
    arr=np.asarray(board)
    if arr.ndim!=2 or not arr.size or arr.size>192 or arr.shape[1]>=64:
        raise ValueError('board must be nonempty rectangular 2D, at most 192 cells and 63 columns')
    if not np.issubdtype(arr.dtype,np.integer) or np.any(arr<0) or np.any(arr>9):
        raise ValueError('board cells must be integers 0..9 (0 means already empty)')
    rows,cols=arr.shape
    vals=np.ascontiguousarray(arr.ravel(),dtype=np.int16)
    masks=np.zeros((1,3),np.uint64)
    edges=np.zeros((2,3),np.uint64)
    center=np.empty(vals.size,np.int32)
    center_total=0
    for i,value in enumerate(vals):
        r,c=divmod(i,cols)
        center[i]=2*(rows+cols)-abs(2*r-(rows-1))-abs(2*c-(cols-1))
        if value:
            masks[0,i//64] |= _U(1)<<_U(i%64)
            center_total+=int(center[i])
        if c==0:
            edges[0,i//64] |= _U(1)<<_U(i%64)
        if c==cols-1:
            edges[1,i//64] |= _U(1)<<_U(i%64)
    return vals,rows,cols,masks,edges,center,center_total


def _solve_once(board:list[list[int]],beam:int=256,seed:int=0,time_limit:float=10.0,*,weights=None,target_score=None,pair_first=False,cut_penalty:float=1000.0)->dict:
    """Find a legal best-found path, stopping immediately at a full clear.

    ``beam`` bounds unique states per depth. ``seed`` deterministically affects
    state ranking, never legality or mask equality. ``time_limit=None`` disables
    deadline checks. Timed runs can stop at different depths across machines.
    ``optimal`` is true only when the path reaches the independent residue bound.
    """
    started=time.perf_counter()
    if isinstance(beam,bool) or int(beam)!=beam or beam<1:
        raise ValueError('beam must be a positive integer')
    beam=int(beam)
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit<0):
        raise ValueError('time_limit must be nonnegative finite seconds or None')
    if not math.isfinite(cut_penalty) or cut_penalty<0:
        raise ValueError('cut_penalty must be finite and nonnegative')
    deadline=math.inf if time_limit is None else started+time_limit
    vals,rows,cols,current,edges,center,center_total=_input(board)
    initial=int(np.count_nonzero(vals))
    total=int(vals.sum())
    dp=[initial+1]*10
    dp[0]=0
    for value in vals:
        if value:
            old=dp[:]
            for residue in range(10):
                target=(residue+int(value))%10
                dp[target]=min(dp[target],old[residue]+1)
    min_remaining=dp[total%10]
    upper_bound=initial-min_remaining
    if target_score is not None:
        if isinstance(target_score,bool) or int(target_score)!=target_score or not 0<=target_score<=initial:
            raise ValueError('target_score must be an integer from zero to initial live count')
        target_score=int(target_score)
    current_counts=np.array([initial],np.int16)
    current_centers=np.array([center_total],np.int32)
    # At a fixed depth every state has removed the same total value. A negative
    # count weight preserves low-valued cells needed to pair with high digits.
    if weights is None:
        weights=(-32.0,5.0,0.025,4.0,14.0)
    weights=np.asarray(weights,dtype=np.float64)
    if weights.shape!=(5,) or not np.isfinite(weights).all():
        raise ValueError('weights must contain five finite values')
    parents=[np.array([-1],np.int64)]
    moves=[np.zeros((1,4),np.int16)]
    offsets=[0]
    current_offset=0
    node_count=1
    best_count=initial
    best_parent=-1
    best_rect=None
    best_mask=current[0].copy()
    stats=dict(expanded=0,generated=0,duplicate_hits=0,layers=0,retained=1)
    status='full_clear' if initial==0 else 'search_exhausted'
    phase=bool(pair_first)
    for depth in range(total//10+1):
        if best_count<=min_remaining:
            status='full_clear' if best_count==0 else 'upper_bound_reached'
            break
        if target_score is not None and initial-best_count>=target_score:
            status='target_reached'
            break
        if time.perf_counter()>=deadline:
            status='time_limit'
            break
        out=_expand(vals,rows,cols,current,current_counts,current_centers,beam,
                    _U(int(seed) & ((1<<64)-1)),weights,center,edges[0],edges[1],current_offset,phase,float(cut_penalty))
        (following,following_counts,following_centers,p,m,count,bp,rect,state,
         expanded,generated,duplicates,cleared)=out
        stats['expanded']+=int(expanded)
        stats['generated']+=int(generated)
        stats['duplicate_hits']+=int(duplicates)
        stats['layers']+=1
        if count<best_count:
            best_count=int(count)
            best_parent=int(bp)
            best_rect=rect.copy()
            best_mask=state.copy()
        if cleared:
            status='full_clear'
            break
        if not len(following):
            if phase:
                phase=False
                continue
            status='search_exhausted'
            break
        offsets.append(node_count)
        parents.append(p.copy())
        moves.append(m.copy())
        current_offset=node_count
        node_count+=len(following)
        stats['retained']+=len(following)
        current,current_counts,current_centers=following,following_counts,following_centers
    if best_count<=min_remaining:
        status='full_clear' if best_count==0 else 'upper_bound_reached'
    elif target_score is not None and initial-best_count>=target_score:
        status='target_reached'
    path=[]
    if best_rect is not None:
        path.append(best_rect.tolist())
        pointer=best_parent
        import bisect
        while pointer>0:
            layer=bisect.bisect_right(offsets,pointer)-1
            local=pointer-offsets[layer]
            path.append(moves[layer][local].tolist())
            pointer=int(parents[layer][local])
        path.reverse()
    elapsed=time.perf_counter()-started
    return dict(path=path,score=initial-best_count,optimal=best_count<=min_remaining,upper_bound=upper_bound,
                full_clear=best_count==0,status=status,remaining=best_count,
                initial_live=initial,total_value=total,elapsed_seconds=elapsed,
                time_limit_seconds=time_limit,budget_exceeded=elapsed>deadline-started,
                deadline_kind='cooperative_per_layer_including_conversion_and_jit',
                seed=int(seed),beam=beam,stats=stats,
                remaining_mask=[int(x) for x in best_mask],
                solver='complete_unique_multiset_beam_v3',cut_penalty=float(cut_penalty),pair_first=bool(pair_first))


def enumerate_moves(board):
    """Independent-use legal successor enumeration, with no beam truncation.

    Intended for scanner differential tests. Returned boxes are minimally
    trimmed; all syntactically different boxes with the same removal coincide.
    """
    vals,rows,cols,state,edges,center,center_total=_input(board)
    bound=rows*(rows+1)//2*cols
    out=_expand(vals,rows,cols,state,np.array([np.count_nonzero(vals)],np.int16),
                np.array([center_total],np.int32),bound,_U(0),np.zeros(5),
                center,edges[0],edges[1],0)
    if out[-1]:
        # Full-clear short circuit is correct for solve, but this diagnostic
        # helper must still include other legal moves. Exhaustively enumerate.
        a=np.asarray(board)
        found={}
        for r1 in range(rows):
            for r2 in range(r1,rows):
                for c1 in range(cols):
                    for c2 in range(c1,cols):
                        if int(a[r1:r2+1,c1:c2+1].sum())==10:
                            live=tuple((r,c) for r in range(r1,r2+1) for c in range(c1,c2+1) if a[r,c])
                            found[live]=(min(r for r,c in live),min(c for r,c in live),max(r for r,c in live),max(c for r,c in live))
        return [list(x) for x in found.values()]
    return out[4].tolist()


legal_moves = enumerate_moves


def solve(board:list[list[int]],beam:int=256,seed:int=0,time_limit:float=10.0,*,weights=None,target_score=None,pair_first=False,cut_penalty:float=1000.0)->dict:
    """Anytime complete-move beam search with multiset-aware deterministic repair.

    A budgeted search spends spare time on seeded restarts and suffix repair.
    With time_limit=None, run exactly one deterministic unbounded base search.
    The budget is cooperative per compiled layer and includes conversion/JIT.

    Default ``pair_first=False`` uses all legal moves from the first layer.
    Setting it True first runs pair-only search and continues the entire deepest
    surviving beam with unrestricted moves. No game mode is assumed.

    ``weights`` is (remaining_count, isolated_cells, centrality, endgame_island_
    multiplier, deterministic_noise_amplitude). The rank subtracts the first
    three terms, so the default remaining_count=-32 preserves low digits at
    fixed depth. The island multiplier applies below 30 remaining cells.

    ``cut_penalty`` (default1000, finite nonnegative) multiplies the sum of
    positive violations of seven necessary digit-multiset inequalities. This
    is a soft ranking penalty even on impossible-to-clear boards. Passing zero
    disables it. Passing explicit weights applies them to every restart/repair;
    otherwise the count weight stays -32 and the other terms diversify.

    A deterministic seed fixes ordering in an unbounded base run. Timed anytime
    runs may complete different numbers of attempts on different machines.
    """
    started=time.perf_counter()
    first=_solve_once(board,beam,seed,time_limit,weights=weights,target_score=target_score,pair_first=pair_first,cut_penalty=cut_penalty)
    stats={key:int(value) for key,value in first['stats'].items()}
    best=first
    attempts=1
    if time_limit is None or first['optimal'] or (target_score is not None and first['score']>=target_score):
        first['attempts']=attempts
        first['elapsed_seconds']=time.perf_counter()-started
        first['budget_exceeded']=time_limit is not None and first['elapsed_seconds']>time_limit
        return first
    deadline=started+float(time_limit)
    source=np.asarray(board,dtype=np.int16)
    rng=np.random.default_rng(int(seed)&((1<<64)-1))
    original_live=int(first['initial_live'])
    initial_total=int(first['total_value'])
    bound=int(first['upper_bound'])
    while time.perf_counter()<deadline:
        attempts+=1
        restart=attempts%5==0 or len(best['path'])<12
        if restart:
            prefix=[]
            partial=source.copy()
        else:
            # Later repairs are wider, while regular root restarts retain
            # access to branches excluded by the current incumbent prefix.
            span=int(rng.integers(10,36))
            if attempts%3==0:
                span=int(rng.integers(25,max(26,len(best['path']))))
            keep=max(0,len(best['path'])-span)
            prefix=best['path'][:keep]
            partial=source.copy()
            for r1,c1,r2,c2 in prefix:
                partial[r1:r2+1,c1:c2+1]=0
        removed=original_live-int(np.count_nonzero(partial))
        childseed=int(rng.integers(0,2**63))
        noise=float(rng.choice(np.array([14.,32.,70.,130.])))
        island=float(rng.choice(np.array([0.,2.,5.,12.,25.])))
        center=float(rng.choice(np.array([-.06,-.02,0.,.025,.06])))
        # Restarts begin with pairs. Repairs alternate score-driven and
        # value-preserving ranks, adding broad state-level diversification.
        countweight=-32.
        trialweights=(countweight,island,center,4.,noise) if weights is None else weights
        result=_solve_once(partial,beam,childseed,max(0.,deadline-time.perf_counter()),
                           weights=trialweights,target_score=None if target_score is None else max(0,target_score-removed),
                           pair_first=restart and pair_first,cut_penalty=cut_penalty)
        for key,value in result['stats'].items():
            stats[key]+=int(value)
        result['score']+=removed
        result['path']=prefix+result['path']
        better=(result['score'],len(result['path']))>(best['score'],len(best['path']))
        if not better and result['score']==best['score'] and len(result['path'])==len(best['path']):
            better=bool(rng.random()<.2)
        if better:
            best=result
        if best['score']>=bound or (target_score is not None and best['score']>=target_score):
            break
    best['initial_live']=original_live
    best['total_value']=initial_total
    best['upper_bound']=bound
    best['optimal']=best['score']>=bound
    best['full_clear']=best['remaining']==0
    best['seed']=int(seed)
    best['pair_first']=bool(pair_first)
    best['attempts']=attempts
    best['stats']=stats
    best['elapsed_seconds']=time.perf_counter()-started
    best['time_limit_seconds']=time_limit
    best['budget_exceeded']=best['elapsed_seconds']>time_limit
    if best['full_clear']:
        best['status']='full_clear'
    elif best['optimal']:
        best['status']='upper_bound_reached'
    elif target_score is not None and best['score']>=target_score:
        best['status']='target_reached'
    else:
        best['status']='time_limit'
    return best
