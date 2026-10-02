"""Headless eye -> brain -> hand contracts; no device/server imports or I/O.

Only selected definitions are compiled from god_brain.py/predict.py. OCR image
reads, capture, web sockets, process executors, sleeps, and mouse APIs are fakes.
The optional Node checks execute the actual UI script in a network-free VM.
"""
import ast
import asyncio
import builtins
import copy
from datetime import datetime
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / 'Head/god_brain.py'
HTML = ROOT / 'Head/Head_web.html'


def extract_definitions(names, namespace=None, source_path=BRAIN):
    """Compile selected real definitions, never module initialization."""
    namespace = {} if namespace is None else namespace
    selected = []
    for node in ast.parse(source_path.read_text(encoding='utf-8')).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            selected.append(node)
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), 'exec'), namespace)
    return namespace


def extracted_worker(package, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.syspath_prepend(str(ROOT / 'Head'))
    namespace = {'np': np, '__name__': 'Head._contract_worker' if package else '_contract_worker',
                 '__package__': 'Head' if package else None}
    return extract_definitions({'_solve_process_hydra'}, namespace)['_solve_process_hydra']


def replay(board, result):
    remaining = copy.deepcopy(board)
    removed = 0
    for rect in result['path']:
        assert len(rect) == 4 and all(type(x) is int for x in rect)
        r1, c1, r2, c2 = rect
        assert 0 <= r1 <= r2 < len(board)
        assert 0 <= c1 <= c2 < len(board[0])
        live = [(r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1) if remaining[r][c]]
        assert sum(remaining[r][c] for r, c in live) == 10
        removed += len(live)
        for r, c in live:
            remaining[r][c] = 0
    assert removed == result['score']
    assert sum(bool(x) for row in remaining for x in row) == result['remaining']
    return remaining


@pytest.mark.parametrize('package', [False, True], ids=['script-import', 'package-import'])
def test_complete_worker_import_routes_mask_orientation_and_metadata(package, monkeypatch):
    worker = extracted_worker(package, monkeypatch)
    # Distinct row/column extents; stale masked-out values must be ignored.
    values = [99, 1, 9, -7, 4, 6, 55, 0, 3, 7, 88, 0]
    mask = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    before = (values[:], mask[:])
    personality = {'name': 'contract', 'role': 'fixture'}
    out = worker((mask, values, 3, 4, 16, 'complete', 17, None, personality))
    board = [[0, 1, 9, 0], [4, 6, 0, 0], [3, 7, 0, 0]]
    assert not any(x for row in replay(board, out) for x in row)
    assert out['score'] == 6 and out['full_clear'] and out['optimal']
    assert out['worker_id'] == 17 and out['personality'] == personality
    assert out['iterations'] == max(0, out['attempts'] - 1)
    assert (values, mask) == before
    # WebSocket serialization must not leak ndarray/NumPy integer values.
    assert json.loads(json.dumps(out))['path'] == out['path']


@pytest.mark.parametrize('rows,cols', [(16, 10), (10, 16), (12, 16)])
def test_complete_worker_preserves_supported_eye_board_dimensions(rows, cols, monkeypatch):
    worker = extracted_worker(True, monkeypatch)
    values = [0] * (rows * cols)
    values[-cols + 2:-cols + 4] = [1, 9]
    mask = [int(v != 0) for v in values]
    out = worker((mask, values, rows, cols, 8, 'complete', 3, None, {}))
    assert out['path'] == [[rows - 1, 2, rows - 1, 3]]
    assert out['score'] == 2 and out['full_clear']


@pytest.mark.parametrize('values', [[1, 0, 9, 0, 0, 0], [0] * 6])
def test_ocr_zero_cells_are_empty_even_when_legacy_ui_mask_marks_them_live(values, monkeypatch):
    worker = extracted_worker(True, monkeypatch)
    out = worker(([1] * 6, values, 2, 3, 8, 'complete', 5, None, {}))
    board = [values[:3], values[3:]]
    assert not any(x for row in replay(board, out) for x in row)
    assert out['initial_live'] == sum(bool(x) for x in values)
    assert out['full_clear']


@pytest.mark.parametrize('mode', ['classic', 'omni', 'god'])
def test_legacy_modes_never_import_new_optional_solver(mode):
    calls = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert 'certified_solver' not in name
        return original_import(name, *args, **kwargs)

    def base_search(mask, values, rows, cols, beam, search_mode, score, path, weights):
        calls.append(search_mode)
        return {'map': mask, 'path': [], 'score': 0}

    namespace = {'np': np, 'random': random, 'time': SimpleNamespace(time=lambda: 1.),
                 '_seed_search_rng': lambda seed: None, '_run_core_search_logic': base_search,
                 '__builtins__': dict(vars(builtins), __import__=guarded_import)}
    worker = extract_definitions({'_solve_process_hydra'}, namespace)['_solve_process_hydra']
    result = worker(([1, 1], [1, 9], 1, 2, 8, mode, 17, 0., {}))
    assert calls == (['classic', 'omni'] if mode == 'god' else [mode])
    assert set(result) == {'worker_id', 'score', 'path', 'iterations', 'personality'}


def test_ocr_recognizer_emits_native_int_row_major_16_by_10_without_device_imports():
    path = ROOT / 'eyes/Sum10_Labeling_Tool/predict.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in {'ROWS', 'COLS', 'CROP_RATIO'}}
    assert (constants['ROWS'], constants['COLS']) == (16, 10)
    expected = np.fromfunction(lambda r, c: (7 * r + 3 * c) % 10, (16, 10), dtype=int)
    image = np.repeat(np.repeat(expected, 10, axis=0), 10, axis=1)
    recognizer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Sum10Recognizer')
    method = next(n for n in recognizer.body if isinstance(n, ast.FunctionDef) and n.name == 'recognize_board')
    namespace = dict(constants, cv2=SimpleNamespace(imread=lambda path: image,
                         cvtColor=lambda img, mode: img, COLOR_BGR2GRAY=0),
                     Image=SimpleNamespace(fromarray=lambda img: img), print=lambda *args: None)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    fake = SimpleNamespace(predict_image=lambda img: int(img[0, 0]))
    matrix = namespace['recognize_board'](fake, 'synthetic.png')
    assert matrix == expected.tolist()
    assert all(type(cell) is int and 0 <= cell <= 9 for row in matrix for cell in row)


@pytest.mark.parametrize('method', ['DIRECT_INPUT', 'WIN32_API'])
def test_hand_inclusive_row_column_rectangles_map_to_calibrated_pixel_centers(method, monkeypatch):
    events = []
    mouse = SimpleNamespace(moveTo=lambda *a, **k: events.append(('move', a, k)),
                            mouseDown=lambda *a, **k: events.append(('down', a, k)),
                            mouseUp=lambda *a, **k: events.append(('up', a, k)))
    monkeypatch.setitem(sys.modules, 'pyautogui', mouse)
    namespace = {'INPUT_METHOD': method, 'pydirectinput': mouse,
                 'time': SimpleNamespace(sleep=lambda seconds: None)}
    hand = extract_definitions({'GodHand'}, namespace)['GodHand']()
    # Non-square, sheared quadrilateral catches x/y, r/c, and br/bl swaps.
    hand.calibrate((100, 200), (500, 220), (130, 800), (550, 860), 3, 5)
    hand.set_offset(3, -4)
    assert hand.get_screen_pos(0, 0) == (103, 196)
    assert hand.get_screen_pos(0, 4) == (503, 216)
    assert hand.get_screen_pos(2, 0) == (133, 796)
    assert hand.get_screen_pos(2, 4) == (553, 856)
    assert hand.get_screen_pos(1, 2) == (323, 516)
    hand.execute_move([1, 1, 2, 4])
    moves = [args[:2] for action, args, kwargs in events if action == 'move']
    assert moves[0] == hand.get_screen_pos(1, 1)
    assert moves[-1] == (553, 856)
    assert [action for action, args, kwargs in events].count('down') == 1
    assert [action for action, args, kwargs in events].count('up') == 1


def test_hand_no_input_without_calibration_or_input_backend():
    def forbidden(*args, **kwargs):
        raise AssertionError('Uncalibrated or disabled hand must not call device API')
    namespace = {'INPUT_METHOD': 'DIRECT_INPUT', 'pydirectinput': SimpleNamespace(moveTo=forbidden),
                 'time': SimpleNamespace(sleep=forbidden)}
    hand = extract_definitions({'GodHand'}, namespace)['GodHand']()
    hand.execute_move([0, 0, 0, 1])
    hand.calibrate((1, 1), (2, 1), (1, 2), (2, 2), 2, 2)
    namespace['INPUT_METHOD'] = 'NONE'
    hand.execute_move([0, 0, 0, 1])


class FakeDisconnect(Exception):
    pass


class FakeSocket:
    def __init__(self, requests):
        self.requests = list(requests)
        self.messages = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        if not self.requests:
            raise FakeDisconnect()
        return json.dumps(self.requests.pop(0))

    async def send_json(self, payload):
        self.messages.append(json.loads(json.dumps(payload)))


class FakeExecutor:
    def shutdown(self, **kwargs):
        pass


async def no_sleep(seconds):
    pass


class ImmediateLoop:
    def run_in_executor(self, executor, function, *args):
        future = asyncio.get_running_loop().create_future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future


def exercise_endpoint(requests, results=None, matrix=None, hand=None):
    results = iter(results or [])
    collector = []
    calls = []

    def worker(args):
        calls.append(args)
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)

    def capture(**kwargs):
        assert kwargs == {'coords': None, 'use_timestamp': True, 'silent': True}
        return object(), 'synthetic.png'

    namespace = {'WebSocket': FakeSocket, 'WebSocketDisconnect': FakeDisconnect,
        'ProcessPoolExecutor': FakeExecutor, 'EXECUTORS': set(),
        'asyncio': SimpleNamespace(sleep=no_sleep, get_running_loop=ImmediateLoop,
                                  get_event_loop=ImmediateLoop, as_completed=asyncio.as_completed),
        'json': json, 'random': random, 'datetime': datetime, 'INPUT_METHOD': 'NONE',
        'data_collector': SimpleNamespace(save_record=collector.append),
        '_solve_process_hydra': worker, 'god_hand': hand or SimpleNamespace(is_calibrated=False),
        'OCR_AVAILABLE': matrix is not None, 'auto_capture_and_unwarp': capture,
        'recognizer': SimpleNamespace(recognize_board=lambda path: matrix),
        'os': SimpleNamespace(path=SimpleNamespace(basename=lambda path: path))}
    endpoint = extract_definitions({'websocket_endpoint'}, namespace)['websocket_endpoint']
    socket = FakeSocket(requests)
    asyncio.run(endpoint(socket))
    assert socket.accepted and not namespace['EXECUTORS']
    return socket.messages, collector, calls


def start_request(threads=1):
    return {'cmd': 'START', 'map': [1, 1], 'vals': [1, 9], 'rows': 1, 'cols': 2,
            'beamWidth': 8, 'mode': 'complete', 'threads': threads}


def solution(score=2):
    return {'worker_id': 17, 'path': [[0, 0, 0, 1]] if score else [], 'score': score,
            'iterations': 0, 'personality': {'name': 'fake'}, 'optimal': score == 2,
            'full_clear': score == 2, 'upper_bound': 2, 'status': 'full_clear' if score == 2 else 'time_limit',
            'remaining': 2 - score, 'initial_live': 2}


def test_ocr_websocket_preserves_raw_data_and_matrix():
    matrix = [[(r + c) % 10 for c in range(10)] for r in range(16)]
    messages, collector, calls = exercise_endpoint([{'cmd': 'RUN_OCR'}], matrix=matrix)
    result = next(msg for msg in messages if msg['type'] == 'OCR_RESULT')
    assert result['matrix'] == matrix
    assert result['raw_data'] == ''.join(str(cell) for row in matrix for cell in row)
    assert len(result['raw_data']) == 160
    assert not collector and not calls


def test_success_protocol_keeps_old_fields_and_adds_proof_metadata():
    messages, records, calls = exercise_endpoint([start_request()], [solution()])
    result = next(msg for msg in messages if msg['type'] == 'BETTER_SOLUTION')
    assert result['score'] == 2 and result['worker'] == 17 and result['path'] == [[0, 0, 0, 1]]
    assert result['optimal'] is True and result['full_clear'] is True
    assert result['upper_bound'] == 2 and result['status'] == 'full_clear'
    done = next(msg for msg in messages if msg['type'] == 'DONE')
    assert done['success'] is True
    assert len(records) == len(calls) == 1
    assert calls[0][2:6] == (1, 2, 8, 'complete')


def test_all_worker_errors_are_reported_as_unsuccessful_completion():
    messages, records, calls = exercise_endpoint([start_request(2)],
                                                [ValueError('bad board'), ImportError('numba missing')])
    assert any(msg['type'] == 'SOLVER_ERROR' for msg in messages)
    assert not any(msg['type'] == 'BETTER_SOLUTION' for msg in messages)
    assert next(msg for msg in messages if msg['type'] == 'DONE')['success'] is False
    assert not records and len(calls) == 2


def test_one_worker_failure_does_not_discard_successful_worker():
    messages, records, calls = exercise_endpoint([start_request(2)], [ValueError('bad worker'), solution()])
    assert any(msg['type'] == 'SOLVER_ERROR' for msg in messages)
    assert next(msg for msg in messages if msg['type'] == 'DONE')['success'] is True
    assert next(msg for msg in messages if msg['type'] == 'BETTER_SOLUTION')['score'] == 2
    assert len(records) == 1 and len(calls) == 2


def run_ui(extra):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed only for isolated JavaScript UI contract checks')
    scripts = re.findall(r'<script(?:\s[^>]*)?>(.*?)</script>', HTML.read_text(encoding='utf-8'), re.S)
    script = next(script for script in scripts if 'const GameState' in script)
    harness = r'''
const vm = require('vm');
const nodes = new Map();
const element = () => ({innerText: '', innerHTML: '', disabled: false, style: {}, value: '8',
  classList: {add(){}, remove(){}, contains(){return false;}}, appendChild(){}});
const document = {body: element(), createElement: element,
  getElementById(id){ if(!nodes.has(id)) nodes.set(id, element()); return nodes.get(id); },
  querySelector(){ return element(); }, querySelectorAll(){ return []; }};
class Socket { static OPEN = 1; constructor(){this.readyState = 1; this.sent = [];}
  send(value){ this.sent.push(JSON.parse(value)); } }
const context = vm.createContext({document, WebSocket: Socket,
  window:{innerWidth:1280, innerHeight:800}, setTimeout(){return 0;}, clearTimeout(){},
  setInterval(){return 0;}, clearInterval(){}, console,
  fetch(){throw new Error('Network forbidden in UI tests');}});
'''
    harness += '\nconst result = vm.runInContext(' + json.dumps(script + '\n' + extra) + ', context);'
    harness += '\nprocess.stdout.write(JSON.stringify(result));'
    output = subprocess.run([node, '-'], input=harness, text=True, capture_output=True, check=True, timeout=15)
    return json.loads(output.stdout)


def test_ui_default_is_god_and_ocr_start_payload_keeps_row_major_schema():
    result = run_ui('''
const initialMode = GameState.mode;
handleServerMsg({type:'OCR_RESULT', raw_data:'19'.repeat(80)});
setMode('complete');
startSolver();
({initialMode, payload:GameState.socket.sent.at(-1), rows:GameState.rows, cols:GameState.cols});
''')
    assert result['initialMode'] == 'god'
    assert result['rows'] == 16 and result['cols'] == 10
    payload = result['payload']
    assert payload['cmd'] == 'START' and payload['mode'] == 'complete'
    assert payload['map'] == [1] * 160 and payload['vals'] == [1, 9] * 80
    assert (payload['rows'], payload['cols']) == (16, 10)


def test_ui_failed_or_empty_result_never_arms_hand_or_claims_optimality():
    result = run_ui('''
setMode('complete');
handleServerMsg({type:'SOLVER_ERROR', msg:'bad board'});
handleServerMsg({type:'DONE', success:false, msg:'No solution available'});
({state:GameState.execState, solution:GameState.bestSolution,
  disabled:document.getElementById('btn-exec').disabled,
  toast:document.getElementById('toast-msg').innerText});
''')
    assert result['state'] == 'disabled' and result['disabled'] is True
    assert not result['solution'] and 'optimal path' not in result['toast'].lower()
    result = run_ui('''
setMode('complete');
handleServerMsg({type:'BETTER_SOLUTION', score:0, path:[], worker:17, optimal:false, full_clear:false});
handleServerMsg({type:'DONE', success:true, optimal:false, full_clear:false});
({state:GameState.execState, disabled:document.getElementById('btn-exec').disabled,
  toast:document.getElementById('toast-msg').innerText});
''')
    assert result['state'] == 'disabled' and result['disabled'] is True
    assert 'optimal path' not in result['toast'].lower()


def isolated_worker_code(package, block_numba=False):
    """A fresh process must not inherit pytest's extra import search paths."""
    return f'''
import ast, builtins, json, sys
from pathlib import Path
import numpy as np
root = Path({str(ROOT)!r})
# Package launch exposes the repository; script launch exposes Head only.
sys.path = [p for p in sys.path if p and Path(p).resolve() not in {{root, root / 'Head'}}]
sys.path.insert(0, str(root if {package!r} else root / 'Head'))
forbidden = {{'torch', 'torchvision', 'cv2', 'PIL', 'pyautogui', 'pydirectinput',
              'fastapi', 'uvicorn', 'auto_capture', 'predict'}}
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in forbidden:
        raise AssertionError('Unexpected device/web/OCR dependency: ' + name)
    if {block_numba!r} and name.split('.')[0] == 'numba':
        raise ImportError('Numba intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
source = root / 'Head/god_brain.py'
node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
            if isinstance(n, ast.FunctionDef) and n.name == '_solve_process_hydra')
namespace = {{'np': np, '__name__': 'Head._isolated_worker' if {package!r} else '_isolated_worker',
             '__package__': 'Head' if {package!r} else None}}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
with np.errstate(over='ignore'):
    result = namespace['_solve_process_hydra'](([1] * 6, [1, 9, 2, 8, 5, 5],
                                               2, 3, 16, 'complete', 17, None, {{}}))
assert result['score'] == 6 and result['full_clear']
assert not (forbidden & set(sys.modules)), forbidden & set(sys.modules)
if {block_numba!r}:
    assert 'numba' not in sys.modules
print(json.dumps(result))
'''


@pytest.mark.parametrize('package', [False, True], ids=['script-no-numba', 'package-no-numba'])
def test_complete_worker_without_optional_numba_or_device_libraries(package):
    process = subprocess.run([sys.executable, '-c', isolated_worker_code(package, block_numba=True)],
                             cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr
    replay([[1, 9, 2], [8, 5, 5]], json.loads(process.stdout))


@pytest.mark.parametrize('package_first', [False, True], ids=['script-then-package', 'package-then-script'])
def test_script_and_package_routes_can_share_numba_disk_cache(package_first, tmp_path):
    # Compilation cache contains a serialized module name. Both launch forms
    # must work after the other has populated that cache, in fresh processes.
    import os
    environment = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / 'numba-cache'))
    for package in (package_first, not package_first):
        process = subprocess.run([sys.executable, '-c', isolated_worker_code(package)],
                                 cwd=ROOT, env=environment, text=True, capture_output=True, timeout=90)
        assert process.returncode == 0, process.stderr
        replay([[1, 9, 2], [8, 5, 5]], json.loads(process.stdout))


@pytest.mark.parametrize('command', ['EXECUTE_PATH', 'EMERGENCY_EXECUTE'])
def test_existing_execute_commands_forward_rectangles_unchanged_to_stub_hand(command):
    path = [[0, 1, 2, 3], [4, 0, 5, 2]]
    executed = []
    hand = SimpleNamespace(is_calibrated=True, execute_move=executed.append)
    messages, records, calls = exercise_endpoint([{'cmd': command, 'path': path}], hand=hand)
    assert executed == path
    assert any(msg['type'] == 'EXEC_PROGRESS' and msg['total'] == 2 for msg in messages)
    assert not records and not calls


@pytest.mark.parametrize('optimal,expected', [(False, 'Best-found'), (True, 'Verified upper bound')])
def test_ui_nonempty_result_arms_hand_and_reports_only_proven_optimality(optimal, expected):
    result = run_ui('''
setMode('complete');
handleServerMsg({type:'BETTER_SOLUTION', score:2, path:[[0,0,0,1]], worker:17});
const armed = GameState.execState;
handleServerMsg({type:'DONE', success:true, optimal:''' + json.dumps(optimal) + '''});
({armed, state:GameState.execState, disabled:document.getElementById('btn-exec').disabled,
  toast:document.getElementById('toast-msg').innerText});
''')
    assert result['armed'] == 'armed' and result['state'] == 'hot'
    assert result['disabled'] is False and expected in result['toast']


def test_new_ui_search_clears_stale_path_before_any_worker_response():
    result = run_ui('''
handleServerMsg({type:'OCR_RESULT', raw_data:'19'.repeat(80)});
GameState.bestSolution = [[0,0,0,1]];
setExecState('hot');
document.getElementById('btn-run').innerText = 'INITIALIZE';
startSolver();
({solution:GameState.bestSolution, state:GameState.execState,
  disabled:document.getElementById('btn-exec').disabled});
''')
    assert result == {'solution': None, 'state': 'disabled', 'disabled': True}


@pytest.mark.parametrize('imports', [
    'import certified_solver\nimport Head.certified_solver',
    'import Head.certified_solver\nimport certified_solver',
    'import Head\nimport certified_solver\nimport Head.certified_solver',
], ids=['script-then-package-attribute', 'package-then-script-attribute', 'parent-before-script-attribute'])
def test_mixed_public_imports_bind_package_attribute_in_fresh_process(imports):
    code = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
sys.path.insert(0, {str(ROOT / 'Head')!r})
{imports}
assert Head.certified_solver is certified_solver
assert Head.certified_solver.solve is certified_solver.solve
assert sys.modules['Head.certified_solver'] is sys.modules['certified_solver']
assert not ({{'pyautogui', 'pydirectinput', 'cv2', 'torch', 'fastapi'}} & set(sys.modules))
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, text=True,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
