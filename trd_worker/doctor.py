"""
trd-worker doctor — end-to-end environment diagnostic.

Drop this into trd_worker/doctor.py. Then register in cli.py:

    from trd_worker import doctor
    doctor_parser = subparsers.add_parser('doctor', help='Run diagnostic')
    doctor_parser.add_argument('--json', action='store_true')
    doctor_parser.set_defaults(func=doctor.cmd_doctor)

Runs 8 checks: Python version, installed version, PyPI latest, backend
reachability, power source, GPU detect, disk space, models directory.

Exit codes: 0=pass, 1=critical fail, 2=warnings only.
"""
from __future__ import annotations
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

_IS_TTY = sys.stdout.isatty()


def _c(s: str, code: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if _IS_TTY else s


def green(s):  return _c(s, '32')
def red(s):    return _c(s, '31')
def yellow(s): return _c(s, '33')
def dim(s):    return _c(s, '2')
def bold(s):   return _c(s, '1')


@dataclass
class Check:
    name: str
    status: str
    message: str = ''
    detail: dict = field(default_factory=dict)
    critical: bool = True

    @property
    def icon(self) -> str:
        return {
            'pass': green('OK '),
            'warn': yellow('WARN'),
            'fail': red('FAIL'),
            'skip': dim('-- '),
        }.get(self.status, '?')


def _safe_run(cmd, timeout=5.0):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _check_python() -> Check:
    v = sys.version_info
    if v >= (3, 9):
        return Check('Python version', 'pass', f'{v.major}.{v.minor}.{v.micro}')
    return Check('Python version', 'fail', f'{v.major}.{v.minor} (need 3.9+)')


def _check_installed_version() -> Check:
    try:
        from trd_worker import __version__
        return Check('trd-worker installed', 'pass', f'v{__version__}',
                     detail={'version': __version__})
    except Exception as e:
        return Check('trd-worker installed', 'fail', f'cannot import: {e}')


def _check_pypi_latest() -> Check:
    try:
        import urllib.request
        req = urllib.request.Request(
            'https://pypi.org/pypi/trd-worker/json',
            headers={'User-Agent': 'trd-worker-doctor/1.0'}
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
            latest = data['info']['version']
        try:
            from trd_worker import __version__ as current
        except Exception:
            return Check('PyPI latest', 'skip', 'no installed version to compare',
                         critical=False)
        if current == latest:
            return Check('PyPI latest', 'pass', f'on latest (v{latest})',
                         detail={'latest': latest, 'current': current})
        return Check('PyPI latest', 'warn',
                     f'v{current} installed, v{latest} available',
                     detail={'latest': latest, 'current': current,
                             'upgrade_cmd': 'pip install -U trd-worker'},
                     critical=False)
    except Exception as e:
        return Check('PyPI latest', 'skip',
                     f'check failed ({type(e).__name__})',
                     critical=False)


def _check_backend_reachable() -> Check:
    backend = os.environ.get(
        'TRD_BACKEND_URL',
        'https://trd-cn-backend-production.up.railway.app'
    )
    try:
        import urllib.request
        req = urllib.request.Request(backend, method='HEAD')
        start = time.time()
        with urllib.request.urlopen(req, timeout=5) as resp:
            elapsed_ms = round((time.time() - start) * 1000)
            if 200 <= resp.status < 500:
                return Check('Backend reachable', 'pass',
                             f'{elapsed_ms}ms ({backend})',
                             detail={'backend': backend, 'latency_ms': elapsed_ms,
                                     'status': resp.status})
            return Check('Backend reachable', 'warn',
                         f'HTTP {resp.status}', critical=False)
    except Exception as e:
        return Check('Backend reachable', 'fail',
                     f'{type(e).__name__}',
                     detail={'backend': backend})


def _check_power_source() -> Check:
    try:
        if platform.system() == 'Darwin':
            out = _safe_run(['pmset', '-g', 'batt'])
            if out and 'AC Power' in out:
                return Check('Power source', 'pass', 'AC (plugged in)')
            elif out and 'Battery Power' in out:
                return Check('Power source', 'warn',
                             'on battery (pauses if pause_when_on_battery=true)',
                             critical=False)
        elif platform.system() == 'Linux':
            sysfs = '/sys/class/power_supply/AC/online'
            if os.path.exists(sysfs):
                with open(sysfs) as f:
                    if f.read().strip() == '1':
                        return Check('Power source', 'pass', 'AC (plugged in)')
                return Check('Power source', 'warn', 'on battery', critical=False)
    except Exception:
        pass
    return Check('Power source', 'skip', 'could not detect', critical=False)


def _check_gpu() -> Check:
    nvidia = _safe_run(['nvidia-smi', '--query-gpu=name,memory.total',
                        '--format=csv,noheader'])
    if nvidia:
        gpus = [line.strip() for line in nvidia.split('\n') if line.strip()]
        return Check('GPU detected', 'pass',
                     f'NVIDIA: {gpus[0] if gpus else "yes"}',
                     detail={'vendor': 'nvidia', 'gpus': gpus})

    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        out = _safe_run(['system_profiler', 'SPDisplaysDataType'])
        if out and 'Apple' in out:
            chipset = 'Apple Silicon'
            for line in out.splitlines():
                if 'Chipset Model' in line:
                    chipset = line.split(':', 1)[1].strip()
                    break
            return Check('GPU detected', 'pass',
                         f'Apple Silicon: {chipset}',
                         detail={'vendor': 'apple', 'chipset': chipset})

    return Check('GPU detected', 'warn',
                 'no GPU detected, CPU-only will be very slow',
                 critical=False)


def _check_disk_space() -> Check:
    home = os.path.expanduser('~')
    try:
        stat = shutil.disk_usage(home)
        free_gb = stat.free / (1024 ** 3)
        if free_gb >= 10:
            return Check('Disk space', 'pass', f'{free_gb:.1f} GB free',
                         detail={'free_gb': round(free_gb, 1)})
        elif free_gb >= 5:
            return Check('Disk space', 'warn',
                         f'{free_gb:.1f} GB free (tight for multiple models)',
                         critical=False)
        return Check('Disk space', 'fail',
                     f'{free_gb:.1f} GB free (insufficient)',
                     detail={'free_gb': round(free_gb, 1)})
    except Exception:
        return Check('Disk space', 'skip', 'could not check', critical=False)


def _check_models_dir() -> Check:
    model_dir = os.path.expanduser(
        os.environ.get('TRD_WORKER_MODELS_DIR', '~/.trd-worker/models')
    )
    if not os.path.exists(model_dir):
        return Check('Models directory', 'warn',
                     f'{model_dir} does not exist',
                     detail={'path': model_dir},
                     critical=False)
    try:
        entries = [f for f in os.listdir(model_dir) if not f.startswith('.')]
        if not entries:
            return Check('Models directory', 'warn',
                         'empty (run: trd-worker models pull qwen2.5-7b-instruct)',
                         detail={'path': model_dir, 'count': 0},
                         critical=False)
        preview = ', '.join(entries[:3])
        more = f' +{len(entries)-3} more' if len(entries) > 3 else ''
        return Check('Models directory', 'pass',
                     f'{len(entries)} model(s): {preview}{more}',
                     detail={'path': model_dir, 'count': len(entries),
                             'models': entries})
    except Exception as e:
        return Check('Models directory', 'skip', str(e), critical=False)


def _print_report(checks, use_json: bool):
    if use_json:
        payload = {
            'timestamp': time.time(),
            'platform': platform.system(),
            'arch': platform.machine(),
            'checks': [asdict(c) for c in checks],
            'summary': {
                'total': len(checks),
                'passed': sum(1 for c in checks if c.status == 'pass'),
                'warned': sum(1 for c in checks if c.status == 'warn'),
                'failed': sum(1 for c in checks if c.status == 'fail'),
                'skipped': sum(1 for c in checks if c.status == 'skip'),
            }
        }
        print(json.dumps(payload, indent=2))
        return

    print()
    print(bold('  trd-worker doctor  '))
    print()
    for c in checks:
        line = f'  [{c.icon}]  {c.name:.<28} {c.message}'
        if c.status == 'pass':
            print(line)
        elif c.status == 'warn':
            print(yellow(line))
        elif c.status == 'fail':
            print(red(line))
        else:
            print(dim(line))
    print()

    fails = [c for c in checks if c.status == 'fail' and c.critical]
    warns = [c for c in checks if c.status == 'warn']
    if fails:
        print(red(f'  {len(fails)} critical check(s) failed.'))
    elif warns:
        print(yellow(f'  {len(warns)} warning(s). Worker may underperform.'))
    else:
        print(green('  All checks passed. Run: trd-worker start'))
    print()


def cmd_doctor(args=None):
    use_json = getattr(args, 'json', False) if args else '--json' in sys.argv

    checks = [
        _check_python(),
        _check_installed_version(),
        _check_pypi_latest(),
        _check_backend_reachable(),
        _check_power_source(),
        _check_gpu(),
        _check_disk_space(),
        _check_models_dir(),
    ]

    _print_report(checks, use_json)

    fails = [c for c in checks if c.status == 'fail' and c.critical]
    warns = [c for c in checks if c.status == 'warn']
    if fails:
        sys.exit(1)
    if warns:
        sys.exit(2)
    sys.exit(0)


if __name__ == '__main__':
    cmd_doctor()
