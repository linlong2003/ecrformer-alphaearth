"""Check proposed Git upload files without staging, committing, or contacting GitHub."""

import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_ROOTS = {'datasets', 'results', 'experiments', 'pretrained', 'checkpoints',
                 'logs', 'run_logs', 'lightning_logs', '.aws', '.ssh', '.codex', '.agents'}
BLOCKED_SUFFIXES = {'.ckpt', '.pt', '.pth', '.tif', '.tiff', '.npz', '.npy',
                    '.zip', '.rar', '.pem', '.key', '.p12', '.pfx', '.log'}
SECRET_PATTERNS = (
    re.compile(rb'github_pat_[A-Za-z0-9_]{20,}'),
    re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}'),
    re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----'),
    re.compile(rb'https://[^\s/@]+:[^\s/@]+@github\.com'),
)
LOCAL_PATH = re.compile(rb'(?:/mnt/data/[A-Za-z0-9_]|/home/[A-Za-z0-9_]|[A-Za-z]:\\Users\\)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-file-mib', type=float, default=10)
    args = parser.parse_args()
    raw = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    names = sorted(set(name.decode('utf-8') for name in raw.split(b'\0') if name))
    issues, total = [], 0
    for name in names:
        relative = Path(name)
        path = ROOT / relative
        if path.is_symlink():
            issues.append(f'{name}: symlink requires manual review')
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if (relative.parts[0] in BLOCKED_ROOTS or path.suffix.lower() in BLOCKED_SUFFIXES
                or path.name in {'AGENTS.md', 'README.TXT', '.env', '.netrc', '.git-credentials'}
                or (path.name.startswith('.env.') and path.name != '.env.example')):
            issues.append(f'{name}: private/generated file in upload candidate list')
        if size > args.max_file_mib * 1024**2:
            issues.append(f'{name}: exceeds the preparation size limit')
            continue
        content = path.read_bytes()
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            issues.append(f'{name}: credential-like content (value not printed)')
        if LOCAL_PATH.search(content):
            issues.append(f'{name}: machine-specific absolute path')
    if issues:
        raise SystemExit('\n'.join(issues))
    print(f'PASS: {len(names)} upload candidates, {total / 1024**2:.2f} MiB total.')
    print('No blocked artifacts, known token patterns, or private host paths found. '
          'This check is not a substitute for manual review or a license review.')


if __name__ == '__main__':
    main()
