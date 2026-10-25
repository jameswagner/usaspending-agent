"""Replay the public query tasks, preserving request/response evidence."""

import argparse
import concurrent.futures
import datetime
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def resolve(value, context):
    if isinstance(value, dict):
        return {key: resolve(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, context) for item in value]
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r'\{\{(\w+)\}\}', value)
    if match:
        return context[match[1]]
    return re.sub(r'\{\{(\w+)\}\}', lambda match: str(context[match[1]]), value)


def at_path(data, path):
    for component in path.split('.'):
        data = data[int(component)] if isinstance(data, list) else data[component]
    return data


def execute(task, context, timeout):
    try:
        spec = resolve(task['request'], context)
    except KeyError as error:
        return {'task_id': task['id'], 'status': 'dependency_unresolved', 'error': str(error)}
    path = spec['path']
    if not path.startswith(('/api/v1/', '/api/v2/')) or '?' in path or '#' in path:
        raise ValueError(f'Unexpected API path: {path}')
    url = 'https://api.usaspending.gov' + path
    if spec.get('query'):
        url += '?' + urllib.parse.urlencode(spec['query'])
    data = json.dumps(spec['body']).encode() if spec['method'] == 'POST' else None
    request = urllib.request.Request(url, data=data, method=spec['method'], headers={
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'USAspending-public-website-audit/1.0',
        'X-Requested-With': 'USASpendingFrontend',
    })
    start = time.monotonic()
    result = {'task_id': task['id'], 'requested_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'request': spec, 'url': url}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result['http_status'] = response.status
            result['content_type'] = response.headers.get('Content-Type')
            raw = response.read()
    except urllib.error.HTTPError as error:
        result['http_status'] = error.code
        raw = error.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        result.update(status='transport_error', error=str(error), elapsed_seconds=round(time.monotonic()-start, 3))
        return result
    result['elapsed_seconds'] = round(time.monotonic()-start, 3)
    result['response_bytes'] = len(raw)
    try:
        result['response'] = json.loads(raw)
        result['status'] = 'live_ok' if 200 <= result['http_status'] < 300 else 'http_error'
    except (json.JSONDecodeError, UnicodeDecodeError):
        result['response_text'] = raw.decode(errors='replace')[:20000]
        result['status'] = 'non_json_response'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=int, required=True)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--timeout', type=int, default=35)
    args = parser.parse_args()
    inventory = json.loads((ROOT / 'inventory.json').read_text())
    context_path = ROOT / 'context.json'
    context = json.loads(context_path.read_text()) if context_path.exists() else {}
    directory = ROOT / 'evidence'
    directory.mkdir(exist_ok=True)
    tasks = [task for task in inventory['tasks'] if task['phase'] == args.phase and task['execute'] and (not args.ids or task['id'] in args.ids)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(execute, task, context.copy(), args.timeout): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            result = future.result()
            evidence_path = directory / f'{task["id"]}.json'
            if evidence_path.exists():
                prior = json.loads(evidence_path.read_text())
                result['previous_attempts'] = prior.pop('previous_attempts', []) + [prior]
            evidence_path.write_text(json.dumps(result, indent=2) + '\n')
            if result['status'] == 'live_ok':
                for name, path in task.get('extract_context', {}).items():
                    try:
                        context[name] = at_path(result['response'], path)
                    except (KeyError, IndexError, TypeError, ValueError) as error:
                        print(f'{task["id"]} context {name} failed: {error}', flush=True)
            context_path.write_text(json.dumps(context, indent=2) + '\n')
            print(task['id'], result['status'], result.get('http_status', ''), result.get('elapsed_seconds', ''), flush=True)


if __name__ == '__main__':
    main()
