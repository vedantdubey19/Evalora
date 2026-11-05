"""CI-only container acceptance check with generated dummy credentials; no Gemini calls."""
import base64
import json
from pathlib import Path
import ssl
import subprocess
import time
import urllib.error
import urllib.request

from dotenv import dotenv_values
from deploy.configure import configure


def main():
    configure('localhost', 'ci@example.invalid', 'unused-ci-key')
    env = dotenv_values('.env.production')
    credentials = dict(line.split(': ', 1) for line in Path('.credentials.local').read_text().splitlines())
    compose = ['docker', 'compose', '--env-file', '.env.production']
    subprocess.run(compose+['config', '--quiet'], check=True)
    subprocess.run(compose+['build', 'api'], check=True)
    subprocess.run(compose+['up', '-d', '--no-build', '--wait', '--wait-timeout', '180'], check=True)
    context = ssl._create_unverified_context()  # CI localhost CA is not installed in runner trust.

    def request(path, headers=None):
        try:
            with urllib.request.urlopen(urllib.request.Request('https://localhost'+path, headers=headers or {}),
                                        context=context, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    for _ in range(40):
        try:
            if request('/api/health')[0] == 200:
                break
        except OSError:
            pass
        time.sleep(1)
    else:
        raise RuntimeError('Gateway did not become ready.')
    assert request('/')[0] == 401
    assert request('/_stcore/health')[0] == 401
    assert request('/api/reports')[0] == 401
    basic = base64.b64encode(('admin:'+credentials['Dashboard password']).encode()).decode()
    assert request('/', {'Authorization': 'Basic '+basic})[0] == 200
    auth = {'Authorization': 'Bearer '+env['API_ACCESS_TOKEN']}
    assert request('/api/reports', auth)[0] == 200
    # Create a report through the image's real atomic writer, then recreate both apps.
    script = "from app.services.report_store import ReportWriter; ReportWriter('/var/lib/evalora/reports').finish({'results': []})"
    subprocess.run(compose+['exec', '-T', 'api', 'python', '-c', script], check=True)
    subprocess.run(compose+['up', '-d', '--force-recreate', '--wait', 'api', 'dashboard'], check=True)
    report_names = json.loads(request('/api/reports', auth)[1])['reports']
    assert len(report_names) == 2
    name = next(name for name in report_names if name.endswith('.json'))
    assert json.loads(request('/api/reports/'+name, auth)[1])['results'] == []
    # Shared HTTP quota also survives app recreation; at most 30 admitted per minute.
    statuses = []
    for _ in range(100):
        statuses.append(request('/api/reports', auth)[0])
        if statuses[-1] == 429:
            break
    assert 429 in statuses
    print('Container checks passed: HTTPS gateway, dashboard login, API auth, persistent reports and request limits. No provider calls.')


if __name__ == '__main__':
    main()
