"""Generate deployment secrets locally without printing them or replacing files."""
import argparse
import getpass
import os
from pathlib import Path
import re
import secrets
import subprocess


def configure(domain, email, gemini_key, directory=Path('.')):
    if not re.fullmatch(r'(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', domain):
        raise ValueError('Use a hostname without a scheme, port or path.')
    if not re.fullmatch(r'[A-Za-z0-9._+%-]+@[A-Za-z0-9.-]+', email):
        raise ValueError('Use a valid certificate contact email.')
    if not gemini_key or any(c in gemini_key for c in "\r\n'"):
        raise ValueError('A nonempty single-line Gemini key is required.')
    environment, credentials = directory / '.env.production', directory / '.credentials.local'
    if environment.exists() or credentials.exists():
        raise ValueError('Deployment files already exist; preserve them and rotate secrets explicitly.')
    password = secrets.token_urlsafe(32)
    # No secrets in process arguments, stdout or diagnostic errors.
    hashed = subprocess.run(['docker', 'run', '--rm', '-i', 'caddy:2.10-alpine',
                             'caddy', 'hash-password', '--algorithm', 'bcrypt'],
                            input=password+'\n', text=True, capture_output=True, check=False)
    if hashed.returncode or not re.fullmatch(r'\$2[aby]\$[0-9]{2}\$[./A-Za-z0-9]{53}', hashed.stdout.strip()):
        raise ValueError('Password hashing failed. Start Docker and ensure the Caddy image can be pulled.')
    token, gateway, cookie = (secrets.token_urlsafe(48) for _ in range(3))
    values = dict(DOMAIN=domain, TLS_EMAIL=email, DASHBOARD_USER='admin',
                  DASHBOARD_PASSWORD_HASH=hashed.stdout.strip(), API_ACCESS_TOKEN=token,
                  GATEWAY_SECRET=gateway, COOKIE_SECRET=cookie, GEMINI_API_KEY=gemini_key)
    created = []
    try:
        for path, text in [(credentials, 'Dashboard username: admin\nDashboard password: '+password+'\nAPI bearer token: '+token+'\n'),
                           (environment, ''.join(f"{k}='{v}'\n" for k,v in values.items()))]:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created.append(path)
            with os.fdopen(descriptor, 'w') as handle:
                handle.write(text)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return environment, credentials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--email', required=True)
    args = parser.parse_args()
    try:
        configure(args.domain, args.email, getpass.getpass('Gemini API key (hidden): '))
    except (OSError, ValueError) as error:
        parser.exit(1, str(error)+'\n')
    print('Created .env.production and .credentials.local with owner-only permissions. Store the credentials in your password manager. No secrets were printed.')


if __name__ == '__main__':
    main()
