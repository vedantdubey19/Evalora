"""Pure ASGI admission control: authenticate before parsing or invoking handlers."""
import hmac
import sqlite3

from starlette.responses import JSONResponse

from app.services.limits import LimitExceeded, LimitStore


class AdmissionMiddleware:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings
        self.protected = settings.deployment_mode == 'production' or bool(settings.api_access_token.get_secret_value())
        self.store = LimitStore(settings.state_dir) if self.protected else None

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        async def secure_send(message):
            if message['type'] == 'http.response.start':
                message['headers'] = list(message.get('headers', [])) + [
                    (b'x-content-type-options', b'nosniff'),
                    (b'cache-control', b'no-store'),
                    (b'x-frame-options', b'DENY'),
                    (b'referrer-policy', b'no-referrer')]
            await send(message)

        async def reject(status, detail, headers=None):
            await JSONResponse({'detail': detail}, status_code=status, headers=headers)(scope, receive, secure_send)

        if self.protected and scope['path'] != '/health':
            # Duplicate authorization headers are rejected instead of ambiguously parsed.
            credentials = [v for k, v in scope['headers'] if k.lower() == b'authorization']
            expected = ('Bearer ' + self.settings.api_access_token.get_secret_value()).encode()
            if len(credentials) != 1 or not hmac.compare_digest(credentials[0], expected):
                await reject(401, 'Authentication required.', {'WWW-Authenticate': 'Bearer'})
                return
            try:
                self.store.consume('api-minute', self.settings.api_requests_per_minute)
            except LimitExceeded as error:
                await reject(429, str(error), {'Retry-After': str(error.retry_after)})
                return
            except (OSError, sqlite3.Error):
                await reject(503, 'Request limits are temporarily unavailable.')
                return
        # Buffer a bounded body; limits apply to chunked requests too.
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            body.extend(message.get('body', b''))
            if len(body) > self.settings.max_request_bytes:
                await reject(413, 'Request body too large.')
                return
            if not message.get('more_body', False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()

        await self.app(scope, bounded_receive, secure_send)
