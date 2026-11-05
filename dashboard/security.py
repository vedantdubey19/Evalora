"""Only the HTTPS gateway may open the production dashboard, including reruns."""
import hmac
import streamlit as st


def gateway_authenticated(settings, headers):
    if settings.deployment_mode != 'production':
        return True
    supplied = headers.get('X-Evalora-Gateway', '')
    return hmac.compare_digest(supplied.encode(), settings.gateway_secret.get_secret_value().encode())


def require_gateway(settings):
    if not gateway_authenticated(settings, st.context.headers):
        st.error('Open this dashboard through the authenticated HTTPS gateway.')
        st.stop()
