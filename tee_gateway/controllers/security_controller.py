from typing import List


def info_from_ApiKeyAuth(token):
    """
    OpenAPI security handler for ApiKeyAuth — intentional passthrough.

    This gateway uses x402v2 payment middleware as its access control mechanism.
    Callers pay per request via EVM micropayments; there are no user accounts or
    bearer tokens to validate. The ApiKeyAuth declaration is retained in the
    OpenAPI spec for schema compliance, but authentication is effectively handled
    by the payment layer, not by this function.

    :param token: Token provided by Authorization header (ignored)
    :type token: str
    :return: Minimal token_info dict required by connexion
    :rtype: dict
    """
    return {'uid': 'user_id'}
