import pytest
from app import App


@pytest.fixture
def app(mocker):
    config = mocker.MagicMock()
    config.configure_mock(**{
                             'session.url': 'http://localhost:8080',
                             'process_manager.size': 1,
                             'process_manager.timeout': 10
                          })
    return App(config)


def test_subscription_query(mocker, app):
    """
    Test a subscription GraphQL query
    """

    def response(self, query):
        yield '{"data": {"subscription_query": {"data": "test"}}}'
    mocker.patch(
        'common.queries.Session.subscribe',
        response
    )
    
    assert app._check_for_updates() == True
