
from common.queries import subscribe

def test_subscription_query(mocker):
    """
    Test a subscription GraphQL query
    """
    def response(self):
        yield '{"data": {"subscription_query": {"data": "test"}}}'
    mocker.patch(
        'common.queries.subscribe',
        response
    )


