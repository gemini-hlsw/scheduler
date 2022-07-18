import pytest
from common.queries import Session, test_subscription_query
from unittest.mock import AsyncMock, patch


def subscribe_to_one():
    return {"observationEdit": {"editType": "UPDATE", "value": {"id": "o-6"}}}


def subscribe_to_all(self, session, query):
    yield ({"task1": "done"})

@pytest.fixture
def session():
    return Session()

# @patch('common.queries.Session._subscribe', return_value=subscribe_to_one(), autospec=True)
@patch('gql.Client')
# @patch('common.queries.Session._subscribe', return_value=subscribe_to_one(), autospec=True)
@pytest.mark.asyncio
async def test_on_update(mocked_client, session):
    """
    Test a subscription GraphQL subcription change
    """
    print("dfsadadasd")
    mocked_client.return_value.__aenter__.return_value = subscribe_to_one
    resp = await session.on_update(test_subscription_query)
    
    print(resp)
    assert isinstance(resp, dict)  # Check that only one response is returned
    assert mocked_client.assert_called_once()
    # assert resp['observationEdit']['editType'] == 'UPDATE'
    # assert resp['observationEdit']['value']['id'] == 'o-6'

'''
@pytest.mark.asyncio
async def test_subscribe_all(session):
    """
    Test a subscription GraphQL query
    """
    resp = await session.subscribe_all()
    assert isinstance(resp, tuple)
'''
