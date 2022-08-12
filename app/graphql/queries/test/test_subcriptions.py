import pytest
import gql
import asyncio
from app.graphql.queries import Session, test_subscription_query
from unittest.mock import patch


@pytest.fixture
def session():
    return Session()


class MockClient(gql.Client):

    def __init__(self, *args, **kwargs):
        self.transport = None
    
    async def subscribe(self, query):
        yield {"observationEdit": {"editType": "UPDATE", "value": {"id": "o-6"}}}
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@patch('app.graphql.session.Client', MockClient)
@pytest.mark.asyncio
async def test_on_update(session):
    """
    Test a subscription GraphQL subcription change
    """
   
    # mocked_client.return_value.__aenter__.return_value.subscribe = subscribe_to_one
    resp = await session.on_update(test_subscription_query)
    assert isinstance(resp, dict)  # Check that only one response is returned
    assert resp['observationEdit']['editType'] == 'UPDATE'
    assert resp['observationEdit']['value']['id'] == 'o-6'


@patch('app.graphql.session.Client', MockClient)
@pytest.mark.asyncio
async def test_subscribe_all(session):
    """
    Test an array of subscriptions
    """
    resp = await session.subscribe_all()
    print(type(resp[0].pop()))
    assert isinstance(resp, tuple)
    first_task = resp[0].pop()
    assert isinstance(first_task, asyncio.Task)
    assert first_task.done()  # Check that the first task is done
