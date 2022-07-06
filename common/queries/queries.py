from gql import gql

test_mutation_query = gql('''
                            query testMutation {
                              data
                            }
                        ''')

test_subscription_query = gql('''
                            subscription testSubscription {
                              data
                             }
                        ''')
