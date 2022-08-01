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

observation_update = gql('''
                            subscription{
                              observationEdit{
                                editType,
                                value{
                                   id
                                },
                                id
                              }
                            }
                        ''')

program_update = gql('''
                      subscription{
                        programEdit{
                          editType,
                          value{
                            id
                          },
                          id
                        }
                      }
                    ''')

target_update = gql('''
                      subscription{
                        targetEdit{
                          editType,
                          value{
                            id
                          },
                          id
                        }
                      }
                    ''')
