# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

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

new_schedule_mutation = gql('''
                            mutation {
                              newSchedule(newScheduleInput:{ startTime:"2018-10-01 08:00:00",
                                                             endTime:"2018-10-03 08:00:00"}){
                                 __typename
                                 ... on NewScheduleSuccess{
                                    success
                                }
                                 ... on NewScheduleError{
                                    error
                                }
                              }
                            }
                            ''')
