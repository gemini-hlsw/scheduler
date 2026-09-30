# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from os import environ

from gql import gql

__all__ = [
    "RESOURCE_URL",
    "TELESCOPE_NIGHTS_QUERY",
]

RESOURCE_URL = environ.get('RESOURCE_URL', "http://localhost:4001/graphql")

# Shared between OpsResourceService (queried synchronously, to determine instrument/FPU
# availability) and night_monitor.ResourceEventSource (queried asynchronously, to notice
# changes). Kept dependency-free (no scheduler-internal imports) so that either side can
# import it without pulling the other in and creating a circular import.
TELESCOPE_NIGHTS_QUERY = gql(
    """
    query Resources($site: Site!, $start: Date!, $end: Date!) {
        telescopeNights(site: $site, nights: {
            start: $start,
            end: $end
        }) {
            site
            observingNight
            interval {
                start
                end
            }
            dataAvailable
            subsystems {
                subsystem
            }
            instrumentAvailability {
                interval {
                    start
                    end
                }
                instrument
                usage
            }
            components {
                interval {
                    start
                    end
                }
                usage
                component {
                    code
                    instrument
                    componentType
                    name
                    barcode
                    aliases
                }
            }
        }
    }
    """
)
