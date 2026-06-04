# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Background visibility-aggregator: keeps the Sight DB's Stage 1 (targets) and
Stage 2 (observations) current for the available GPP programs.

Run as a Heroku Scheduler one-off dyno:
    python -m scheduler.services.visibility_aggregator.runner
"""
