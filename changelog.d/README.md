This directory contains changelog fragments for Towncrier.

## How to add a fragment

Create a file named: `<issue-or-pr-number>.<type>.md`

Types:
  - feature     → New features
  - fix         → Bug fixes
  - breaking    → Breaking changes
  - deprecation → Deprecations
  - improvement → Improvements (non-feature enhancements)
  - doc         → Documentation changes
  - misc        → Internal changes (content not shown in changelog)

Examples:
  - 142.feature.md     → "Add batch retry with exponential backoff"
  - 155.fix.md         → "Fix timeout handling in multiprocess mode"
  - 160.breaking.md    → "Remove deprecated /v1/sync endpoint"
  - noissue.misc.md    → (no content needed for misc)

If there's no issue/PR number, use a short descriptive name:
  - healthcheck.improvement.md
  - deps-march.misc.md

The content of the file is a single sentence or short paragraph
describing the change from a user's perspective.
