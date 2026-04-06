## Scope

<!-- CI auto-detects this, but it helps reviewers to see at a glance. -->

- [ ] Backend (`backend/`)
- [ ] Frontend (`frontend/`)
- [ ] Docs (`docs/`)
- [ ] CI/Infra (`.github/`)

## Changelog

<!-- 
  REQUIRED: Add a changelog fragment file to this PR.
  
  Quickest way (auto-detects ticket from branch name):
    ./scripts/changelog-fragment "Your change description"
    ./scripts/changelog-fragment "Your change description" fix
    ./scripts/changelog-fragment "" misc

  Manual:
    echo "Description" > changelog.d/REL-190.feature.md

  Types: feature, fix, breaking, deprecation, improvement, doc, misc
  CI will fail if no fragment is found (unless only CI/docs files changed).
-->

**Fragment:** `changelog.d/` <!-- auto-populated by PR Changelog Sync workflow -->

## Jira

<!-- Auto-linked from branch name (e.g. REL-190/description) -->

## Checklist

- [ ] Changelog fragment added to `changelog.d/`
- [ ] Commit messages follow conventional commits (`feat(backend):`, `fix(frontend):`, etc.)
- [ ] No breaking changes (or `breaking` fragment added)
