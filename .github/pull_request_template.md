## Scope

<!-- CI auto-detects this, but it helps reviewers to see at a glance. -->

- [ ] Backend (`backend/`)
- [ ] Frontend (`frontend/`)
- [ ] Docs (`docs/`)
- [ ] CI/Infra (`.github/`)


<!-- 
  REQUIRED: Add a changelog fragment file to this PR.
  
  Quickest way (auto-detects ticket from branch name):
    ./scripts/changelog-fragment "Your change description"
    ./scripts/changelog-fragment "Your change description" fix
    ./scripts/changelog-fragment "" misc

  Manual:
    echo "Description" > changelog.d/GSCHED-190.feature.md

  Types: feature, fix, breaking, deprecation, improvement, doc, misc
  CI will fail if no fragment is found (unless only CI/docs files changed).
-->


## Jira

<!-- JIRA:START -->
<!-- Auto-linked from branch name (e.g. GSCHED-190/description) by the PR Jira Link workflow. -->
<!-- JIRA:END -->

## Checklist

- [ ] Changelog fragment added to `changelog.d/`
- [ ] Commit messages follow conventional commits (`feat(backend):`, `fix(frontend):`, etc.)
- [ ] No breaking changes (or `breaking` fragment added)
