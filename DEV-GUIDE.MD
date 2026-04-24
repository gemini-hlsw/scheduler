Current branch structure. We are still going to be using a main branch to hold both development and production code would be separated by using tags. 

```
 Feature Branch ──► PR to main ──► CI (detects what changed)
                                          │
                                ┌─────────┴─────────┐
                                ▼                   ▼
                         Backend changed?     Frontend changed?
                         ├─ Python lint        ├─ ESLint
                         ├─ pytest             
                         ├─ Docker build       ├─ Vitest (TBI)
                         └─ Schema lint        └─ Vite build
                                │                   │
                                └─────────┬─────────┘
                                          ▼
                                    Merge to main
                                    │         │
                                    ▼         ▼
                           Auto-deploy     Docs deploy
                            to DEV       (GitHub Pages)
                          (selective)
                                    │
                                    ▼
                          Ready for production?
                                    │
                                    ▼
                          Actions → "Promote to PROD"
                           (manual trigger only)
                                    │
                                    ▼
                          Version bump + tag (auto)
                                    │
                                    ▼
                              Approval gate
                                    │
                          ┌─────────┼─────────┐
                          ▼         ▼         ▼
                       Backend   Backend    Frontend
                      Realtime  Multiproc
                          │         │         │
                          └─────────┼─────────┘
                                    ▼
                          GitHub Release + Changelog

```

## Day-to-day workflow

1. Create a branch following the convention <JIRA-ticket>/<description> (e.g. GSCHED-955/modify-steps-in-atom-sequence-parsing). hotfix is also available for fast issues that might not need tracking in JIRA.

2. Make changes in backend/, frontend/, or both

3. Add fragment: ./scripts/changelog-fragment "Add steps in ProgramProvider" feature . The scripts looks for this convention `<fragment comment> <type>`.\
Types: `feature fix breaking deprecation improvement doc misc`

4. Commit with the template: `<type>[(<scope>)]: <description> `\
Types: `feat|fix|chore|docs|refactor|test|ci|style|perf|build|revert`\
Scopes: `backend|frontend|`. Can be empty if is outside this two.\
Example: feat(backend): Add step count in ProgramProvider

5. Push → CI runs relevant checks

6. Open PR to main → changelog auto-populates in PR description

7. Review, approve, merge

8. Dev auto-deploys (only changed components)

## Promoting to Production

1. Verify dev is working as expected

2. Go to Actions → "Promote to PROD" → Run workflow

3. Optionally check "dry run" first to preview

4. Approver accepts the environment gate

5. All components deploy to prod

6. Tag created, GitHub Release published with changelog

## Versioning: CalVer

Format: YYYY.MM.PATCH (e.g., 2026.04.1, 2026.04.2, 2026.05.1)

The "Create Release Tag" workflow auto-calculates the next version. Within the same month, patch increments. New month resets to 1. You can override with a specific version.

Tags are the source of truth for what's in production.