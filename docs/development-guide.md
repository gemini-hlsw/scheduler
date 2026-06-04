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

## Working with `gpp-client` development versions

`gpp-client` ships two parallel tracks on PyPI: stable releases (used by the GPP PRODUCTION environment) and `.devN` pre-releases (used by the GPP DEVELOPMENT environment). They are managed via two conflicting dependency groups in [backend/pyproject.toml](https://github.com/gemini-hlsw/scheduler/blob/main/backend/pyproject.toml):

- `gpp-prod = ["gpp-client>=X.Y.Z"]` — latest stable, used by CI and the Docker build.
- `gpp-dev = ["gpp-client>=X.Y.Z.dev0,<X.Y.Z.a0"]` — restricts the resolver to `.devN` pre-releases of the current dev cycle `X.Y.Z`.

You must pass `--group` explicitly; running plain `uv sync` from the workspace root does not pick these groups up:

```sh
# stable (matches CI/Docker)
uv sync --group gpp-prod --no-group gpp-dev

# switch local venv to the latest .devN of the current cycle
uv sync --group gpp-dev --no-group gpp-prod

# pull a newer .devN later
uv sync --group gpp-dev --no-group gpp-prod --upgrade-package gpp-client
```

When the gpp-client team rolls a new dev cycle (e.g. `X.Y.Z` finalizes and `.devN` starts being published for the next version `X.Y.(Z+1)` or `X.(Y+1).0`), bump **both** numbers in the `gpp-dev` specifier together — `>=X.Y.Z.dev0,<X.Y.Za0`. If they diverge the range is empty and `uv sync` will fail.

### Which version is used by each deployment

The [backend/Dockerfile](https://github.com/gemini-hlsw/scheduler/blob/main/backend/Dockerfile) accepts a `GPP_GROUP` build-arg that picks the track at image-build time:

| Deployment | Workflow | `GPP_GROUP` | gpp-client track |
| --- | --- | --- | --- |
| Auto-deploy → DEV | [.github/workflows/deploy-dev.yml](https://github.com/gemini-hlsw/scheduler/blob/main/.github/workflows/deploy-dev.yml) | `gpp-dev` | latest `.devN` of the current cycle (talks to GPP DEVELOPMENT) |
| Promote → PROD | [.github/workflows/promote-prod.yml](https://github.com/gemini-hlsw/scheduler/blob/main/.github/workflows/promote-prod.yml) | `gpp-prod` | latest stable (talks to GPP PRODUCTION) |

The arg is passed via `heroku container:push --arg GPP_GROUP=…`. If you build the image locally without the arg, it defaults to `gpp-prod`.

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