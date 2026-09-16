# Working agreement for this repo

## Git workflow: never push directly to `main`

`main` has GitHub branch protection (`required_pull_request_reviews` enabled,
`enforce_admins: true`) — a direct push is rejected server-side, including
from an admin token. This is deliberate, not a bug to route around.

When asked to commit changes:

1. Create a feature branch (`git checkout -b <descriptive-name>`).
2. Commit and push that branch (`git push -u origin <branch>`).
3. Open a PR (`gh pr create`) describing the change.
4. **Stop there.** Never run `gh pr merge` or push to `main` directly, even
   if asked to "commit and push" in a way that sounds like it means all the
   way to `main`. Merging is the user's decision, made by clicking Merge on
   GitHub after reviewing the diff themselves — that review-then-merge
   click is the approval step, not a formal "Approve" review (GitHub can't
   let the user approve their own PR anyway, since commits are pushed under
   their own account/token).
5. Tell the user the PR URL and stop. Don't merge unless explicitly told to
   (and even then, prefer that they click Merge on GitHub themselves).

If a task requires multiple related commits, they can go on the same
branch/PR — no need for one PR per commit.

## Verify against the live API and upstream repo, not memory or docstrings alone

Before filing an issue or writing a fix that touches USAspending API
behavior (a tool returning a wrong/misleading value, a missing filter or
category, an endpoint choice), check:

- **The live endpoint docs**: https://api.usaspending.gov/docs/endpoints —
  confirm a field/param/enum actually exists and behaves as assumed by
  hitting the real API, not by trusting this codebase's docstrings or your
  own training-data recall. Docstrings here can drift from the live
  contract; the API is the source of truth.
- **The upstream contract/source**:
  https://github.com/fedspendingtransparency/usaspending-api/tree/master/usaspending_api
  (the `api_contracts/contracts/` subtree has per-endpoint field tables) —
  useful for confirming an enum's exact members or a field's real meaning
  when the live response alone is ambiguous.
- **Upstream's own issue tracker**
  (fedspendingtransparency/usaspending-api, not this repo) before assuming
  a weird API behavior is unreported or fixable — several "bugs" turn out
  to be long-standing, deliberate design choices the USAspending team has
  already responded to (e.g. #118's overlap-vs-action-date behavior,
  confirmed via fedspendingtransparency/usaspending-api#1707, open since
  2019). Citing that context in an issue/PR write-up saves someone
  re-discovering it later and sets expectations correctly (caveat vs.
  fixable bug vs. wait-for-upstream).

## Adding or modifying tools

Before starting tool work, see [docs/adding-a-tool.md](docs/adding-a-tool.md) for
the exact registration flow (four files, in order) and the chart/citation step
that's easy to miss. Same checklist applies whether you're adding a new tool or
modifying filter parameters on an existing one.

The shared filter parameter list is now defined once in `SpendingFilterParams`
TypedDict (in `backend/app/agent/tool_filters.py`) — reference that when adding
new filters, not the scattered function signatures.

## Comments: default to none

One line, max, and only when the *why* isn't obvious from the code itself.
No narration ("found live on...", "confirmed that...", "verified against...",
"real-world example: ...") — that belongs in the commit message or PR
description, not the file. If a comment needs more than one line to make
its point, it's doing the commit message's job — cut it or move it there.
