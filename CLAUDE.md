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

## Comments: default to none

One line, max, and only when the *why* isn't obvious from the code itself.
No narration ("found live on...", "confirmed that...", "verified against...",
"real-world example: ...") — that belongs in the commit message or PR
description, not the file. If a comment needs more than one line to make
its point, it's doing the commit message's job — cut it or move it there.
