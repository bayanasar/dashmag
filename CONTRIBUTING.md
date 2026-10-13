# Contributing

## Say the reason; don't point at it

This repository is published as it stands, history included. Its issues and reviews
live on a tracker outside it, which a reader here cannot open.

So comments, commit messages and branch names carry their reason in their own words,
and point at nothing a reader of this repository cannot follow: no issue or review
numbers, no other repositories or their files, no machine names, no internal URLs.
Merge and squash messages are commit messages too, including any `Reviewed-on:` line a
forge adds. When the reason for a line of code was settled in a review, write the
reason down next to it — that is the part worth keeping.

History already published is not rewritten for this.

## Commit under a noreply address

Every commit publishes its author's and committer's addresses, so both are GitHub's
noreply address (`…@users.noreply.github.com`) — the one this repository is published
under — never a personal or machine one. A forge's own "keep my email private" address
does not qualify: it carries the forge's domain, which is one of the names above.

A forge writes new commits whenever a merge does anything but fast-forward — a merge
commit, a squash, a rebase that has to rewrite — and their committer is the merging
account. So fast-forward is the only merge style this repository allows. Making it the
default is not enough: a merge through the API names its own style.

CI checks every commit a pull request adds and every push to `main`. The pull-request
check cannot see a commit the forge creates at merge time; only the push check can, once
that commit is on `main` and already published. A green check on a pull request says the
branch is clean, not that merging it will be — preventing that is the merge-style
setting's job.
