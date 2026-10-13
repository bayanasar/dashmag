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

Every commit publishes its author's and committer's addresses, so both are a noreply
address (`…@users.noreply.github.com`), never a personal or machine one. A merge commit
made by a forge carries the merging account's address: merge fast-forward, or make sure
that account's primary address is its noreply one. CI checks every commit a pull request
adds and every push to `main`.
