# mag-mortar

The placement library of a mag pool. Whatever hosts the pool runs a `Pool`: a home system's
control plane, which links this crate and runs the loop, or, on its own, the program that submits
the job. mag keeps no server of its own, and this crate does no I/O.

On every change the pool is told (a job arrives or ends, a device joins or leaves) the loop lists
the ways to place the jobs waiting, removes those that break a constraint (memory, a part's
tolerance and dtypes, a pin, a job's need, and the guard that keeps local-only data off remote
machines), scores the rest from the costs the devices offer, commits the best, and records every
candidate and why any was removed.

An author asks the pool two things before submitting: `describe`, the pool as one virtual
device, and `plan`, the record a graph's arrival would make, with the pool left as it is. The
pool keeps what it was told, with each record, as a ledger: its host appends each entry, and a
pool comes back from them by `Pool::replay`, which checks every record against the one logged.
With the `files` feature the ledger is written as JSON Lines, and the `mag` example plans a graph
on a pool described in a file:

    cargo run -p mag-mortar --features files --example mag -- plan \
        crates/mag-by/pools/example.json crates/mag-by/graphs/example.json

What mag covers and what it leaves to you, in a home system and on your own, is set out in the
workspace README: [What mag covers, and what it leaves to
you](../../README.md#what-mag-covers-and-what-it-leaves-to-you). In short: you write the graph and
the pool file; the placement is this crate's; who may use which machine is not mag's.

This step runs on offers given as data: no member, network or device yet. It does not probe
candidates, count the time of edges between machines, divide a chain across devices or move a
running job; those come with the steps that measure them.
