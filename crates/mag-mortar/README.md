# mag-mortar

The placement library of a mag pool. A host (a person with a list of machines, Slurm, or a home
system) gives the pool its machines and takes them back; it never reads a graph. One of the pool's
machines, which the host names, coordinates: it runs a `Pool`, which places every job submitted to
the pool against the others. mag keeps no server of its own, and this crate does no I/O.

On every change the pool is told (a job arrives or ends, a device joins or leaves) the loop lists
the ways to place the jobs waiting, removes those that break a constraint (memory, a part's
tolerance and dtypes, a pin, a job's need, the devices its host's stamp allows, and the guard,
which keeps data kept at a level of trust off every member trusted below it), scores the rest
from the costs the devices offer, commits the best, and records every candidate and why any was
removed. A job arrives with the host's stamp, and its priority is held to the stamp's ceiling.

An author asks the pool two things before submitting: `describe`, the pool as one virtual device,
and `plan`, the record a graph's arrival would make, with the pool left as it is. The pool keeps
what it was told, with each record, as a ledger: the coordinator is to append each entry, and a pool
comes back from them by `Pool::replay`, which checks every record against the one logged. The ledger
names the version of this crate that began it; a ledger of version 1 replays to the records it
logged. With the `files` feature the ledger is written as JSON Lines, and the `mag` example plans a
graph on a pool described in a file:

    cargo run -p mag-mortar --features files --example mag -- plan \
        crates/mag-by/pools/example.json crates/mag-by/graphs/example.json

What mag covers and what it leaves to the host, on a list of machines, under Slurm and in a home
system, is set out in the workspace README: [What mag covers, and what it leaves to the
host](../../README.md#what-mag-covers-and-what-it-leaves-to-the-host). In short: you write each
part's function and the graph; the placement, the queue of jobs and the priority between them are
this crate's; which machines are in the pool is the host's.

This step runs on offers given as data: no member, network or device yet. It does not probe
candidates, count the time of edges between machines, divide a chain across devices or move a
running job; those come with the steps that measure them. Nor does it yet choose which device to
give up when the host takes one back by need, say which kind of device it most wants, or begin
from the arrangement its members hold after a coordinator stops.
