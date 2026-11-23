# mag-mortar

The controller of a mag pool: the placement loop. On every event (a job arrives or ends, a device
joins or leaves) it lists the ways to place the jobs the event touches, removes those that break
a constraint (memory, a part's tolerance and dtypes, a pin, and the guard that keeps local-only
data off remote machines), scores the rest from the costs the devices offer, commits the best, and
records every candidate and why any was removed.

This first step runs on offers given as data: no member, network or device yet. It does not probe
candidates, count the time of edges between machines, divide a chain across devices or move a
running job; those come with the steps that measure them.
