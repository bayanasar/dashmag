# The coordinator of the list-of-machines recipe, planning only: it builds `mag` and plans the
# example graph on the hand-written pool file. Build from the repository's root:
#
#   docker build -f examples/machine-list/Dockerfile -t mag-machine-list .
#   docker run --rm mag-machine-list

FROM rust:1.97.1-slim-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --locked --release -p mag-mortar --features files --example mag

FROM debian:bookworm-slim
COPY --from=build /src/target/release/examples/mag /usr/local/bin/mag
COPY examples/machine-list/pool.json /etc/mag/pool.json
COPY crates/mag-by/graphs/example.json /etc/mag/graph.json
ENTRYPOINT ["mag"]
CMD ["plan", "/etc/mag/pool.json", "/etc/mag/graph.json"]
