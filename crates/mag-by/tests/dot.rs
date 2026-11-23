//! The graph as Graphviz DOT.

use std::path::Path;

use mag_by::Graph;

const EXAMPLE: &str = include_str!("../graphs/example.json");

/// Fails when the example's DOT is not the kept one. `MAG_BY_WRITE_DOT=1` writes it.
#[test]
fn the_examples_dot_is_the_kept_one() {
    let dot = Graph::from_json(EXAMPLE).unwrap().to_dot();
    let kept = Path::new(env!("CARGO_MANIFEST_DIR")).join("graphs/example.dot");
    if std::env::var_os("MAG_BY_WRITE_DOT").is_some() {
        std::fs::write(&kept, &dot).unwrap();
    }
    assert!(
        std::fs::read_to_string(kept).unwrap_or_default() == dot,
        "graphs/example.dot differs; run the tests with MAG_BY_WRITE_DOT=1 to write it"
    );
}

#[test]
fn every_input_part_and_edge_is_drawn() {
    let g = Graph::from_json(EXAMPLE).unwrap();
    let dot = g.to_dot();
    for name in g
        .inputs
        .iter()
        .map(|i| &i.name)
        .chain(g.parts.iter().map(|p| &p.name))
    {
        assert!(dot.contains(&format!("\"{name}\" [")), "{name} has no node");
    }
    assert_eq!(dot.matches(" -> ").count(), g.edges.len());
    // The released count is dashed and not red; the frames' edge is red; sizes in binary units.
    assert!(
        dot.contains(
            "\"count\" -> \"job's results\" [label=\"i64 64 B\\nreleased\", style=dashed];"
        )
    );
    assert!(dot.contains("\"frames\" -> \"decode\" [label=\"bytes 1.12 GiB\", color=red];"));
    for size in [
        "4 KiB", "156 KiB", "18.4 MiB", "287 MiB", "1.12 GiB", "64 B",
    ] {
        assert!(dot.contains(size), "{size}");
    }
    assert!(dot.contains("rankdir=TB;"));
}

#[test]
fn names_are_quoted() {
    let mut g = Graph::from_json(EXAMPLE).unwrap();
    g.job.name = "a \"quoted\" \\ name".into();
    assert!(
        g.to_dot()
            .starts_with("digraph \"a \\\"quoted\\\" \\\\ name\" {")
    );
}

#[path = "../examples/dot.rs"]
mod example;

/// What the README's command prints for the example file: `graphs/example.dot`, byte for byte.
#[test]
fn the_dot_example_prints_the_kept_dot() {
    let kept = Path::new(env!("CARGO_MANIFEST_DIR")).join("graphs/example.dot");
    assert_eq!(
        example::run(EXAMPLE).unwrap(),
        std::fs::read_to_string(kept).unwrap()
    );
    let refused = example::run(&EXAMPLE.replace("\"mag-graph\"", "\"other\"")).unwrap_err();
    assert!(refused.to_string().contains("other"), "{refused}");
}
