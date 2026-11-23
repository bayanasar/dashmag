//! Reads a graph file and prints it as Graphviz DOT, or says why the file is refused.
//!
//! ```text
//! cargo run -p mag-by --features graph --example dot -- crates/mag-by/graphs/example.json
//! cargo run -p mag-by --features graph --example dot -- your-graph.json | dot -Tsvg > graph.svg
//! ```

use mag_by::{Graph, ReadError};

/// The file's text as DOT, or why it is refused.
pub fn run(text: &str) -> Result<String, ReadError> {
    Ok(Graph::from_json(text)?.to_dot())
}

#[allow(dead_code)]
fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: dot GRAPH.json");
        std::process::exit(2);
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(2);
        }
    };
    match run(&text) {
        Ok(dot) => print!("{dot}"),
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    }
}
