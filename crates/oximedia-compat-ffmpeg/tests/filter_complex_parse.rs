//! Integration tests for the `-filter_complex` recursive-descent parser.

use oximedia_compat_ffmpeg::filter_complex::{FilterComplexError, FilterGraph};

#[test]
fn test_simple_scale_with_pads() {
    let g = FilterGraph::parse("[in]scale=1280:720[out]").expect("parse");
    assert_eq!(g.chains.len(), 1);
    let chain = &g.chains[0];
    assert_eq!(chain.input_pads, vec!["in"]);
    assert_eq!(chain.output_pads, vec!["out"]);
    assert_eq!(chain.filters.len(), 1);
    assert_eq!(chain.filters[0].name, "scale");
}

#[test]
fn test_two_chain_graph() {
    let g = FilterGraph::parse("[0:v]scale=1920:1080[bg];[bg][1:v]overlay=x=10:y=10[out]")
        .expect("parse two chains");
    assert_eq!(g.chains.len(), 2);
    assert_eq!(g.chains[0].output_pads, vec!["bg"]);
    assert_eq!(g.chains[1].input_pads, vec!["bg", "1:v"]);
    assert_eq!(g.chains[1].output_pads, vec!["out"]);
}

#[test]
fn test_filter_no_pads() {
    let g = FilterGraph::parse("scale=1280:720").expect("parse no pads");
    assert_eq!(g.chains.len(), 1);
    assert!(g.chains[0].input_pads.is_empty());
    assert!(g.chains[0].output_pads.is_empty());
    assert_eq!(g.chains[0].filters[0].name, "scale");
}

#[test]
fn test_named_options() {
    let g = FilterGraph::parse("overlay=x=10:y=20").expect("parse overlay");
    let filter = &g.chains[0].filters[0];
    assert_eq!(filter.name, "overlay");
    assert!(
        filter
            .options
            .iter()
            .any(|o| o.key.as_deref() == Some("x") && o.value == "10"),
        "x=10 option not found"
    );
    assert!(
        filter
            .options
            .iter()
            .any(|o| o.key.as_deref() == Some("y") && o.value == "20"),
        "y=20 option not found"
    );
}

#[test]
fn test_multi_filter_in_chain() {
    let g = FilterGraph::parse("[in]scale=1280:720,unsharp=5:5:1.0[out]").expect("multi filter");
    assert_eq!(g.chains[0].filters.len(), 2);
    assert_eq!(g.chains[0].filters[0].name, "scale");
    assert_eq!(g.chains[0].filters[1].name, "unsharp");
}

#[test]
fn test_amix_two_inputs() {
    let g = FilterGraph::parse("[0:a][1:a]amix=inputs=2:duration=first[aout]").expect("parse amix");
    assert_eq!(g.chains[0].input_pads, vec!["0:a", "1:a"]);
    assert_eq!(g.chains[0].output_pads, vec!["aout"]);
    assert_eq!(g.chains[0].filters[0].name, "amix");
}

#[test]
fn test_filter_count() {
    let g = FilterGraph::parse("[0:v]scale=640:360[s1];[s1]unsharp[out]").expect("parse");
    assert_eq!(g.filter_count(), 2);
}

#[test]
fn test_output_labels() {
    let g = FilterGraph::parse("[0:v]scale=640:360[v1];[0:a]aformat=fltp[a1]").expect("parse");
    let labels = g.output_labels();
    assert!(labels.contains(&"v1"), "missing v1");
    assert!(labels.contains(&"a1"), "missing a1");
}

#[test]
fn test_empty_error() {
    assert!(matches!(
        FilterGraph::parse(""),
        Err(FilterComplexError::Empty)
    ));
    assert!(matches!(
        FilterGraph::parse("   "),
        Err(FilterComplexError::Empty)
    ));
}

#[test]
fn test_display_roundtrip() {
    let inputs = vec!["[in]scale=1280:720[out]", "scale=1280:720"];
    for input in inputs {
        let g = FilterGraph::parse(input).expect("parse");
        let s = g.to_string();
        let g2 = FilterGraph::parse(&s).expect("re-parse");
        assert_eq!(
            g.chains.len(),
            g2.chains.len(),
            "roundtrip chain count mismatch for '{}'",
            input
        );
    }
}
