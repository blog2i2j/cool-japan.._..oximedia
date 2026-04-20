use crate::diagnostics::TranslationError;
use crate::filter_complex::{Filter, FilterChain, FilterGraph, FilterOption};

pub fn parse_vf(vf_str: &str) -> Result<FilterGraph, TranslationError> {
    parse_filter_shorthand(vf_str)
}

pub fn parse_af(af_str: &str) -> Result<FilterGraph, TranslationError> {
    parse_filter_shorthand(af_str)
}

fn parse_filter_shorthand(input: &str) -> Result<FilterGraph, TranslationError> {
    let mut filters = Vec::new();

    for entry in input.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            return Err(TranslationError::FilterParseError(
                "empty filter shorthand entry".to_string(),
            ));
        }

        filters.push(parse_filter_entry(entry)?);
    }

    if filters.is_empty() {
        return Err(TranslationError::FilterParseError(
            "empty filter shorthand".to_string(),
        ));
    }

    Ok(FilterGraph {
        chains: vec![FilterChain {
            input_pads: vec!["in".to_string()],
            filters,
            output_pads: vec!["out".to_string()],
        }],
    })
}

fn parse_filter_entry(entry: &str) -> Result<Filter, TranslationError> {
    let (name, raw_options) = match entry.split_once('=') {
        Some((name, options)) => (name.trim(), Some(options.trim())),
        None => (entry.trim(), None),
    };

    if name.is_empty() {
        return Err(TranslationError::FilterParseError(format!(
            "missing filter name in '{}'",
            entry
        )));
    }

    let options = match raw_options {
        Some(options) if !options.is_empty() => parse_options(options),
        Some(_) => {
            return Err(TranslationError::FilterParseError(format!(
                "missing filter options in '{}'",
                entry
            )));
        }
        None => Vec::new(),
    };

    Ok(Filter::with_options(name.to_string(), options))
}

fn parse_options(options: &str) -> Vec<FilterOption> {
    options
        .split(':')
        .filter(|segment| !segment.is_empty())
        .map(|segment| match segment.split_once('=') {
            Some((key, value)) => FilterOption::named(key.trim(), value.trim()),
            None => FilterOption::positional(segment.trim()),
        })
        .collect()
}
