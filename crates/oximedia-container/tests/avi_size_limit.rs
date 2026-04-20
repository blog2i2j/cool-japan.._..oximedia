// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: AVI size-limit error type.

use oximedia_container::mux::avi::AviError;

#[test]
fn avi_size_limit_error() {
    // Verify AviError::FileTooLarge exists, formats correctly, and is the
    // right variant.  Actually producing >1 GiB is not feasible in a unit
    // test so we construct the error value directly.
    let err = AviError::FileTooLarge(2_000_000_000);
    let msg = err.to_string();
    assert!(
        msg.contains("1 GB"),
        "error message should mention '1 GB'; got: {msg}"
    );
    assert!(
        msg.contains("2000000000"),
        "error message should include the byte count; got: {msg}"
    );
}

#[test]
fn avi_unsupported_codec_error() {
    let err = AviError::UnsupportedCodec("H.264".to_owned());
    let msg = err.to_string();
    assert!(
        msg.contains("H.264"),
        "UnsupportedCodec error should include codec name; got: {msg}"
    );
}

#[test]
fn avi_error_is_debug() {
    let err = AviError::FileTooLarge(42);
    let dbg = format!("{err:?}");
    assert!(dbg.contains("FileTooLarge"));
}
