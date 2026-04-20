use oximedia_core::error_context::{ErrorContext, ErrorFrame};

#[test]
fn test_error_frame_display() {
    let frame = ErrorFrame {
        file: "src/lib.rs",
        line: 42,
        function: "my_func",
        message: std::borrow::Cow::Borrowed("something failed"),
    };
    let s = frame.to_string();
    assert!(s.contains("src/lib.rs"));
    assert!(s.contains("42"));
    assert!(s.contains("something failed"));
}

#[test]
fn test_error_chain_multi_frame() {
    let mut ctx = ErrorContext::new("comp", "op", "msg");
    ctx.push_frame(ErrorFrame {
        file: "a.rs",
        line: 1,
        function: "fn1",
        message: std::borrow::Cow::Borrowed("level 1"),
    });
    ctx.push_frame(ErrorFrame {
        file: "b.rs",
        line: 2,
        function: "fn2",
        message: std::borrow::Cow::Borrowed("level 2"),
    });
    ctx.push_frame(ErrorFrame {
        file: "c.rs",
        line: 3,
        function: "fn3",
        message: std::borrow::Cow::Borrowed("level 3"),
    });
    let display = ctx.frames_display();
    assert!(display.contains("fn1"), "frame 1 not in display");
    assert!(display.contains("fn2"), "frame 2 not in display");
    assert!(display.contains("fn3"), "frame 3 not in display");
}

#[test]
fn test_error_context_empty_display() {
    let ctx = ErrorContext::new("comp", "op", "msg");
    let s = ctx.frames_display();
    assert!(!s.is_empty());
}

#[test]
fn test_error_frame_function_field() {
    let frame = ErrorFrame {
        file: "src/codec.rs",
        line: 100,
        function: "decode_frame",
        message: std::borrow::Cow::Borrowed("bitstream error"),
    };
    let s = frame.to_string();
    assert!(s.contains("decode_frame"));
    assert!(s.contains("100"));
    assert!(s.contains("bitstream error"));
}

#[test]
fn test_with_frame_chaining() {
    let ctx = ErrorContext::new("muxer", "write", "disk full")
        .with_frame(ErrorFrame {
            file: "mux.rs",
            line: 55,
            function: "write_packet",
            message: std::borrow::Cow::Borrowed("I/O error"),
        })
        .with_frame(ErrorFrame {
            file: "io.rs",
            line: 10,
            function: "flush",
            message: std::borrow::Cow::Borrowed("disk full"),
        });
    assert_eq!(ctx.frame_count(), 2);
    let display = ctx.frames_display();
    assert!(display.contains("write_packet"));
    assert!(display.contains("flush"));
}

#[test]
fn test_error_frame_cow_owned() {
    let value = 99u32;
    let frame = ErrorFrame {
        file: "src/main.rs",
        line: 7,
        function: "run",
        message: std::borrow::Cow::Owned(format!("count was {value}")),
    };
    let s = frame.to_string();
    assert!(s.contains("99"));
}

#[test]
fn test_ctx_macro_borrowed() {
    let frame: ErrorFrame = oximedia_core::ctx!("literal message");
    let s = frame.to_string();
    assert!(s.contains("literal message"));
}

#[test]
fn test_ctx_macro_formatted() {
    let n = 42u32;
    let frame: ErrorFrame = oximedia_core::ctx!("value was {}", n);
    let s = frame.to_string();
    assert!(s.contains("42"));
}
