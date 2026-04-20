// Copyright 2017 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The library crate is named `oximedia_bitstream`; use the alias that
// all downstream test code expects so files need not be rewritten.
extern crate oximedia_bitstream as bitstream_io;

#[path = "write/limited_writer.rs"]
mod limited_writer;

#[path = "write/writer_be_le.rs"]
mod writer_be_le;

#[path = "write/writer_errors_counter.rs"]
mod writer_errors_counter;

#[path = "write/recorder_misc.rs"]
mod recorder_misc;
