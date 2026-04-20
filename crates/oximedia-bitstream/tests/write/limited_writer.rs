// Copyright 2017 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;

pub(super) struct LimitedWriter {
    pub(super) can_write: usize,
}

impl LimitedWriter {
    pub(super) fn new(max_bytes: usize) -> LimitedWriter {
        LimitedWriter {
            can_write: max_bytes,
        }
    }
}

impl io::Write for LimitedWriter {
    fn write(&mut self, buf: &[u8]) -> Result<usize, io::Error> {
        let to_write = buf.len().min(self.can_write);
        self.can_write -= to_write;
        Ok(to_write)
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        Ok(())
    }
}
