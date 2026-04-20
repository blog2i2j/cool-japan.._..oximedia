// Copyright 2017 Brian Langenberger
// Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
//
// Licensed under the Apache License, Version 2.0 or the MIT license,
// at your option. See the LICENSE-APACHE / LICENSE-MIT files for details.

//! Traits and implementations for writing bits to a stream.
//!
//! Split across multiple files during the 0.1.4 refactor so that no
//! single source exceeds the COOLJAPAN 2 000-line guideline.

#![warn(missing_docs)]

use super::{
    BitCount, Checkable, CheckedSigned, CheckedUnsigned, Endianness, Integer, Numeric, PhantomData,
    Primitive, SignedBitCount, SignedInteger, UnsignedInteger, VBRInteger,
};

#[cfg(feature = "alloc")]
pub use bit_recorder::BitRecorder;

#[cfg(feature = "alloc")]
mod bit_recorder;
mod bit_write;
mod bit_write2;
mod bit_writer;
mod byte_writer;
mod counter;
mod stream_traits;

pub use bit_write::BitWrite;
pub use bit_write2::BitWrite2;
pub use bit_writer::BitWriter;
pub use byte_writer::{ByteWrite, ByteWriter};
#[allow(deprecated)]
pub use counter::BitCounter;
pub use counter::{BitsWritten, Counter, Overflowed};
pub use stream_traits::{
    ToBitStream, ToBitStreamUsing, ToBitStreamWith, ToByteStream, ToByteStreamUsing,
    ToByteStreamWith,
};
