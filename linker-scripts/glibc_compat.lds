/*
 * glibc 2.38+ compatibility linker script for OxiMedia.
 *
 * The pre-compiled ONNX Runtime objects (libort_sys) reference ISO C23
 * aliases introduced in glibc 2.38:
 *
 *   __isoc23_strtol   -> strtol
 *   __isoc23_strtoll  -> strtoll
 *   __isoc23_strtoull -> strtoull
 *
 * On glibc 2.35 (our build system) these symbols do not exist.
 * PROVIDE() defines each alias only when it is not already defined —
 * so on glibc 2.38+ this script is a no-op.
 */
PROVIDE(__isoc23_strtol   = strtol);
PROVIDE(__isoc23_strtoll  = strtoll);
PROVIDE(__isoc23_strtoull = strtoull);
