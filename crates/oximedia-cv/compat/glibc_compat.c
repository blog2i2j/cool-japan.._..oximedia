/*
 * glibc 2.38+ compatibility shim for ONNX Runtime
 *
 * The pre-compiled ONNX Runtime binary (libonnxruntime.a) was compiled
 * against glibc 2.38+, which introduces ISO C23 aliases for the standard
 * string-to-integer conversion functions:
 *
 *   __isoc23_strtol   -> strtol  (ISO C23 edition of strtol)
 *   __isoc23_strtoll  -> strtoll (ISO C23 edition of strtoll)
 *   __isoc23_strtoull -> strtoull (ISO C23 edition of strtoull)
 *
 * In glibc 2.38+, when compiling with -std=c23 or newer, the macros
 * strtol/strtoll/strtoull expand to their __isoc23_* counterparts.
 * On glibc 2.35 these symbols do not exist.
 *
 * This shim provides the missing symbols by delegating to the standard
 * glibc 2.35 equivalents, which have exactly the same semantics.
 */

#include <stdlib.h>

/*
 * __isoc23_strtol: ISO C23 strtol - identical behavior to POSIX strtol.
 * Converts str to long, using given base, storing end pointer in endptr.
 */
long __isoc23_strtol(const char *restrict nptr,
                     char **restrict endptr,
                     int base)
{
    return strtol(nptr, endptr, base);
}

/*
 * __isoc23_strtoll: ISO C23 strtoll - identical behavior to POSIX strtoll.
 * Converts str to long long, using given base, storing end pointer in endptr.
 */
long long __isoc23_strtoll(const char *restrict nptr,
                           char **restrict endptr,
                           int base)
{
    return strtoll(nptr, endptr, base);
}

/*
 * __isoc23_strtoull: ISO C23 strtoull - identical behavior to POSIX strtoull.
 * Converts str to unsigned long long, using given base, storing end pointer.
 */
unsigned long long __isoc23_strtoull(const char *restrict nptr,
                                     char **restrict endptr,
                                     int base)
{
    return strtoull(nptr, endptr, base);
}
