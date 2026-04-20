use std::path::PathBuf;

use crate::diagnostics::TranslationError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PassPhase {
    First { stats_path: PathBuf },
    Second { stats_path: PathBuf },
}

pub fn parse_pass(args: &[String]) -> Result<Option<PassPhase>, TranslationError> {
    let mut phase: Option<&str> = None;
    let mut stats_path: Option<PathBuf> = None;
    let mut i = 0usize;

    while i < args.len() {
        match args[i].as_str() {
            "-pass" => {
                i += 1;
                if i >= args.len() {
                    return Err(TranslationError::ParseError(
                        "flag '-pass' requires an argument".to_string(),
                    ));
                }
                phase = Some(args[i].as_str());
            }
            "-passlogfile" => {
                i += 1;
                if i >= args.len() {
                    return Err(TranslationError::ParseError(
                        "flag '-passlogfile' requires an argument".to_string(),
                    ));
                }
                stats_path = Some(PathBuf::from(&args[i]));
            }
            _ => {}
        }
        i += 1;
    }

    let Some(phase) = phase else {
        return Ok(None);
    };

    let stats_path = stats_path.unwrap_or_else(|| PathBuf::from("ffmpeg2pass-0.log"));
    match phase {
        "1" => Ok(Some(PassPhase::First { stats_path })),
        "2" => Ok(Some(PassPhase::Second { stats_path })),
        _ => Err(TranslationError::ParseError(format!(
            "invalid -pass value '{}': must be 1 or 2",
            phase
        ))),
    }
}

pub(crate) fn phase_from_parts(
    pass: Option<u8>,
    passlogfile: Option<&str>,
) -> Result<Option<PassPhase>, TranslationError> {
    let mut args = Vec::new();

    if let Some(pass) = pass {
        args.push("-pass".to_string());
        args.push(pass.to_string());
    }

    if let Some(passlogfile) = passlogfile {
        args.push("-passlogfile".to_string());
        args.push(passlogfile.to_string());
    }

    parse_pass(&args)
}
