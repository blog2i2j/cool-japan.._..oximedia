//! Star rating system for clips.

use crate::error::{ClipError, ClipResult};
use serde::{Deserialize, Serialize};

/// Star rating for clips (1-5 stars).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum Rating {
    /// No rating assigned.
    #[default]
    Unrated,
    /// One star (poor).
    OneStar,
    /// Two stars (fair).
    TwoStars,
    /// Three stars (good).
    ThreeStars,
    /// Four stars (very good).
    FourStars,
    /// Five stars (excellent).
    FiveStars,
}

impl Rating {
    /// Creates a rating from a numeric value (1-5).
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not in the range 1-5.
    pub fn from_value(value: u8) -> ClipResult<Self> {
        match value {
            1 => Ok(Self::OneStar),
            2 => Ok(Self::TwoStars),
            3 => Ok(Self::ThreeStars),
            4 => Ok(Self::FourStars),
            5 => Ok(Self::FiveStars),
            _ => Err(ClipError::InvalidRating(i32::from(value))),
        }
    }

    /// Returns the numeric value of the rating (0 for unrated, 1-5 for stars).
    #[must_use]
    pub const fn to_value(&self) -> u8 {
        match self {
            Self::Unrated => 0,
            Self::OneStar => 1,
            Self::TwoStars => 2,
            Self::ThreeStars => 3,
            Self::FourStars => 4,
            Self::FiveStars => 5,
        }
    }

    /// Returns whether this is a rated clip.
    #[must_use]
    pub const fn is_rated(&self) -> bool {
        !matches!(self, Self::Unrated)
    }

    /// Returns all possible ratings.
    #[must_use]
    pub const fn all() -> [Self; 6] {
        [
            Self::Unrated,
            Self::OneStar,
            Self::TwoStars,
            Self::ThreeStars,
            Self::FourStars,
            Self::FiveStars,
        ]
    }
}

impl std::fmt::Display for Rating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unrated => write!(f, "Unrated"),
            Self::OneStar => write!(f, "★"),
            Self::TwoStars => write!(f, "★★"),
            Self::ThreeStars => write!(f, "★★★"),
            Self::FourStars => write!(f, "★★★★"),
            Self::FiveStars => write!(f, "★★★★★"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rating_from_value() {
        assert_eq!(
            Rating::from_value(1).expect("from_value should succeed"),
            Rating::OneStar
        );
        assert_eq!(
            Rating::from_value(5).expect("from_value should succeed"),
            Rating::FiveStars
        );
        assert!(Rating::from_value(0).is_err());
        assert!(Rating::from_value(6).is_err());
    }

    #[test]
    fn test_rating_to_value() {
        assert_eq!(Rating::Unrated.to_value(), 0);
        assert_eq!(Rating::OneStar.to_value(), 1);
        assert_eq!(Rating::FiveStars.to_value(), 5);
    }

    #[test]
    fn test_rating_ordering() {
        assert!(Rating::OneStar < Rating::TwoStars);
        assert!(Rating::FourStars < Rating::FiveStars);
        assert!(Rating::Unrated < Rating::OneStar);
    }
}
