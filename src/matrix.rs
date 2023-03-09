use num::traits::Zero;
use std::iter::Sum;
use std::ops::{Add, Index, Sub, Mul};
use core::array::from_fn;

use crate::vector::TVector;

#[derive(Clone,Copy,PartialEq,Debug)]
pub struct TMatrix<T, const M: usize, const N: usize> {
    pub data: [[T ; M] ; N],
}

impl<T: Copy> TMatrix<T, 2, 2> {
    pub fn new(e0: T, e1: T,
               e2: T, e3: T) -> Self {
        Self {
            data: [[e0, e2],
                   [e1, e3]]
        }
    }
}

impl<T: Copy> TMatrix<T, 3, 3> {
    pub fn new(e0: T, e1: T, e2: T,
               e3: T, e4: T, e5: T,
               e6: T, e7: T, e8: T) -> Self {
        Self {
            data: [[e0, e3, e6],
                   [e1, e4, e7],
                   [e2, e5, e8]]
        }
    }
}

impl<T: Copy> TMatrix<T, 4, 4> {
    pub fn new(e0: T, e1: T, e2: T, e3: T,
               e4: T, e5: T, e6: T, e7: T,
               e8: T, e9: T, e10: T, e11: T,
               e12: T, e13: T, e14: T, e15: T) -> Self {
        Self {
            data: [[e0, e4, e8, e12],
                   [e1, e5, e9, e13],
                   [e2, e6, e10, e14],
                   [e3, e7, e11, e15]]
        }
    }
}

fn element_wise<T: Copy,
                const M: usize,
                const N: usize,
                F: Fn(T, T) -> T>(a: TMatrix<T, M, N>,
                                  b: TMatrix<T, M, N>,
                                  f: F) -> TMatrix<T, M, N> {
    TMatrix::<T, M, N> {
        data: from_fn(|i| from_fn(|j| f(a.data[i][j], b.data[i][j])))
    }
}

/// Add implementation for matrices
impl<T: Add<Output = T> + Copy, const M: usize, const N: usize> Add for TMatrix<T, M, N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        element_wise(self, other, |a, b| a + b)
    }
}

/// Sub implementation for matrices
impl<T: Sub<Output = T> + Copy, const M: usize, const N: usize> Sub for TMatrix<T, M, N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        element_wise(self, other, |a, b| a - b)
    }
}

impl<T, const M: usize, const N: usize> Index<(usize, usize)> for TMatrix<T, M, N> {
    type Output = T;

    fn index(&self, (ind0, ind1): (usize, usize)) -> &Self::Output {
        &self.data[ind0][ind1]
    }
}


/// Square matrix multiplication
impl<T: Clone + Copy + Sum + Add<Output = T> + Mul<Output = T> + Zero, const N: usize> Mul<TVector<T, N>> for TMatrix<T, N, N> {
    type Output = TVector<T, N>;

    fn mul(self, other: TVector<T, N>) -> Self::Output {
        Self::Output {
            data: from_fn(|i| from_fn::<T, N, _>(|j| self.data[j][i] * other.data[j]).iter().cloned().sum())
        }
    }
}

/// Shorthands
pub type Matrix<const M: usize, const N: usize> = TMatrix<f32, M, N>;

pub type Mat2 = Matrix<2, 2>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn can_construct_mat2() {
        let m0 = Mat2::new(1.0, 2.0,
                           3.0, 4.0);

        assert_eq!(m0[(0, 0)], 1.0);
        assert_eq!(m0[(0, 1)], 3.0);
        assert_eq!(m0[(1, 0)], 2.0);
        assert_eq!(m0[(1, 1)], 4.0);
    }
}
