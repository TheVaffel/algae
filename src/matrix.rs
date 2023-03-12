use num::traits::Zero;
use std::iter::Sum;
use std::ops::{Add, Index, Sub, Mul};
use core::array::from_fn;

use crate::vector::{TVector, Vec2};

/// M * N matrix with elements of type T. Stored column-major
#[derive(Clone,Copy,PartialEq,Debug)]
pub struct TMatrix<T, const M: usize, const N: usize> {
    pub data: [[T ; M] ; N],
}

/// Constructors

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

impl<T: Copy, const M: usize, const N: usize> TMatrix<T, M, N> {
    pub fn from_array(arr: &[T]) -> Self {
        Self {
            data: from_fn(|i| from_fn(|j| {
                let ind = j * N + i;
                arr[ind]
            }))
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


/// Matrix-vector multiplication
impl<T: Clone + Copy + Sum + Add<Output = T> + Mul<Output = T> + Zero, const M: usize, const N: usize> Mul<TVector<T, N>> for TMatrix<T, M, N> {
    type Output = TVector<T, M>;

    fn mul(self, other: TVector<T, N>) -> Self::Output {
        Self::Output {
            data: from_fn(|i| from_fn::<T, N, _>(|j| self.data[j][i] * other.data[j]).iter().cloned().sum())
        }
    }
}

/// Matrix-matrix multiplication
impl<T: Clone + Copy + Sum + Add<Output = T> + Mul<Output = T> + Zero,
     const L: usize,
     const M: usize,
     const N: usize>
    Mul<TMatrix<T, M, N>> for TMatrix<T, L, M> {
        type Output = TMatrix<T, L, N>;
        fn mul(self, other: TMatrix<T, M, N>) -> Self::Output {
            Self::Output {
                data: from_fn(
                    |i| from_fn::<T, L, _>(
                        |j| from_fn::<T, M, _>(
                            |k| self.data[k][j] * other.data[i][k]
                        ).iter().cloned().sum()
                    )
                )
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

    #[test]
    fn can_construct_mat2_from_array() {
        let m0 = Mat2::from_array(&[1.0, 2.0,
                                    3.0, 4.0]);

        assert_eq!(m0[(0, 0)], 1.0);
        assert_eq!(m0[(0, 1)], 3.0);
        assert_eq!(m0[(1, 0)], 2.0);
        assert_eq!(m0[(1, 1)], 4.0);
    }

    #[test]
    fn can_multiply_vec2_by_mat2() {
        let m0 = Mat2::new(1.0, 2.0,
                           3.0, 4.0);
        let v0 = Vec2::new(5.0, 6.0);

        let v1 = m0 * v0;

        assert_eq!(v1[0], 17.0);
        assert_eq!(v1[1], 39.0);
    }

    #[test]
    fn can_multiply_vec2_by_mat32() {

        let m0 = Matrix::<3, 2>::from_array(&[1.0, 2.0,
                                              3.0, 4.0,
                                              5.0, 6.0]);
        let v0 = Vec2::new(7.0, 8.0);

        let v1 = m0 * v0;

        assert_eq!(v1[0], 7.0 + 16.0);
        assert_eq!(v1[1], 21.0 + 32.0);
        assert_eq!(v1[2], 35.0 + 48.0);
    }

    #[test]
    fn can_multiply_mat2_by_mat2() {
        let m0 = Mat2::new(1.0, 2.0,
                           3.0, 4.0);
        let m1 = Mat2::new(5.0, 6.0,
                           7.0, 8.0);

        let m2 = m0 * m1;

        assert_eq!(m2[(0, 0)], 5.0 + 14.0);
        assert_eq!(m2[(0, 1)], 15.0 + 28.0);
        assert_eq!(m2[(1, 0)], 6.0 + 16.0);
        assert_eq!(m2[(1, 1)], 18.0 + 32.0);
    }

    #[test]
    fn can_multiply_mat23_by_mat32() {

        let m0 = Matrix::<2, 3>::from_array(&[1.0, 2.0, 3.0,
                                              4.0, 5.0, 6.0]);
        let m1 = Matrix::<3, 2>::from_array(&[7.0, 8.0,
                                              9.0, 10.0,
                                              11.0, 12.0]);
        let m2 = m0 * m1;

        assert_eq!(m2[(0, 0)], 7.0 + 18.0 + 33.0);
        assert_eq!(m2[(0, 1)], 28.0 + 45.0 + 66.0);
        assert_eq!(m2[(1, 0)], 8.0 + 20.0 + 36.0);
        assert_eq!(m2[(1, 1)], 32.0 + 50.0 + 72.0);
    }
}
