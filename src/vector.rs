use std::ops::{Add, Index, Sub, Mul, Div};
use core::array::from_fn;
use num::traits::Zero;

#[derive(Clone,Copy,PartialEq,Debug)]
pub struct TVector<T, const N: usize> {
    pub data: [T ; N],
}
/// Constructors
impl<T: Copy> TVector<T, 2> {
    pub fn new(e0: T, e1: T) -> Self{
        TVector::<T, 2> {
            data: [e0, e1]
        }
    }
}

impl<T: Copy> TVector<T, 3> {
    pub fn new(e0: T, e1: T, e2: T) -> Self {
        TVector::<T, 3> {
            data: [e0, e1, e2]
        }
    }
}

impl<T: Copy> TVector<T, 4> {
    pub fn new(e0: T, e1: T, e2: T, e3: T) -> Self {
        TVector::<T, 4> {
            data: [e0, e1, e2, e3]
        }
    }
}

fn element_wise<T: Copy,
                const N: usize,
                F: Fn(T, T) -> T>(a: TVector<T, N>,
                                  b: TVector<T, N>,
                                  f: F) -> TVector<T, N> {
    TVector::<T, N> {
        data: from_fn(|i| f(a.data[i], b.data[i]))
    }
}

impl<T: Zero + Copy, const N: usize> Zero for TVector<T, N> {
    fn zero() -> Self {
        Self {
            data: [T::zero(); N]
        }
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|a| a.is_zero())
    }
}

/// Add implementation for vectors
impl<T: Add<Output = T> + Copy, const N: usize> Add for TVector<T, N> {
    type Output = TVector<T, N>;

    fn add(self, other: Self) -> Self {
        element_wise(self, other, |a, b| a + b)
    }
}

/// Sub implementation for vectors
impl<T: Sub<Output = T> + Copy, const N: usize> Sub for TVector<T, N> {
    type Output = TVector<T, N>;

    fn sub(self, other: Self) -> Self {
        element_wise(self, other, |a, b| a - b)
    }
}


/// Mul implementation for vectors (element-wise multiplication)
impl<T: Mul<Output = T> + Copy, const N: usize> Mul for TVector<T, N> {
    type Output = TVector<T, N>;

    fn mul(self, other: Self) -> Self {
        element_wise(self, other, |a, b| a * b)
    }
}

/// Div implementation for vectors (element-wise division)
impl<T: Div<Output = T> + Copy, const N: usize> Div for TVector<T, N> {
    type Output = TVector<T, N>;

    fn div(self, other: Self) -> Self {
        element_wise(self, other, |a, b| a / b)
    }
}

/// Index implementation for vectors
impl<T, const N: usize> Index<usize> for TVector<T, N> {
    type Output = T;

    fn index(&self, ind: usize) -> &T {
        &self.data[ind]
    }
}


/// Shorthands
pub type Vector<const N: usize> = TVector<f32, N>;
pub type DVector<const N: usize> = TVector<f64, N>;

pub type Vec2 = Vector<2>;
pub type Vec3 = Vector<3>;
pub type Vec4 = Vector<4>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create_zero_vectors() {
        let z2 = Vec2::zero();
        let z3 = Vec3::zero();
        let z4 = Vec4::zero();

        (0..2).for_each(|i| assert_eq!(z2[i], 0.0));
        (0..3).for_each(|i| assert_eq!(z3[i], 0.0));
        (0..4).for_each(|i| assert_eq!(z4[i], 0.0));
    }

    #[test]
    fn can_initialize_vectors_with_values() {
        let v2 = Vec2::new(1.0, 2.0);
        let v3 = Vec3::new(4.0, 3.0, 1.0);
        let v4 = Vec4::new(5.0, 2.0, 1.0, 3.0);

        assert_eq!(v2[0], 1.0);
        assert_eq!(v2[1], 2.0);

        assert_eq!(v3[0], 4.0);
        assert_eq!(v3[1], 3.0);
        assert_eq!(v3[2], 1.0);

        assert_eq!(v4[0], 5.0);
        assert_eq!(v4[1], 2.0);
        assert_eq!(v4[2], 1.0);
        assert_eq!(v4[3], 3.0);
    }

    #[test]
    fn can_add_basic_vectors() {
        let v20 = Vec2::new(1.0, 2.0);
        let v21 = Vec2::new(4.0, 5.0);

        let v30 = Vec3::new(1.0, 4.0, 5.0);
        let v31 = Vec3::new(4.0, 5.0, 6.0);

        let v40 = Vec4::new(3.0, 4.0, 1.0, 3.0);
        let v41 = Vec4::new(1.0, 4.0, 5.0, 1.0);

        let res2 = v20 + v21;
        let res3 = v30 + v31;
        let res4 = v40 + v41;

        (0..2).for_each(|i| assert_eq!(res2[i], v20[i] + v21[i]));
        (0..3).for_each(|i| assert_eq!(res3[i], v30[i] + v31[i]));
        (0..4).for_each(|i| assert_eq!(res4[i], v40[i] + v41[i]));
    }

    #[test]
    fn can_sub_v3() {
        let v0 = Vec3::new(1.0, 3.0, 4.0);
        let v1 = Vec3::new(3.0, 2.0, 1.0);

        let v2 = v0 - v1;

        assert_eq!(v2[0], -2.0);
        assert_eq!(v2[1], 1.0);
        assert_eq!(v2[2], 3.0);
    }

    #[test]
    fn can_mul_v3() {
        let v0 = Vec3::new(1.0, 3.0, 4.0);
        let v1 = Vec3::new(3.0, 2.0, 1.0);

        let v2 = v0 * v1;

        assert_eq!(v2[0], 3.0);
        assert_eq!(v2[1], 6.0);
        assert_eq!(v2[2], 4.0);
    }

    #[test]
    fn can_div_v3() {
        let v0 = Vec3::new(1.0, 3.0, 4.0);
        let v1 = Vec3::new(3.0, 2.0, 1.0);

        let v2 = v0 / v1;

        assert_eq!(v2[0], 1.0 / 3.0);
        assert_eq!(v2[1], 3.0 / 2.0);
        assert_eq!(v2[2], 4.0 / 1.0);
    }
}
