extern crate rayon;

pub type T = f32;
pub const EPS: T = 1e-8;

pub mod default {
    use super::{EPS, T};
    use rand::Rng;

    pub fn accum(a: &mut [T], b: &[T]) {
        debug_assert!(a.len() == b.len());
        for (ai, bi) in a.iter_mut().zip(b.iter()) {
            *ai += *bi;
        }
    }

    pub fn dotprod(a: &[T], b: &[T]) -> T {
        debug_assert!(a.len() == b.len());
        let mut v: T = 0.0;
        for (ai, bi) in a.iter().zip(b.iter()) {
            v += (*ai) * (*bi);
        }
        v
    }

    /// Matmul: W (d,n) @ x (n,) -> xout (d,)
    pub fn _matmul(w: &[T], x: &[T], d: usize, out: &mut [T]) {
        debug_assert!(w.len() == d * x.len());
        debug_assert!(d == out.len());

        let n = x.len();
        for i in 0..d {
            let mut v: T = 0.0;
            for j in 0..n {
                v += w[i * n + j] * x[j];
            }
            out[i] = v;
        }
    }

    pub fn rmsnorm(a: &[T], weight: &[T], out: &mut [T]) {
        debug_assert!(a.len() == out.len());
        debug_assert!(a.len() == weight.len());
        let mut v: T = 0.0;
        let len = a.len();
        for i in 0..len {
            v += a[i] * a[i];
        }
        v /= len as f32;
        v = 1.0 / (v.sqrt().max(EPS));
        // normalize and scale
        for i in 0..len {
            out[i] = weight[i] * v * a[i];
        }
        // for ((oi, wi), ai) in out.iter_mut().zip(weight.iter()).zip(a.iter()) {
        //     *oi = (*wi) * v * (*ai);
        // }
    }

    /// F.silu; silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    pub fn silu_inplace(x: &mut [T]) {
        for xi in x.iter_mut() {
            *xi *= 1.0f32 / (1.0f32 + (-(*xi)).exp());
        }
    }

    pub fn softmax_inplace(a: &mut [T]) {
        debug_assert!(!a.is_empty());
        let max = a.iter().copied().fold(T::NEG_INFINITY, T::max);
        for ai in a.iter_mut() {
            *ai = (*ai - max).exp();
        }
        let sum: T = a.iter().sum();
        for ai in a.iter_mut() {
            *ai /= sum;
        }
    }

    pub fn argmax(a: &[T]) -> usize {
        debug_assert!(!a.is_empty());
        let mut max_idx: usize = 0;
        let mut max = T::NEG_INFINITY;
        for (i, ai) in a.iter().enumerate() {
            if *ai > max {
                max = *ai;
                max_idx = i;
            }
        }
        max_idx
    }

    pub fn _logits_to_prob(logits: &mut [T], temperature: f32) {
        logits.iter_mut().for_each(|v| {
            *v /= temperature;
        });
        // apply softmax to the logits to get the probabilities for next token
        softmax_inplace(logits);
    }

    pub fn sample(a: &[T], rng: &mut rand::rngs::ThreadRng) -> usize {
        debug_assert!(!a.is_empty());
        let mut v: f32 = rng.gen();
        for (i, ai) in a.iter().enumerate() {
            if v <= *ai {
                return i;
            }
            v -= *ai;
        }
        return a.len() - 1;
    }
}

// #[cfg(feature = "rayon")]
pub mod parallel {
    use super::default;
    use super::T;
    use rayon::iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    };

    // pub fn init_ops() {}
    // pub fn accum(a: &mut [T], b: &[T]) {
    //     debug_assert!(a.len() == b.len());
    //     a.par_iter_mut().zip(b.par_iter()).for_each(|(ai, bi)| {
    //         *ai += *bi;
    //     });
    // }

    pub fn _dotprod(a: &[T], b: &[T]) -> T {
        debug_assert!(a.len() == b.len());
        return a
            .par_iter()
            .zip(b.par_iter())
            .map(|(ai, bi)| (*ai) * (*bi))
            .sum();
    }

    /// Matmul: W (d,n) @ x (n,) -> xout (d,)
    pub fn matmul(w: &[T], x: &[T], d: usize, out: &mut [T]) {
        debug_assert!(w.len() == d * x.len());
        debug_assert!(d == out.len());
        let n = x.len();
        out.par_iter_mut().enumerate().for_each(|(i, out_i)| {
            let offset = i * n;
            *out_i = default::dotprod(&w[offset..offset + n], x);
        });
    }

    // pub fn rmsnorm(a: &[T], weight: &[T], out: &mut [T]) {
    //     debug_assert!(a.len() == out.len());
    //     debug_assert!(a.len() == weight.len());
    //     let len = a.len();
    //     let mut v: T = a.par_iter().map(|ai| ai * ai).sum();
    //     v /= len as f32;
    //     v = 1.0 / (v.sqrt().max(EPS));
    //     out.par_iter_mut().enumerate().for_each(|(i, out_i)| {
    //         *out_i = weight[i] * (v * a[i]);
    //     });
    // }

    pub fn softmax_inplace(a: &mut [T]) {
        debug_assert!(!a.is_empty());
        let max = a.par_iter().copied().max_by(|a, b| a.total_cmp(b)).unwrap();
        a.par_iter_mut().for_each(|ai| {
            *ai = ((*ai) - max).exp();
        });
        let sum: T = a.par_iter().sum();
        a.par_iter_mut().for_each(|ai| {
            *ai /= sum;
        });
    }

    pub fn logits_to_prob(logits: &mut [T], temperature: f32) {
        logits.par_iter_mut().for_each(|v| {
            *v /= temperature;
        });
        // apply softmax to the logits to get the probabilities for next token
        softmax_inplace(logits);
    }
}

pub use default::{accum, argmax, dotprod, rmsnorm, sample, silu_inplace, softmax_inplace};
pub use parallel::{logits_to_prob, matmul};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dotprod() {
        let a = vec![2.0, 3.0];
        let b = vec![3.0, 4.0];
        let ret = dotprod(&a, &b);
        assert_eq!(ret, 18.0);
    }

    #[test]
    fn test_matmul() {
        let a = vec![2.0, 3.0];
        let w = vec![3.0, 4.0, 1.0, 3.0, 1.0, 1.0];
        let dim = 3;
        let mut ret = vec![0.0 as T; dim];
        matmul(&w, &a, dim, &mut ret);
        assert_eq!(ret, vec![18.0, 11.0, 5.0]);
    }

    #[test]
    fn test_argmax() {
        let a = vec![1.0, 2.0];
        assert_eq!(argmax(&a), 1);
    }
    #[test]
    fn test_softmax_inplace() {
        let mut a = vec![0.0, 0.0, 0.0, 0.0];
        softmax_inplace(&mut a);
        assert_eq!(a, vec![0.25, 0.25, 0.25, 0.25]);
    }
    #[test]
    fn test_rmsnorm() {
        {
            let a = vec![2.0, 2.0];
            let weight = vec![1.0, 1.0];
            let mut ret = vec![0.0 as T; a.len()];
            rmsnorm(&a, &weight, &mut ret);
            assert_eq!(ret, vec![1.0, 1.0]);
        }
        {
            let a = vec![3.0, 4.0];
            let weight = vec![1.0, 1.0];
            let mut ret = vec![0.0 as T; a.len()];
            rmsnorm(&a, &weight, &mut ret);
            assert_eq!(ret, vec![0.6 * (2.0_f32.sqrt()), 0.8 * (2.0_f32.sqrt())]);
        }
    }

    #[test]
    fn test_silu_inplace() {
        let mut a = vec![-1.0_f32, 0.0, 1.0];
        silu_inplace(&mut a);
        assert_eq!(a, vec![-0.26894143, 0.0, 0.7310586]);
    }
}
