use std::f32::consts::FRAC_2_PI;

use crate::linalg::Vec3;
use crate::linalg::{self, random_0_1};

// TODO: Needs access to blue noise or some low discrepencies noise source to sample uniformly
pub trait BRDF: Send + Sync {
    // Sample the BRDF in normalized shading space (x: tangent, y: normal, z: bitangent)
    // wi: Incoming irradiance vector
    // n  : Normal vector
    // Return: Outgoing radiance vector importance sampled from the BRDF in shading space
    fn sample(&self, wi: Vec3, n: Vec3) -> Vec3;

    // Evaluates the BRDF in normalized shading space (x: tangent, y: normal, z: bitangent)
    // wi: Incoming irradiance vector
    // n  : Normal vector
    // wo: Outgoing vector of reflected radiance
    // Return: Radiance out through the vector wo in shading space
    fn eval(&self, wi: Vec3, n: Vec3, wo: Vec3) -> Vec3;

    // Evaluates the probability density function in normalized shading space (x: tangent, y: normal, z: bitangent)
    // wi: Incoming irradiance vector
    // n  : Normal vector
    // wo: Outgoing vector of reflected radiance
    // Return: Probability of choosing wo when given wi & n in shading space
    fn pdf(&self, wi: Vec3, n: Vec3, wo: Vec3) -> f32;
}

// Lambertian BRDF
// Desc: Cosine distribution of light scattering a.k.a lambertian BRDF
// Src:
#[derive(Debug, Clone, Copy)]
pub struct Lambertian {
    albedo: Vec3,
}

impl Lambertian {
    pub fn new(albedo: Vec3) -> Lambertian {
        Lambertian { albedo }
    }
}

impl BRDF for Lambertian {
    fn sample(&self, _wi: Vec3, n: Vec3) -> Vec3 {
        // TODO: Importance sample the Lambertian BRDF
        linalg::random_point_on_hemisphere(n)

        // FIXME: Returns vector in unit space should be in shading space
        // let r = f32::sqrt(random_0_1());
        // let theta = random_0_1() * 2.0 * std::f32::consts::PI;
        // let psi = random_0_1() * std::f32::consts::FRAC_2_PI;

        // let x = r * f32::cos(theta);
        // let y = r * f32::sin(psi);

        // // Project z up to the unit hemisphere
        // let z = f32::sqrt(1.0 - x * x - y * y);
        // Vec3 { x, y, z }.normalize()
    }

    fn eval(&self, _wi: Vec3, n: Vec3, wo: Vec3) -> Vec3 {
        wo.dot(n) * self.albedo * std::f32::consts::FRAC_1_PI
    }

    fn pdf(&self, wi: Vec3, n: Vec3, wo: Vec3) -> f32 {
        wo.dot(n) * std::f32::consts::FRAC_1_PI
    }
}

// Oren-Nayar BRDF
// Desc:
// Src[0]: https://en.wikipedia.org/wiki/Oren%E2%80%93Nayar_reflectance_model
// Src[1]: http://www1.cs.columbia.edu/CAVE/publications/pdfs/Oren_SIGGRAPH94.pdf
#[derive(Debug, Clone, Copy)]
pub struct OrenNayar {
    albedo: Vec3,
    a: f32,
    b: f32,
}

impl OrenNayar {
    pub fn new(albedo: Vec3, roughness: f32) -> OrenNayar {
        let sigma2 = roughness * roughness;
        let a = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
        let b = 0.45 * sigma2 / (sigma2 + 0.09);
        OrenNayar { albedo, a, b }
    }
}

impl BRDF for OrenNayar {
    fn eval(&self, wi: Vec3, _n: Vec3, wo: Vec3) -> Vec3 {
        // TODO: Phi calculations not 100% correct?
        let sign = |s: f32| -> f32 {
            if s < 0.0 {
                -1.0
            } else {
                1.0
            }
        };

        let phi_i = sign(wi.dot(Vec3::z())) * std::f32::consts::PI * (1.0 - wi.dot(Vec3::x()));
        let phi_r = sign(wo.dot(Vec3::z())) * std::f32::consts::PI * (1.0 - wo.dot(Vec3::x()));
        let max_cos = linalg::fmax(0.0, (phi_r - phi_i).cos());

        let theta_i = wi.dot(Vec3::y());
        let theta_r = wo.dot(Vec3::y());
        let alpha = linalg::fmax(theta_i, theta_r);
        let beta = linalg::fmin(theta_i, theta_r);

        let brdf = self.a + (self.b * max_cos * alpha.sin() * beta.tan());

        self.albedo * std::f32::consts::FRAC_1_PI * brdf
    }

    fn sample(&self, _wi: Vec3, n: Vec3) -> Vec3 {
        // TODO: Importance sample the Oren-Nayar BRDF
        linalg::random_point_on_hemisphere(n)
    }

    fn pdf(&self, wi: Vec3, n: Vec3, wo: Vec3) -> f32 {
        1.0
    }
}
