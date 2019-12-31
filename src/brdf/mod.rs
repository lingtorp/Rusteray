use crate::linalg;
use crate::linalg::Vec3;

pub trait BRDF: Send + Sync {
    // Evaluates the BRDF in normalized shading space (x: tangent, y: normal, z: bitangent)
    // w_i: Incoming vector irradiance vector
    // w_r: Outgoing vector of reflected radiance
    fn eval(&self, w_i: Vec3, w_r: Vec3) -> Vec3;
}

// Lambertian BRDF
// Desc:
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
    fn eval(&self, _w_i: Vec3, _w_r: Vec3) -> Vec3 {
        self.albedo * std::f32::consts::FRAC_1_PI
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
    fn eval(&self, w_i: Vec3, w_r: Vec3) -> Vec3 {
        // TODO: Phi calculations not 100% correct?
        let sign = |s: f32| -> f32 {
            if s < 0.0 {
                -1.0
            } else {
                1.0
            }
        };

        let phi_i = sign(w_i.dot(Vec3::z())) * std::f32::consts::PI * (1.0 - w_i.dot(Vec3::x()));
        let phi_r = sign(w_r.dot(Vec3::z())) * std::f32::consts::PI * (1.0 - w_r.dot(Vec3::x()));
        let max_cos = linalg::fmax(0.0, (phi_r - phi_i).cos());

        let theta_i = w_i.dot(Vec3::y());
        let theta_r = w_r.dot(Vec3::y());
        let alpha = linalg::fmax(theta_i, theta_r);
        let beta = linalg::fmin(theta_i, theta_r);

        let brdf = self.a + (self.b * max_cos * alpha.sin() * beta.tan());

        self.albedo * std::f32::consts::FRAC_1_PI * brdf
    }
}
