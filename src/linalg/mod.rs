extern crate rand;
use rand_distr::{Distribution, Uniform, UnitSphere};

#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn from(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x: x, y: y, z: z }
    }

    pub fn new_from(v: Vec3) -> Vec3 {
        Vec3 {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }

    pub fn new(s: f32) -> Vec3 {
        Vec3 { x: s, y: s, z: s }
    }

    pub fn one() -> Vec3 {
        Vec3::new(1.0)
    }

    pub fn zero() -> Vec3 {
        Vec3::new(0.0)
    }

    pub fn x() -> Vec3 {
        Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn y() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    }

    pub fn z() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }

    pub fn lng(self) -> f32 {
        self.dot(self).sqrt()
    }

    // NOTE: Assumes that the normal is normalized
    pub fn reflect(self, n: Vec3) -> Vec3 {
        self - 2.0 * (self.dot(n)) * n
    }

    pub fn abs(&self) -> Vec3 {
        Vec3 {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    pub fn normalize(self) -> Vec3 {
        let lng = 1.0 / self.dot(self).sqrt();
        Vec3 {
            x: self.x * lng,
            y: self.y * lng,
            z: self.z * lng,
        }
    }

    pub fn dot(&self, v: Vec3) -> f32 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    pub fn cross(&self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * v.z - self.z * v.y,
            y: self.z * v.x - self.x * v.z,
            z: self.x * v.y - self.y * v.x,
        }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Default for Vec3 {
    fn default() -> Vec3 {
        Vec3::zero()
    }
}

impl std::ops::Div<Vec3> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
        }
    }
}

impl std::ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f32) -> Vec3 {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl std::ops::Div<i32> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: i32) -> Vec3 {
        let f = rhs as f32;
        self / f
    }
}

impl std::ops::Div<u32> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: u32) -> Vec3 {
        let f = rhs as f32;
        self / f
    }
}

impl std::ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Mul<i32> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: i32) -> Vec3 {
        let f = rhs as f32;
        self * f
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f32) -> Vec3 {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl std::ops::Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

pub fn random_0_1() -> f32 {
    let step = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();
    step.sample(&mut rng)
}

// NOTE: http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
// Returns vector of a random uniformly chosen vector oriented by the normal (n) on the positive hemisphere
pub fn random_point_on_hemisphere(n: Vec3) -> Vec3 {
    loop {
        let v: [f32; 3] = UnitSphere.sample(&mut rand::thread_rng());
        let p = Vec3 {
            x: v[0],
            y: v[1],
            z: v[2],
        };
        if p.dot(n) > 0.0 {
            return p;
        }
    }
}

pub fn fmin(t0: f32, t1: f32) -> f32 {
    if t0 < t1 {
        t0
    } else {
        t1
    }
}

pub fn fmax(t0: f32, t1: f32) -> f32 {
    if t0 > t1 {
        t0
    } else {
        t1
    }
}
