
pub struct Vec3 {
    pub x: f32, y: f32, z: f32
}

impl Vec3 {
    pub fn new(s: f32) -> Vec3 {
        Vec3{x: s, y: s, z: s}
    }

    pub fn zero() -> Vec3 {
        Vec3::new(0.0)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f32) -> Vec3 {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs
        }
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z
        }
    }
}

