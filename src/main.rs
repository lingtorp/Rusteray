extern crate minifb;

mod linalg;
use linalg::Vec3;

use minifb::{Key, WindowOptions, Window};

const WINDOW_WIDTH: usize = 640;
const WINDOW_HEIGHT: usize = 360;

struct Camera {
    u: Vec3,
    v: Vec3,
    fov: f32,
    position: Vec3,
    direction: Vec3
}

impl Camera {
    fn new() -> Camera {
        Camera { 
            fov: 75.0,
            position: Vec3{x: 0.0, y: 0.0, z: 0.0},
            direction: Vec3{x: 0.0, y: 0.0, z: 1.0}
       }
    }

    fn ray(&self, s: f32, t: f32) {
        let direction = Vec3::zero();
        Ray{origin: self.position + s * self.u + t * self.v, direction: direction};
    }
}

struct HitData {

}

enum RaycastResult {
    Miss,
    Hit(HitData)
}

struct Ray {
    pub origin: Vec3,
    pub direction: Vec3
}

struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3
}

struct Scene {
    triangles: Vec<Triangle>
}

impl Scene {
    fn new() -> Scene {
        Scene{
            vec![] // TODO: ... 
        }
    }

    fn trace(&self, ray: &Ray) {
        
    }
}

fn main() {
    let mut buffer: Vec<u32> = vec![0; WINDOW_WIDTH * WINDOW_HEIGHT];

    let mut window = Window::new("Rusteray - ESC to exit",
                                 WINDOW_WIDTH,
                                 WINDOW_HEIGHT,
                                 WindowOptions::default()).unwrap_or_else(|e| {
                                     panic!("{}", e);
                                 });

    let scene = Scene::new();
    let camera = Camera::new();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for x in 0..WINDOW_WIDTH {
            for y in 0..WINDOW_HEIGHT {
                for s in 0..1 {
                    let ray = camera.ray(x / WINDOW_WIDTH, y / WINDOW_HEIGHT);
                    buffer[x * WINDOW_WIDTH + y] = scene.trace(ray);
                }
            }
        }
        
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer).unwrap();
    }
}
