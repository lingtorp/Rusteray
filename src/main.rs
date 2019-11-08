extern crate minifb;

use std::path::Path;
extern crate tobj;

mod linalg;
use linalg::Vec3;

use minifb::{Key, WindowOptions, Window};

const WINDOW_WIDTH: usize = 400;
const WINDOW_HEIGHT: usize = 400;

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
            u: Vec3{x: 1.0, y: 0.0, z: 0.0},
            v: Vec3{x: 0.0, y: 1.0, z: 0.0},
            fov: 75.0,
            position: Vec3{x: 0.0, y: 0.0, z: 0.0},
            direction: Vec3{x: 0.0, y: 0.0, z: 1.0}
       }
    }

    fn ray(&self, s: f32, t: f32) -> Ray {
        Ray{origin: self.position + s * self.u + t * self.v, direction: self.direction}
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

impl Triangle {

}

struct Scene {
    triangles: Vec<Triangle>
}

impl Scene {
    fn new(path: &str) -> Scene {
        let obj = tobj::load_obj(&Path::new(path));
        assert!(obj.is_ok());
        let (models, materials) = obj.unwrap();

        println!("# of models: {}", models.len());
        println!("$ of materials: {}", materials.len());

        let mut triangles: Vec<Triangle> = Vec::new();

        for (i, m) in models.iter().enumerate() {
            let mesh = &m.mesh;
            println!("model[{}].name = \'{}\'", i, m.name);
            println!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);

            println!("model[{}].vertices: {}", i, mesh.positions.len() / 3);
            assert!(mesh.positions.len() % 3 == 0);

            println!("model[{}].indices: {}", i, mesh.indices.len());
            assert!(mesh.indices.len() % 3 == 0);

            for idxs in mesh.indices.chunks(3) {
                let mut v = [Vec3::zero(), Vec3::zero(), Vec3::zero()];
                for i in 0..3 {
                    let idx = idxs[i] as usize;
                    let x = mesh.positions[3 * idx];
                    let y = mesh.positions[3 * idx + 1];
                    let z = mesh.positions[3 * idx + 2];
                    v[i] = Vec3{x: x, y: y, z: z}
                }

                triangles.push(Triangle{v0: v[0], v1: v[1], v2: v[2]});
            }
            
        }

        Scene{triangles: triangles}
    }

    fn trace(&self, ray: &Ray) -> u32 {
        255
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

    let scene = Scene::new("/home/alexander/Desktop/models/rust_logo.obj");
    let camera = Camera::new();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for x in 0..WINDOW_WIDTH {
            for y in 0..WINDOW_HEIGHT {
                for s in 0..1 {
                    let s = (x / WINDOW_WIDTH) as f32;
                    let t = (y / WINDOW_HEIGHT) as f32;
                    let ray = camera.ray(s, t);
                    buffer[x * WINDOW_WIDTH + y] = scene.trace(&ray);
                }
            }
        }
        
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer).unwrap();
    }
}
