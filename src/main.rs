extern crate minifb;

use std::path::Path;
extern crate tobj;

mod linalg;
use linalg::Vec3;

use minifb::{Key, WindowOptions, Window};

const WINDOW_WIDTH: usize = 400;
const WINDOW_HEIGHT: usize = 400;

struct Camera {
    frame_anchor: Vec3,
    u: Vec3,
    v: Vec3,
    fov: f32,
    position: Vec3,
    direction: Vec3
}

impl Camera {
    fn new() -> Camera {
        let position = Vec3::z() * 10.0;
        let fov: f32 = 75.0;

        let aspect = (WINDOW_HEIGHT as f32) / (WINDOW_WIDTH as f32);
        let theta = fov.to_radians();

        let half_height = (theta / 2.0).tan();
        let half_width  = aspect * half_height;

        let u = Vec3::x() * (WINDOW_WIDTH as f32);
        let v = Vec3::y() * (WINDOW_HEIGHT as f32);
        let frame_anchor = position - half_width * u - half_height * v;

        let direction =  Vec3{x: 0.0, y: 0.0, z: 1.0};

        Camera {
            frame_anchor: frame_anchor,
            u: u,
            v: v,
            fov: fov,
            position: position,
            direction: direction
       }
    }

    // (s, t) screen based offsets
    fn ray(&self, s: f32, t: f32) -> Ray {
        let o =  self.position + s * self.u + t * self.v;
        let d = self.frame_anchor + self.direction - self.position;
        Ray{origin: o, direction: d}
    }
}

#[derive(Debug)]
enum RaycastResult {
    Miss,
    Hit{t: f32, u: f32, v: f32}
}

#[derive(Debug)]
struct Ray {
    pub origin: Vec3,
    pub direction: Vec3
}

#[derive(Debug)]
struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3
}

impl Triangle {
    // Ray-triangle intersection test from Real-Time Rendering 4th Edition (p. 964)
    fn intersects(&self, ray: &Ray) -> RaycastResult {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        let q = ray.direction.cross(&e2);

        let a = e1.dot(&q);
        if a < std::f32::EPSILON && a > std::f32::EPSILON {
            return RaycastResult::Miss;
        }

        let f = 1.0 / a;
        let s = ray.origin - self.v0;
        let u = f * (s.dot(&q));

        if u < 0.0 {
            return RaycastResult::Miss;
        }

        let r = s.cross(&e1);
        let v = f * (ray.direction.dot(&r));
        if v < 0.0 || u + v > 1.0 {
            return RaycastResult::Miss;
        }

        let t = f * (e2.dot(&r));
        RaycastResult::Hit{t, u, v}
    }
}

struct Scene {
    triangles: Vec<Triangle>
}

impl Scene {
    fn new(path: &str) -> Scene {
        // let mut triangles: Vec<Triangle> = Vec::new();
        // triangles.push(Triangle{v0: Vec3{x: 0.0, y: 0.5, z: 0.0}, v1: Vec3{x: -0.5, y: 0.0, z: 0.0}, v2: Vec3{x: 0.5, y: 0.0, z: 0.0}});
        // Scene{triangles: triangles}

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
        for triangle in &self.triangles {
            let res = triangle.intersects(&ray);
            match res {
                RaycastResult::Hit{t, u, v} => return 255,
                RaycastResult::Miss => continue
            };
        }
        0
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
                    let s = (x as f32) / (WINDOW_WIDTH as f32);
                    let t = (y as f32) / (WINDOW_HEIGHT as f32);
                    let ray = camera.ray(s, t);
                    println!("{:?}", ray);
                    buffer[x * WINDOW_WIDTH + y] = scene.trace(&ray);
                }
            }
        }
        
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer).unwrap();
    }
}
