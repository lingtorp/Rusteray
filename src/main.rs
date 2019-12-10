extern crate minifb;
use minifb::{Key, Window, WindowOptions};

extern crate tobj;
use std::path::Path;

extern crate rand;
use rand::prelude::*;

mod linalg;
use linalg::Vec3;

use std::sync::Arc;

extern crate threadpool;
use threadpool::ThreadPool;

const WINDOW_WIDTH: usize = 500;
const WINDOW_HEIGHT: usize = 500;
const SAMPLE_COUNT: usize = 50;
const RAY_DEPTH_MAX: usize = 50;

#[derive(Debug, Copy, Clone)]
struct Camera {
    frame_anchor: Vec3,
    u: Vec3,
    v: Vec3,
    position: Vec3,
    fov: f32,
}

impl Camera {
    fn new(fov_degrees: f32, from: Vec3, to: Vec3) -> Camera {
        let aspect = (WINDOW_WIDTH as f32) / (WINDOW_HEIGHT as f32);

        let half_height = (fov_degrees.to_radians() / 2.0).tan();
        let half_width = aspect * half_height;

        let wup = Vec3::y();

        let w = (from - to).normalize();
        let u = w.cross(wup).normalize();
        let v = w.cross(u).normalize();

        let horizontal = 2.0 * half_width * u;
        let vertical = 2.0 * half_height * v;

        let frame_anchor = from - half_width * u - half_height * v - w;

        Camera {
            frame_anchor: frame_anchor,
            u: horizontal,
            v: vertical,
            position: from,
            fov: fov_degrees,
        }
    }

    // (s, t) screen based offsets
    fn ray(&self, s: f32, t: f32) -> Ray {
        let d = (self.frame_anchor + s * self.u + t * self.v - self.position).normalize();
        Ray {
            origin: self.position,
            direction: d,
            depth: 1,
        }
    }
}

#[derive(Debug)]
enum RaycastResult {
    Miss,
    Hit {
        t: f32,
        u: f32,
        v: f32,
        triangle: Triangle,
    },
}

#[derive(Debug)]
struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub depth: usize,
}

impl Ray {
    fn at(&self, t: f32) -> Vec3 {
        self.origin + t * self.direction
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    n0: Vec3,
    n1: Vec3,
    n2: Vec3,
}

impl Triangle {
    // Ray-triangle intersection test from Real-Time Rendering 4th Edition (p. 964) (Möller-Trumbore algorithm)
    // Non-backface culling version of the algorithm
    // Src [0]: http://www.graphics.cornell.edu/pubs/1997/MT97.pdf
    fn intersects(&self, ray: &Ray) -> RaycastResult {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        let q = ray.direction.cross(e2);

        let a = e1.dot(q); // Determinant
        if a < std::f32::EPSILON && a > -std::f32::EPSILON {
            return RaycastResult::Miss;
        }

        let f = 1.0 / a;
        let s = ray.origin - self.v0;
        let u = f * (s.dot(q));

        if u < 0.0 || u > 1.0 {
            return RaycastResult::Miss;
        }

        let r = s.cross(e1);
        let v = f * (ray.direction.dot(r));
        if v < 0.0 || u + v > 1.0 {
            return RaycastResult::Miss;
        }

        let t = f * (e2.dot(r));

        // Line intersection - not Ray
        if t < 0.0 {
            return RaycastResult::Miss;
        }

        RaycastResult::Hit {
            t,
            u,
            v,
            triangle: *self,
        }
    }

    fn normal_at(&self, u: f32, v: f32) -> Vec3 {
        ((1.0 - u - v) * self.n0 + u * self.n1 + v * self.n2).normalize()
    }

    fn centroid(&self) -> Vec3 {
        Vec3 {
            x: (self.v0.x + self.v1.x + self.v2.x) / 3.0,
            y: (self.v0.y + self.v1.y + self.v2.y) / 3.0,
            z: (self.v0.z + self.v1.z + self.v2.z) / 3.0,
        }
    }
}

#[derive(Debug)]
struct AABB {
    min: Vec3,
    max: Vec3,
}

impl AABB {
    // AABB-Ray intersection test using 3D slab-method
    fn intersects(&self, ray: &Ray) -> bool {
        let min = (self.min - ray.origin) / ray.direction;
        let max = (self.max - ray.origin) / ray.direction;

        let mut tmin = std::f32::MIN;
        let mut tmax = std::f32::MAX;

        {
            let t0 = linalg::fmin(min.x, max.x);
            let t1 = linalg::fmax(min.x, max.x);
            tmin = linalg::fmax(t0, tmin);
            tmax = linalg::fmin(t1, tmax);
            if tmax < tmin {
                return false;
            }
        }

        {
            let t0 = linalg::fmin(min.y, max.y);
            let t1 = linalg::fmax(min.y, max.y);
            tmin = linalg::fmax(t0, tmin);
            tmax = linalg::fmin(t1, tmax);
            if tmax < tmin {
                return false;
            }
        }

        {
            let t0 = linalg::fmin(min.z, max.z);
            let t1 = linalg::fmax(min.z, max.z);
            tmin = linalg::fmax(t0, tmin);
            tmax = linalg::fmin(t1, tmax);
            if tmax < tmin {
                return false;
            }
        }

        true
    }

    fn from_triangles(triangles: &[Triangle]) -> AABB {
        let mut min_x = std::f32::MAX;
        let mut min_y = std::f32::MAX;
        let mut min_z = std::f32::MAX;
        let mut max_x = std::f32::MIN;
        let mut max_y = std::f32::MIN;
        let mut max_z = std::f32::MIN;

        for triangle in triangles {
            // Min
            if triangle.v0.x < min_x {
                min_x = triangle.v0.x;
            }
            if triangle.v1.x < min_x {
                min_x = triangle.v1.x;
            }
            if triangle.v2.x < min_x {
                min_x = triangle.v2.x;
            }

            if triangle.v0.y < min_y {
                min_y = triangle.v0.y;
            }
            if triangle.v1.y < min_y {
                min_y = triangle.v1.y;
            }
            if triangle.v2.y < min_y {
                min_y = triangle.v2.y;
            }

            if triangle.v0.z < min_z {
                min_z = triangle.v0.z;
            }
            if triangle.v1.z < min_z {
                min_z = triangle.v1.z;
            }
            if triangle.v2.z < min_z {
                min_z = triangle.v2.z;
            }

            // Max
            if triangle.v0.x > max_x {
                max_x = triangle.v0.x;
            }
            if triangle.v1.x > max_x {
                max_x = triangle.v1.x;
            }
            if triangle.v2.x > max_x {
                max_x = triangle.v2.x;
            }

            if triangle.v0.y > max_y {
                max_y = triangle.v0.y;
            }
            if triangle.v1.y > max_y {
                max_y = triangle.v1.y;
            }
            if triangle.v2.y > max_y {
                max_y = triangle.v2.y;
            }

            if triangle.v0.z > max_z {
                max_z = triangle.v0.z;
            }
            if triangle.v1.z > max_z {
                max_z = triangle.v1.z;
            }
            if triangle.v2.z > max_z {
                max_z = triangle.v2.z;
            }
        }

        AABB {
            min: Vec3 {
                x: min_x,
                y: min_y,
                z: min_z,
            },
            max: Vec3 {
                x: max_x,
                y: max_y,
                z: max_z,
            },
        }
    }

    // Based on the separating axis theorem (SAT) and Mr. Akenine-Möller himself and ref #2
    // Reference 1 [p. 2]: http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/pubs/tribox.pdf
    // Referens  2: https://stackoverflow.com/questions/17458562/efficient-aabb-triangle-intersection-in-c-sharp
    fn intersects_triangle(&self, t: Triangle) -> bool {
        let aabb_normals = vec![Vec3::x(), Vec3::y(), Vec3::z()];
        let mins = vec![self.min.x, self.min.y, self.min.z];
        let maxs = vec![self.max.x, self.max.y, self.max.z];

        // 3 tests - AABBs normals
        for i in 0..3 {
            let (min, max) = project(&[t.v0, t.v1, t.v2], aabb_normals[i]);
            if max <= mins[i] || min >= maxs[i] {
                return false;
            }
        }

        // 1 test - triangle normal/plane vs. AABB intersection
        {
            let normal = (t.v0 - t.v1).cross(t.v0 - t.v2);

            let center = (self.min + self.max) * 0.5; // AABB center
            let extents = self.max - center; // Positive extents
            let radius = extents.dot(normal.abs());

            // Compute distance of AABB from plane
            let distance = normal.dot(center);
            if distance <= radius {
                return false;
            }
        }

        // 9 tests - axis formed from edges of
        {
            // Compute edges of translated triangle
            let c = (self.min + self.max) * 0.5; // AABB center
            let h = self.max - c; // +Half-vector
            let edges = vec![
                t.v0 - t.v1 - (2.0 * c),
                t.v0 - t.v2 - (2.0 * c),
                t.v1 - t.v2 - (2.0 * c),
            ];
            let translated = vec![t.v0 - c, t.v1 - c, t.v2 - c];
            for i in 0..3 {
                for j in 0..3 {
                    let axis = edges[i].cross(aabb_normals[j]);
                    let r = h.dot(axis.abs());
                    let (t_min, t_max) = project(&translated, axis);
                    if t_min > r || t_max < -r {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn is_aabb_inside(&self, aabb: &AABB) -> bool {
        self.is_point_inside(aabb.min) && self.is_point_inside(aabb.max)
    }

    fn is_point_inside(&self, p: Vec3) -> bool {
        if p.x > self.max.x {
            return false;
        }
        if p.y > self.max.y {
            return false;
        }
        if p.z > self.max.z {
            return false;
        }
        if p.x < self.min.x {
            return false;
        }
        if p.y < self.min.y {
            return false;
        }
        if p.z < self.min.z {
            return false;
        }
        true
    }

    fn subdivide(&self, dimension: usize) -> Vec<AABB> {
        let size = (self.max - self.min) / (dimension as f32);
        let mut aabbs = Vec::new();

        for i in 0..dimension {
            for j in 0..dimension {
                for k in 0..dimension {
                    let offset = Vec3 {
                        x: i as f32,
                        y: j as f32,
                        z: k as f32,
                    };
                    let min = self.min + size * offset;
                    let aabb = AABB {
                        min: min,
                        max: min + size,
                    };
                    aabbs.push(aabb);
                }
            }
        }
        aabbs
    }
}

fn project(points: &[Vec3], on: Vec3) -> (f32, f32) {
    let mut max = std::f32::MIN;
    let mut min = std::f32::MAX;
    for point in points {
        let t = point.dot(on);
        if t < min {
            min = t;
        }
        if t > max {
            max = t;
        }
    }
    (min, max)
}

// NOTE: Phong reflection model as described by the .MTL format
#[derive(Debug, Clone, Copy)]
struct Material {
    diffuse: Vec3,
    emission: Vec3,
}

impl Material {
    fn new() -> Material {
        Material {
            diffuse: Vec3::zero(),
            emission: Vec3::zero(),
        }
    }

    fn shade(&self, scene: &Scene, ray: Ray, t: f32, u: f32, v: f32, triangle: Triangle) -> Vec3 {
        if ray.depth == RAY_DEPTH_MAX {
            return self.emission;
        }

        let n = triangle.normal_at(u, v);
        let d = linalg::random_point_on_hemisphere(n);

        let next_ray = Ray {
            origin: ray.at(t) + (d * Vec3::new(0.0001)),
            direction: d,
            depth: ray.depth + 1,
        };

        let cos_theta = d.dot(n);
        if cos_theta < 0.0 {
            let i = -ray.direction;
            let o = next_ray.direction;
            println!("I = quiver3({}, {}, {}) \nhold on \nN = quiver3({}, {}, {}) \nhold on \nO = quiver3({}, {}, {}) \n", i.x, i.y, i.z, n.x, n.y, n.z, o.x, o.y, o.z);
            return Vec3::new(0.0);
        }

        // NOTE: If diffuse reflectance exceeds 1 / PI it will reflect more light than it receives ..

        // TODO: Importance sampling the BRDF and lights
        let brdf = std::f32::consts::FRAC_1_PI; // Assuming Lambertian material always
        let pdf_inv = 1.0 / std::f32::consts::FRAC_2_PI;
        self.emission + brdf * pdf_inv * self.diffuse * scene.trace(next_ray)
    }
}

#[derive(Debug)]
struct Octree {
    dimension: usize,
    aabbs: Vec<AABB>,
    triangles: Vec<Vec<Triangle>>,
    material: Material,
}

impl Octree {
    fn new(dimension: usize, triangles: Vec<Triangle>, material: Material) -> Octree {
        let root_aabb = AABB::from_triangles(&triangles);
        println!("Root AABB: {:?}, dimension: {}", root_aabb, dimension);
        let aabbs = root_aabb.subdivide(dimension);

        let mut aabb_triangles: Vec<Vec<Triangle>> = Vec::new();
        for _ in 0..dimension * dimension * dimension {
            aabb_triangles.push(Vec::new());
        }

        for triangle in triangles {
            let mut did_fit_aabb = false;
            let triangle_aabb = AABB::from_triangles(&vec![triangle]);

            for (i, aabb) in aabbs.iter().enumerate() {
                if aabb.is_aabb_inside(&triangle_aabb) {
                    aabb_triangles[i].push(triangle);
                    did_fit_aabb = true;
                    break;
                }
            }

            if !did_fit_aabb {
                // Add triangle to all AABBs that intersects it
                for (i, aabb) in aabbs.iter().enumerate() {
                    if aabb.intersects_triangle(triangle) {
                        aabb_triangles[i].push(triangle);
                    }
                }
            }
        }

        Octree {
            dimension: dimension,
            aabbs: aabbs,
            triangles: aabb_triangles,
            material: material,
        }
    }

    fn intersects(&self, ray: &Ray) -> RaycastResult {
        let mut t0 = std::f32::MAX;
        let mut result = RaycastResult::Miss;
        // FIXME: Searching for ALL triangles hits in the AABBs measure AABB intersection t
        for (i, aabb) in self.aabbs.iter().enumerate() {
            if aabb.intersects(ray) {
                for triangle in &self.triangles[i] {
                    let res = triangle.intersects(ray);
                    match res {
                        RaycastResult::Hit { t, u, v, triangle } => {
                            if t < t0 && t > std::f32::EPSILON {
                                result = res;
                                t0 = t;
                            }
                        }
                        RaycastResult::Miss => continue,
                    };
                }
            }
        }
        result
    }
}

// TODO: Add emitter/light list
struct Scene {
    octrees: Vec<Octree>,
}

impl Scene {
    fn new(path: &str) -> Scene {
        let start = std::time::Instant::now();
        println!("Starting to load scene ...");

        let mut octrees = Vec::new();

        let obj = tobj::load_obj(&Path::new(path));
        assert!(obj.is_ok());
        let (models, materials) = obj.unwrap();

        println!("# of models: {}", models.len());
        println!("$ of materials: {}", materials.len());

        for (i, m) in models.iter().enumerate() {
            let mesh = &m.mesh;
            println!("model[{}].name = \'{}\'", i, m.name);
            println!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);

            println!("model[{}].vertices: {}", i, mesh.positions.len() / 3);
            assert!(mesh.positions.len() % 3 == 0);

            println!("model[{}].indices: {}", i, mesh.indices.len());
            assert!(mesh.indices.len() % 3 == 0);

            println!("model[{}].triangles: {}", i, mesh.indices.len() / 3);

            let compute_normals = mesh.normals.is_empty();
            if compute_normals {
                println!("Computing normals!");
            }

            let mut triangles = Vec::new();
            for idxs in mesh.indices.chunks(3) {
                let mut v = [Vec3::default(); 3];
                for i in 0..3 {
                    let idx = idxs[i] as usize;
                    let x = mesh.positions[3 * idx];
                    let y = mesh.positions[3 * idx + 1];
                    let z = mesh.positions[3 * idx + 2];
                    v[i] = Vec3 { x: x, y: y, z: z }
                }

                let mut n = [Vec3::zero(); 3];
                if compute_normals {
                    // FIXME: Must consider winding order of vertices when computing this
                    let normal = (v[0] - v[1]).cross(v[0] - v[2]).normalize();
                    n[0] = normal;
                    n[1] = n[0];
                    n[2] = n[0];
                    println!("-- COMPUTED NORMAL: {:?}", normal);
                } else {
                    for i in 0..3 {
                        let idx = idxs[i] as usize;
                        let x = mesh.normals[3 * idx];
                        let y = mesh.normals[3 * idx + 1];
                        let z = mesh.normals[3 * idx + 2];
                        n[i] = Vec3 { x: x, y: y, z: z }
                    }
                    println!("-- NORMAL: {:?}", n);
                }

                triangles.push(Triangle {
                    v0: v[0],
                    v1: v[1],
                    v2: v[2],
                    n0: n[0],
                    n1: n[1],
                    n2: n[2],
                });
            }

            let mut material = Material {
                diffuse: Vec3::zero(),
                emission: Vec3::zero(),
            };

            if let Some(mid) = mesh.material_id {
                material.diffuse = Vec3 {
                    x: materials[mid].diffuse[0],
                    y: materials[mid].diffuse[1],
                    z: materials[mid].diffuse[2],
                };

                if let emission = &materials[mid].unknown_param["Ke"] {
                    let strs = emission.split(" ").collect::<Vec<_>>();
                    if strs.len() == 3 {
                        let mut e = vec![0.0; 3];

                        for (i, s) in strs.iter().enumerate() {
                            if let Ok(num) = s.parse::<f32>() {
                                e[i] = num;
                            }
                        }

                        material.emission = Vec3 {
                            x: e[0],
                            y: e[1],
                            z: e[2],
                        };
                    }
                }
            }

            let dimension = 1;
            let octree = Octree::new(dimension, triangles, material);
            octrees.push(octree);
        }

        let end = std::time::Instant::now();
        let diff = end - start;
        println!(
            "Scene loaded in: {}.{} secs",
            diff.as_secs(),
            diff.subsec_millis()
        );

        Scene { octrees: octrees }
    }

    fn trace_background(&self, ray: Ray) -> Vec3 {
        let t = 0.5 * (ray.direction.y + 1.0);
        let white = Vec3::new(1.0);
        let light_blue = Vec3 {
            x: 0.5,
            y: 0.7,
            z: 1.0,
        };
        ((1.0 - t) * white + t * light_blue)
    }

    fn trace(&self, ray: Ray) -> Vec3 {
        let mut t0 = std::f32::MAX;
        let mut result = RaycastResult::Miss;
        let mut material = Material::new();

        // TODO: Create a octree hierarchy (similiar to a BVH)
        for octree in &self.octrees {
            let res = octree.intersects(&ray);
            match res {
                RaycastResult::Hit { t, u, v, triangle } => {
                    if t < t0 && t > std::f32::EPSILON {
                        t0 = t;
                        material = octree.material;
                        result = res;
                    }
                }
                RaycastResult::Miss => continue,
            };
        }

        match result {
            RaycastResult::Hit { t, u, v, triangle } => {
                material.shade(&self, ray, t, u, v, triangle)
            }
            RaycastResult::Miss => self.trace_background(ray),
        }
    }
}

enum Encoding {
    ARGB,
}

fn encode_color(encoding: Encoding, mut color: Vec3) -> u32 {
    match encoding {
        Encoding::ARGB => {
            // TODO: Implement f32.clamp and refactor this, used to avoid fireflies
            color.x = if color.x > 1.0 { 1.0 } else { color.x };
            color.y = if color.y > 1.0 { 1.0 } else { color.y };
            color.z = if color.z > 1.0 { 1.0 } else { color.z };
            color.x = if color.x < 0.0 { 0.0 } else { color.x };
            color.y = if color.y < 0.0 { 0.0 } else { color.y };
            color.z = if color.z < 0.0 { 0.0 } else { color.z };
            let ir = (color.x.sqrt() * 255.99) as u32;
            let ig = (color.y.sqrt() * 255.99) as u32;
            let ib = (color.z.sqrt() * 255.99) as u32;
            let ia = 1_u32;
            let mut pixel = 0_u32;
            pixel += ia << 24;
            pixel += ir << 16;
            pixel += ig << 8;
            pixel += ib;
            pixel
        }
    }
}

// TODO: Execute per region instead per pixel
struct Region {
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
}

fn main() {
    let mut buffer: Vec<u32> = vec![0; WINDOW_WIDTH * WINDOW_HEIGHT];

    let mut window = Window::new(
        "Rusteray - ESC to exit",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let filepath = "/home/alexander/repos/Rusteray/models/cornell_box/cornell_box_empty.obj";
    let scene = Arc::new(Scene::new(filepath));
    // FIXME: From one axis does not work
    let from = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 3.0,
    };
    let to = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    let camera = Camera::new(50.0, from, to);

    let pool = ThreadPool::new(4);

    let (tx, rx) = std::sync::mpsc::channel();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let start = std::time::Instant::now();

        for x in 0..WINDOW_WIDTH {
            for y in 0..WINDOW_HEIGHT {
                let tx = tx.clone();
                let camera = camera.clone();
                let scene = Arc::clone(&scene);
                pool.execute(move || {
                    let mut rng = rand::thread_rng();
                    let mut color = Vec3::zero();
                    for _ in 0..SAMPLE_COUNT {
                        // Flipped variable t s.t y+ axis is up
                        let r: f32 = rng.gen();
                        let s = ((x as f32) + r) / (WINDOW_WIDTH as f32);
                        let t = ((y as f32) + r) / (WINDOW_HEIGHT as f32);
                        let ray = camera.ray(s, t);
                        color = color + scene.trace(ray);
                    }
                    color = color / (SAMPLE_COUNT as f32);
                    tx.send((x, y, encode_color(Encoding::ARGB, color)))
                        .expect("Failed to send pixel!");
                });
            }
        }

        for _ in 0..WINDOW_WIDTH * WINDOW_HEIGHT {
            let (x, y, pixel) = rx.recv().unwrap_or_default();
            buffer[y * WINDOW_WIDTH + x] = pixel;
        }

        let end = std::time::Instant::now();
        let diff = end - start;
        println!(
            "Frame rendered in: {}.{} s",
            diff.as_secs(),
            diff.subsec_millis()
        );

        window.update_with_buffer(&buffer).unwrap();
    }
}
