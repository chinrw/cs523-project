use itertools::Itertools;
use libc::{c_uint, syscall, SYS_getcpu};
use plotters::prelude::*;
use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex};

use packed_simd::f32x8;
use rand::Rng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::env;

const ARRAY_SIZE: usize = 1024;
const INNER_ITERATIONS: usize = 1000;

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_threads: usize = if args.len() > 1 {
        args[1].parse().expect("Invalid number of threads")
    } else {
        12
    };

    let mut rng = rand::thread_rng();
    let array1: Vec<f32> = (0..ARRAY_SIZE).map(|_| rng.gen::<f32>()).collect();
    let array2: Vec<f32> = (0..ARRAY_SIZE).map(|_| rng.gen::<f32>()).collect();

    let simd_thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let num_threads_non_simd = args.get(2).and_then(|arg| arg.parse().ok()).unwrap_or(12);

    let non_simd_thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads_non_simd)
        .build()
        .unwrap();

    let simd_cpu_ids = Arc::new(Mutex::new(HashMap::new()));
    let non_simd_cpu_ids = Arc::new(Mutex::new(HashMap::new()));

    let num_interations = args
        .get(3)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(100000);

    for _ in 0..num_interations {
        let array1_clone = array1.clone();
        let array2_clone = array2.clone();
        let non_simd_cpu_ids_clone = non_simd_cpu_ids.clone();

        non_simd_thread_pool.install(|| {
            elementwise_addition_parallel(&array1_clone, &array2_clone, non_simd_cpu_ids_clone);
        });
        let simd_cpu_ids_clone = simd_cpu_ids.clone();

        simd_thread_pool.install(|| {
            avx2_elementwise_addition_parallel(&array1, &array2, simd_cpu_ids_clone);
        });
        // println!("Finished one round of calculations.");
    }
    let sorted_simd_cpu_id_frequencies: Vec<(usize, usize)> = simd_cpu_ids
        .lock()
        .unwrap()
        .iter()
        .map(|(&k, &v)| (k, v))
        .sorted_by(|b, a| a.1.cmp(&b.1))
        .collect();

    println!(
        "SIMD CPU ID frequencies: {:?}",
        sorted_simd_cpu_id_frequencies
    );

    let sorted_non_simd_cpu_id_frequencies: Vec<(usize, usize)> = non_simd_cpu_ids
        .lock()
        .unwrap()
        .iter()
        .map(|(&k, &v)| (k, v))
        .sorted_by(|b, a| a.1.cmp(&b.1))
        .collect();

    println!(
        "Non-SIMD CPU ID frequencies: {:?}",
        sorted_non_simd_cpu_id_frequencies
    );

    create_chart(
        "simd_cpu_id_frequencies.png",
        "SIMD CPU ID Frequencies",
        &sorted_simd_cpu_id_frequencies,
    )
    .unwrap();

    create_chart(
        "non_simd_cpu_id_frequencies.png",
        "Non-SIMD CPU ID Frequencies",
        &sorted_non_simd_cpu_id_frequencies,
    )
    .unwrap();

    println!("Finished all calculations.");
}

fn avx2_elementwise_addition_parallel(
    a: &[f32],
    b: &[f32],
    cpu_ids: Arc<Mutex<HashMap<usize, usize>>>,
) {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length.");

    let simd_chunks = a.len() / 8;
    let remainder = a.len() % 8;

    (0..simd_chunks).into_par_iter().for_each(|i| {
        let cpu_id = get_current_cpu_id().unwrap_or(0);
        {
            let mut ids = cpu_ids.lock().unwrap();
            *ids.entry(cpu_id).or_insert(0) += 1;
        }

        for _ in 0..INNER_ITERATIONS {
            let a_chunk = f32x8::from_slice_unaligned(&a[i * 8..(i + 1) * 8]);
            let b_chunk = f32x8::from_slice_unaligned(&b[i * 8..(i + 1) * 8]);
            let _sum_chunk = a_chunk + b_chunk;
        }
    });

    if remainder > 0 {
        let offset = simd_chunks * 8;
        for _ in 0..remainder {
            let _sum = a[offset] + b[offset];
        }
    }
}

fn create_chart(
    file_name: &str,
    title: &str,
    data: &[(usize, usize)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_x = data.iter().map(|&(x, _)| x).max().unwrap_or(0) as i32;
    let max_y = data.iter().map(|&(_, y)| y).max().unwrap_or(0) as i32;
    let bar_width = 20;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 40).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..(max_x * bar_width), 0..max_y)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(1)
        .y_labels(5)
        .x_desc("CPU ID")
        .y_desc("Frequency")
        .axis_desc_style(("sans-serif", 15).into_font())
        .x_label_formatter(&|x| format!("{}", x / bar_width))
        .draw()?;

    chart
        .draw_series(
            Histogram::vertical(&chart)
                .style(BLUE.filled())
                .data(data.iter().map(|&(x, y)| (x as i32 * bar_width, y as i32))),
        )?
        .label("Frequency")
        .legend(move |(x, y)| Rectangle::new([(x, y), (x + 20, y + 20)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}


fn elementwise_addition_parallel(a: &[f32], b: &[f32], cpu_ids: Arc<Mutex<HashMap<usize, usize>>>) {
    assert_eq!(a.len(), b.len(), "Arrays must have the same length.");

    // since simd is 8 elements, we need to reduce the total interation by 8
    // to get better frequencies of cpu id
    let reduced_length = a.len() / 8;

    (0..reduced_length).into_par_iter().for_each(|i| {
        let cpu_id = get_current_cpu_id().unwrap_or(0);
        {
            let mut ids = cpu_ids.lock().unwrap();
            *ids.entry(cpu_id).or_insert(0) += 1;
        }
        for _ in 0..INNER_ITERATIONS {
            let index = i * 8;
            let _sum = a[index] + b[index];
        }
    });
}

fn get_current_cpu_id() -> io::Result<usize> {
    let mut cpu: c_uint = 0;
    let mut node: c_uint = 0;

    let result = unsafe {
        syscall(
            SYS_getcpu,
            &mut cpu as *mut c_uint,
            &mut node as *mut c_uint,
            std::ptr::null::<libc::c_void>() as *mut libc::c_void,
        )
    };

    if result == 0 {
        Ok(cpu as usize)
    } else {
        Err(io::Error::last_os_error())
    }
}
