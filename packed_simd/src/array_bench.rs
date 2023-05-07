use libc::{c_uint, syscall, SYS_getcpu};
use std::collections::HashMap;
use plotters::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use packed_simd::f32x8;
use std::io;

pub const ARRAY_SIZE: usize = 1024;
pub const INNER_ITERATIONS: usize = 1000;

pub fn avx2_elementwise_addition_parallel(
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

pub fn create_chart(
    file_name: &str,
    title: &str,
    data: &[(usize, usize)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_x = data.iter().map(|&(x, _)| x).max().unwrap_or(0) as i32;
    let max_y = data.iter().map(|&(_, y)| y).max().unwrap_or(0) as i32 + 10;
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
        .x_labels(data.len())
        .y_labels(10)
        .x_desc("CPU ID")
        .y_desc("Frequency")
        .axis_desc_style(("sans-serif", 40).into_font())
        .x_label_formatter(&|x| format!("{}", x / bar_width))
        .draw()?;

    chart
        .draw_series(
            Histogram::vertical(&chart)
                .style(BLUE.filled())
                .data(data.iter().map(|&(x, y)| (x as i32 * bar_width, y as i32))),
        )?
        .label("Frequency")
        .legend(move |(x, y)| Rectangle::new([(x, y), (x + 20, y + 20)], BLUE));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    Ok(())
}

pub fn elementwise_addition_parallel(
    a: &[f32],
    b: &[f32],
    cpu_ids: Arc<Mutex<HashMap<usize, usize>>>,
) {
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

pub fn get_current_cpu_id() -> io::Result<usize> {
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
