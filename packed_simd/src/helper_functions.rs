use libc::{cpu_set_t, pid_t, sched_setaffinity, CPU_SET, CPU_SETSIZE, CPU_ZERO};
use std::io::{BufRead, BufReader};
use std::process;

pub fn run_command(command_str: &str) {
    let arguments: Vec<&str> = command_str.split_whitespace().collect();
    let command = arguments[0];
    let args = &arguments[1..];

    let mut child = process::Command::new(command)
        .args(args)
        .stdout(process::Stdio::piped())
        .spawn()
        .expect("Failed to execute process");

    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);

    for line in reader.lines() {
        let line = line.unwrap();
        println!("{}", line);
    }

    let _ = child.wait();
}

// Use taskset run on select cpu core
pub fn run_taskset(command_str: &str, cpu_id: usize) {
    let mut cpu_set: cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe { CPU_ZERO(&mut cpu_set) };
    unsafe { CPU_SET(cpu_id, &mut cpu_set) };

    let result = unsafe { sched_setaffinity(0 as pid_t, CPU_SETSIZE as usize, &cpu_set) };
    if result != 0 {
        panic!("Failed to set cpu affinity");
    }

    run_command(command_str);
}
