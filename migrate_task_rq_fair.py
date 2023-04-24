import collections
import json
import re
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from bcc import BPF

ebpf_code = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct data_t {
    u32 cpu;
    u32 pid;
};

BPF_PERF_OUTPUT(events);

int trace_migrate_task_rq_fair(struct pt_regs *ctx, struct task_struct *p, int cpu) {
    struct data_t data = {};
    
    data.cpu = cpu;
    data.pid = p->pid;
    
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""
# SAMPLE_INTERVAL_MS = 100

smt_threads = [1, 3, 5, 7, 9, 11, 13, 15]


def process_event(cpu, data, size):
    event = b["events"].event(data)
    if event.pid != 0:  # Ignore PID 0
        task_balancing.append((event.cpu, event.pid))


def get_core_types():
    output = subprocess.check_output(
        "lscpu -e=CPU,CORE,ONLINE | tail -n +2", shell=True
    ).decode()
    core_mapping = {}
    for line in output.strip().split("\n"):
        cpu, core, online = line.split()
        if online == "yes":
            cpu = int(cpu)
            core = int(core)
            core_mapping[cpu] = "P" if core < 8 else "E"
    return core_mapping


def run_long_task(command):
    process = subprocess.Popen(command, shell=True, user="pt", group="pt")
    last_sample_time = time.time()

    while process.poll() is None:
        b.perf_buffer_poll()
        # current_time = time.time()
        # if (current_time - last_sample_time) * 1000 >= SAMPLE_INTERVAL_MS:
        #     b.perf_buffer_poll()
        #     last_sample_time = current_time
        time.sleep(0.001)  # Sleep for 1 ms to reduce CPU usage

    print("Long task completed.")


def plot_task_balancing_data(task_balancing):
    cpu_counts = collections.defaultdict(int)
    for cpu, _ in task_balancing:
        cpu_counts[cpu] += 1

    core_types = get_core_types()

    x = sorted(cpu_counts.keys())
    y = [cpu_counts[cpu] for cpu in x]
    core_labels = [core_types[cpu] for cpu in x]

    fig, ax = plt.subplots()
    bars = ax.bar(x, y)

    # Color the bars according to core type
    for i, bar in enumerate(bars):
        if core_labels[i] == "P":
            if i in smt_threads:
                bar.set_color("tab:green")
            else:
                bar.set_color("tab:blue")
        else:
            bar.set_color("tab:orange")
        # bar.set_color("tab:blue" if core_labels[i] == "P" else "tab:orange")

    ax.set_xlabel("CPU ID (Blue: Performance Core, Orange: Efficient Core), Green: SMT")
    ax.set_ylabel("Migration Count")
    ax.set_title("Task Migration")

    ax.set_xticks(x)
    # ax.set_xticklabels(
    #     [f'{"*" if cpu in smt_threads else ""}{cpu}' for cpu in x],
    #     rotation=90,
    #     ha="right",
    # )  # Rotate x-axis labels
    ax.set_xticklabels(x)  # Rotate x-axis labels
    plt.tight_layout()  # Adjust spacing

    plt.savefig("task_balancing_bar_chart_cores.png")

def save_data_to_json(task_balancing):
    data = [{"cpu": cpu, "pid": pid } for cpu, pid in task_balancing]
    with open("task_balancing_data.json", "w") as json_file:
        json.dump(data, json_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: sudo python3 migrate_task_rq_fair.py '<long_task_command>'")
        sys.exit(1)

    long_task_command = sys.argv[1]

    b = BPF(text=ebpf_code)
    b.attach_kprobe(event="migrate_task_rq_fair", fn_name="trace_migrate_task_rq_fair")
    b["events"].open_perf_buffer(process_event)

    task_balancing = []

    print(f"Running long task: {long_task_command}")
    run_long_task(long_task_command)

    print("Saving task balancing data to JSON file...")
    save_data_to_json(task_balancing)

    print("Plotting task balancing data and saving the plot to an image file...")
    plot_task_balancing_data(task_balancing)
