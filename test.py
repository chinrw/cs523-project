from bcc import BPF
import ctypes

bpf_source = """
#include <bcc/proto.h>
#include <linux/sched.h>

struct data_t {
    u32 cpu;
    u32 pull_cpu;
    u32 result;
};

BPF_PERF_OUTPUT(events);

int trace_asym_smt_can_pull_tasks(struct pt_regs *ctx, u32 cpu, u32 pull_cpu) {
    struct data_t data = {};
    data.cpu = cpu;
    data.pull_cpu = pull_cpu;
    data.result = PT_REGS_RC(ctx);
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

bpf = BPF(text=bpf_source)
bpf.attach_kprobe(event="asym_smt_can_pull_tasks", fn_name="trace_asym_smt_can_pull_tasks")

class Data(ctypes.Structure):
    _fields_ = [
        ("cpu", ctypes.c_uint32),
        ("pull_cpu", ctypes.c_uint32),
        ("result", ctypes.c_uint32),
    ]

def print_event(cpu, data, size):
    event = ctypes.cast(data, ctypes.POINTER(Data)).contents
    print(f"CPU: {event.cpu}, Pull CPU: {event.pull_cpu}, Result: {event.result}")

bpf["events"].open_perf_buffer(print_event)

while True:
    try:
        bpf.perf_buffer_poll()
    except KeyboardInterrupt:
        exit()

