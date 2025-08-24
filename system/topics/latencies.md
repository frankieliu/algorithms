intra dc: 1ms
1ntra region (btw availability zones): 5-10ms
inter dc: 100ms
cross region: 100ms

---

CPU and GPU
- cache coherence
- data transfer
- kernel launch
- sync
- mem bw
Optimize data transfer
- async transfers
- zero-copy mem
Latency Hiding
- when thread stalls for data, GPU switches execution
- prefetching: fetch data before it is needed
Optimize Cache Utilization
- data locality
- tune for specific workloads
HW and SW Optimization
- monitoring and tuning
  MSI Afterburner
  Riva Tuner
- HBM
