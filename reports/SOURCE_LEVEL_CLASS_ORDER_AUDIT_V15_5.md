# Source-level class-order audit V15.5

For both datasets and seeds 42–51, original ER and DER++ `stream_summary`
orders, original MalCL `class_order.json`, the verified Anchored-HR deterministic
rule, and `numpy.random.RandomState(seed).permutation(100)` are identical.
Anchored-HR is explicitly marked as deterministic reconstruction when an order
was not serialized. There are zero mismatches. Supports were reconstructed
once per method from each validated order and source per-class supports; all
match the source task-support sequences and full test-set totals.
