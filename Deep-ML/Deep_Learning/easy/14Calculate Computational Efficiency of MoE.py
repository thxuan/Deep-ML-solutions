def compute_efficiency(n_experts, k_active, d_in, d_out):
    Dense = n_experts*d_in*d_out
    MoE = k_active*d_in*d_out
    return "{:.1%}".format( (Dense - MoE)/Dense )

print(compute_efficiency(1000, 2, 512, 512))