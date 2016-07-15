#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void solve(
					__global double *u0,
					__global double *u1,
					const double c,
					const int n)
{
	int i = get_global_id(0);
	int iL = ((i - 1) % n + n) % n;
	int iR = ((i + 1) % n + n) % n;

	u1[i] = c * (u0[iL] + u0[iR] - 2 * u0[i]) + 2 * u0[i] - u1[i];
}