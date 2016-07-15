package at.monolith.opencl.waveeq;

import at.monolith.opencl.PerformanceTimer;
import org.jocl.*;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import static org.jocl.CL.*;

/**
 * Created by dmueller on 7/15/16.
 *
 * Numerical evolution of the 1D wave equation using OpenCL.
 */
public class Wave1D {

	private static cl_context context;
	private static cl_command_queue commandQueue;
	private static cl_kernel kernel1;
	private static cl_kernel kernel2;

	public static void main(String[] args) {
		initCL();
		kernel1 = loadProgram("kernels/Wave1D.cl", "solve");
		kernel2 = loadProgram("kernels/Wave1D.cl", "solve");

		// Setup arrays for the wave equation:
		// Two 1D arrays for two time steps
		int n = 1024 * 32;
		double[] u0 = new double[n];
		double[] u1 = new double[n];

		// Initial condition: peaklike disturbance
		for (int i = 0; i < n; i++) {
			u0[i] = 0.0;
			u1[i] = 0.0;
		}
		u0[n/2] = 1.0;
		u1[n/2] = 1.0;

		// Make a copy.
		double[] v0 = new double[n];
		double[] v1 = new double[n];
		System.arraycopy(u0, 0, v0, 0, n);
		System.arraycopy(u1, 0, v1, 0, n);

		// Simulation parameters
		// number of simulation steps
		int numSteps = 2 * 50000;
		// parameter of the discretized wave equation: (dt/dx * c)^2
		double p = 0.05;

		// Initialize pointers.
		Pointer u0p = Pointer.to(u0);
		Pointer u1p = Pointer.to(u1);

		// Create buffer objects from the arrays.
		cl_mem u0b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_double * n, u0p, null);
		cl_mem u1b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_double * n, u1p, null);

		// Set arguments for the kernel.
		clSetKernelArg(kernel1, 0, Sizeof.cl_mem, Pointer.to(u0b));
		clSetKernelArg(kernel1, 1, Sizeof.cl_mem, Pointer.to(u1b));
		clSetKernelArg(kernel1, 2, Sizeof.cl_double, Pointer.to(new double[]{p}));
		clSetKernelArg(kernel1, 3, Sizeof.cl_int, Pointer.to(new int[]{ n }));

		// Second kernel with switched buffers.
		clSetKernelArg(kernel2, 1, Sizeof.cl_mem, Pointer.to(u0b));
		clSetKernelArg(kernel2, 0, Sizeof.cl_mem, Pointer.to(u1b));
		clSetKernelArg(kernel2, 2, Sizeof.cl_double, Pointer.to(new double[]{p}));
		clSetKernelArg(kernel2, 3, Sizeof.cl_int, Pointer.to(new int[]{ n }));

		// Start timer.
		PerformanceTimer timer = new PerformanceTimer();
		timer.active = true;
		timer.reset();

		// Run the simulation!
		long global_work_size[] = new long[]{n};
		long local_work_size[] = new long[]{2};
		for (int i = 0; i < numSteps / 2; i++) {
			// Run kernel.
			clEnqueueNDRangeKernel(commandQueue, kernel1, 1, null, global_work_size, null, 0, null, null);
			clEnqueueNDRangeKernel(commandQueue, kernel2, 1, null, global_work_size, null, 0, null, null);
		}

		// Read the result.
		clEnqueueReadBuffer(commandQueue, u0b, CL_TRUE, 0, n * Sizeof.cl_double, u0p, 0, null, null);
		clEnqueueReadBuffer(commandQueue, u1b, CL_TRUE, 0, n * Sizeof.cl_double, u1p, 0, null, null);

		timer.lap("GPU");

		// Solve on the CPU.
		solve(v0, v1, p, n, numSteps);

		timer.lap("CPU");

		System.out.println("Done!");
	}

	/**
	 * Solves the discretized wave equation in 1D on the CPU.
	 *
	 * @param u0        array 1
	 * @param u1        array 2
	 * @param p         parameter of the wave equation
	 * @param n         size of the lattice
	 * @param numSteps  number of simulation steps to compute
	 */
	private static void solve(double[] u0, double[] u1, double p, int n, int numSteps) {
		// Copy the initial conditions.
		double[] v0 = new double[n];
		double[] v1 = new double[n];

		for (int i = 0; i < n; i++) {
			v0[i] = u0[i];
			v1[i] = u1[i];
		}

		// Solve!
		for (int i = 0; i < numSteps; i++) {
			// Solve one step.
			for (int j = 0; j < n; j++) {
				int jL = ((j - 1) % n + n) % n;
				int jR = ((j + 1) % n + n) % n;
				v1[j] = p * (v0[jL] + v0[jR] - 2 * v0[j]) + 2 * v0[j] - v1[j];
			}

			// Switch buffers.
			double[] temp = v0;
			v0 = v1;
			v1 = temp;
		}

		// Write results to u0 and u1;
		for (int i = 0; i < n; i++) {
			u0[i] = v0[i];
			u1[i] = v1[i];
		}
	}

	/**
	 * Loads a program from an external file and initializes the kernel.
	 *
	 * @param filename      file name of the kernel source
	 * @param programName   name of the kernel
	 */
	private static cl_kernel loadProgram(String filename, String programName) {
		// Load program from external file.
		String programSource = readFile(filename);

		// Compile (or whatever) the kernel.
		cl_program program = clCreateProgramWithSource(context, 1, new String[]{ programSource }, null, null);
		clBuildProgram(program, 0, null, null, null, null);

		return clCreateKernel(program, programName, null);
	}

	/**
	 * Sets up OpenCL context and command queue.
	 */
	private static void initCL() {
		// OpenCL stuff
		final int platformIndex = 0;
		final long deviceType = CL_DEVICE_TYPE_ALL;
		final int deviceIndex = 0;

		// Enable exceptions and subsequently omit error checks in this sample
		CL.setExceptionsEnabled(true);

		// Obtain the number of platforms
		int numPlatformsArray[] = new int[1];
		clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];

		// Obtain a platform ID
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(platforms.length, platforms, null);
		cl_platform_id platform = platforms[platformIndex];

		// Initialize the context properties
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

		// Obtain the number of devices for the platform
		int numDevicesArray[] = new int[1];
		clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];

		// Obtain a device ID
		cl_device_id devices[] = new cl_device_id[numDevices];
		clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		cl_device_id device = devices[deviceIndex];

		// Create a context for the selected device
		context = clCreateContext(
				contextProperties, 1, new cl_device_id[]{device},
				null, null, null);

		// Create a command-queue for the selected device
		commandQueue = clCreateCommandQueue(context, device, 0, null);
	}

	/**
	 * Helper function which reads the file with the given name and returns
	 * the contents of this file as a String. Will exit the application
	 * if the file can not be read.
	 *
	 * @param fileName The name of the file to read.
	 * @return The contents of the file
	 */
	private static String readFile(String fileName)
	{
		try
		{
			BufferedReader br = new BufferedReader(
					new InputStreamReader(new FileInputStream(fileName)));
			StringBuffer sb = new StringBuffer();
			String line = null;
			while (true)
			{
				line = br.readLine();
				if (line == null)
				{
					break;
				}
				sb.append(line).append("\n");
			}
			return sb.toString();
		}
		catch (IOException e)
		{
			e.printStackTrace();
			System.exit(1);
			return null;
		}
	}
}
