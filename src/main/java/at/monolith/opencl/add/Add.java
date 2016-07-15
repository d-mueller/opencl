package at.monolith.opencl.add;

import org.jocl.*;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import static org.jocl.CL.*;

/**
 * Created by dmueller on 7/14/16.
 *
 * Simple addition of two vectors using OpenCL and JOCL.
 *
 * Lot's of stuff taken from:
 * http://www.codeproject.com/Articles/86551/Part-Programming-your-Graphics-Card-GPU-with-Jav
 * http://www.jocl.org/samples/samples.html
 */
public class Add {

	public static void main(String args[])
	{
		// Create input- and output data
		int n = 10;
		float srcArrayA[] = new float[n];
		float srcArrayB[] = new float[n];
		float dstArray[] = new float[n];
		for (int i=0; i<n; i++)
		{
			srcArrayA[i] = i;
			srcArrayB[i] = i;
		}
		Pointer srcA = Pointer.to(srcArrayA);
		Pointer srcB = Pointer.to(srcArrayB);
		Pointer dst = Pointer.to(dstArray);

		// Kernel
		String programSource = readFile("kernels/Add.cl");


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
		cl_context context = clCreateContext(
				contextProperties, 1, new cl_device_id[]{device},
				null, null, null);

		// Create a command-queue for the selected device
		cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null);

		// Setup arguments for CL kernel (a, b and c).
		cl_mem memObjects[] = new cl_mem[3];
		// a and b (input)
		memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * n, srcA, null);
		memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * n, srcB, null);
		// c (output)
		memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * n, null, null);

		// Compile (or whatever) the kernel.
		cl_program program = clCreateProgramWithSource(context, 1, new String[]{ programSource }, null, null);
		clBuildProgram(program, 0, null, null, null, null);

		cl_kernel kernel = clCreateKernel(program, "add", null);

		// Set arguments
		clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memObjects[0]));
		clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memObjects[1]));
		clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memObjects[2]));

		long global_work_size[] = new long[]{n};
		long local_work_size[] = new long[]{1};

		// Run stuff!
		clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null);

		// Read stuff!
		clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, n * Sizeof.cl_float, dst, 0, null, null);

		System.out.println("Done! Checking result.");

		for (int i = 0; i < n; i++) {
			System.out.println(String.format("GPU: %f, CPU: %f", dstArray[i], srcArrayA[i] + srcArrayB[i]));
		}

	}

	/**
	 * Helper function which reads the file with the given name and returns
	 * the contents of this file as a String. Will exit the application
	 * if the file can not be read.
	 *
	 * @param fileName The name of the file to read.
	 * @return The contents of the file
	 */
	static String readFile(String fileName)
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
