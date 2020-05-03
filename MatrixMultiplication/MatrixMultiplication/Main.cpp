#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define size_of_matrix 10

void printMatrix(int* arr, int n, int m) {
	int iterator1, iterator2;
	for (iterator1 = 0; iterator1 < n; ++iterator1) {
		for (iterator2 = 0; iterator2 < m; ++iterator2) {
			printf("%d ", arr[iterator1 * n + iterator2]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(void) {
	// Create the two input vectors
	const int inpLen = sizeof(int) * size_of_matrix * size_of_matrix;
	const int opLen = sizeof(int) * size_of_matrix * size_of_matrix;
	int M1[size_of_matrix * size_of_matrix], M2[size_of_matrix * size_of_matrix];

	for (int iterator1 = 0; iterator1 < size_of_matrix; iterator1++) {
		for (int iterator2 = 0; iterator2 < size_of_matrix; iterator2++) {
			M1[iterator1 * size_of_matrix + iterator2] = iterator1 * size_of_matrix + iterator2;
			M2[iterator1 * size_of_matrix + iterator2] = iterator1 * size_of_matrix + iterator2;
		}
	}

	printMatrix(M1, size_of_matrix, size_of_matrix);
	printMatrix(M2, size_of_matrix, size_of_matrix);

	// Load the kernel source code into the array source_str
	FILE* fp;
	char* source_str;
	size_t source_size;

	fopen_s(&fp, "matrix_kernel.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

	// Create memory buffers on the device for each vector 
	cl_mem M1_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, inpLen, NULL, &ret);
	cl_mem M2_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, inpLen, NULL, &ret);
	cl_mem M3_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, opLen, NULL, &ret);

	// Copy the lists A and B to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, M1_mem_obj, CL_TRUE, 0, inpLen, M1, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, M2_mem_obj, CL_TRUE, 0, inpLen, M2, 0, NULL, NULL);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "multiplicationModule", &ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&M1_mem_obj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&M2_mem_obj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&M3_mem_obj);

	size_t global_item_size[2];
	global_item_size[0] = size_of_matrix;
	global_item_size[1] = size_of_matrix;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, NULL, 0, NULL, NULL);

	// Read the memory buffer C on the device to the local variable C
	int M3[size_of_matrix*size_of_matrix];
	ret = clEnqueueReadBuffer(command_queue, M3_mem_obj, CL_TRUE, 0, opLen, M3, 0, NULL, NULL);

	// Display the result to the screen
	printf("M1 * M2 = \n");
	printMatrix(M3, size_of_matrix, size_of_matrix);

	return 0;
}