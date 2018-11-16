#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    /* Wrapper function for GPU/CUDA error handling. Every CUDA call goes through
       this function. It will return a message giving your the error string,
       file name and line of the error. Aborts on error. */

    if (code != 0)
    {
        fprintf(stderr, "GPUAssert:: %s - %s (%d)\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}

// define a macro for accessing gpuAssert
#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__, true);}


__global__ void basic_math(float *in, float *z)
{
    //int xi = threadIdx.x;
    //int yi = threadIdx.y;
    //int i = yi + 1024 * xi;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    printf("%d\n", i);
	z[i] = in[i] * in[i] ;//+ in[i] * in[i] / in[i] *800;
    printf("z: %f  x: %f  i:%d\n",z[i],in[i],i);
}

int main()
{
    // Loop through the 3 files
    for (int fnum = 0; fnum < 3; ++fnum)
    {
        
        int i = fnum%3+1;
        //Read the file
        FILE *fp;
        int file_len = 1024;//757278936;
        float *input;
        float *d_input;

        int block_size, block_no;
        block_size = 1024;
        block_no = file_len/block_size;
# if __CUDA_ARCH__>=200
        printf("Should print\n");
#endif  
        input = (float*)malloc(file_len*sizeof(float));
        cudaMalloc(&d_input,file_len*sizeof(float));
        
        char file_name[30];
        sprintf(file_name, "fake_comb_%d.txt", i);
        printf("Opening %s, fnum: %d\n", file_name, fnum);
        fp = fopen(file_name, "r");
        printf("Reading %s, fnum: %d\n", file_name, fnum);
        for (int l = 0; l < file_len; ++l)
        {
            fscanf(fp, "%f", &input[l] );
            //fprintf(stderr, "%f\n", input[l]);
            //printf("%f\n", input[l]);
        }
        fclose(fp);


        // does some calcs
        printf("Calculating\n");
        float *out_values;
        float *d_out_values;

        out_values = (float*)malloc(file_len*sizeof(float));
        gpuErrchk(cudaMalloc(&d_out_values,file_len*sizeof(float)));

        //chuck the memory onto the gpu
        gpuErrchk(cudaMemcpy(d_input, input, file_len*sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_out_values, out_values, file_len*sizeof(float), cudaMemcpyHostToDevice));


        //dim3 dimBlock(block_size,1,1);
        //dim3 dimGrid(block_no,1,1);
        printf("Block no: %d    Block_size: %d\n", block_no, block_size);
        basic_math<<<block_no, block_size>>>(d_input, d_out_values);
        gpuErrchk(cudaThreadSynchronize());

        //move off gpu
        gpuErrchk(cudaMemcpy(input, d_input, file_len*sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(out_values, d_out_values, file_len*sizeof(float), cudaMemcpyDeviceToHost));

        // write to other files
        FILE* fo;
        char file_name_out[30];

        sprintf(file_name_out, "fake_fits_file_%d.txt", i);
        printf("Writing to %s\n", file_name_out);

        fo = fopen(file_name_out, "w");
        for (int l = 0; l < file_len; ++l)
        {
            fprintf(fo, "%f\n", out_values[l]);
            //printf("%f\n", out_values[l]);
        }
        fclose(fo);

        //free memory
        cudaFree(d_out_values);
        cudaFree(d_input);
        free(input);
        free(out_values);

        //free(fo);
        //free(fp);


    }
}
