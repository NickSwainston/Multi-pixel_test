#include <stddef.h>
#include <stdio.h>
#include <omp.h>

__global__ void basic_math(int *x, int *z)
{
	int i = threadIdx.x;
	z[i] = x[i]*i
}

int main()
{
    // Loop through the 3 files
    for (int i = 1; i < 4; ++i)
    {
        //Read the file
        FILE *fp;
        int file_len = 10;
        int buffer[file_len];
        
        char file_name[30];
        sprintf(file_name, "fake_comb_file_%d.txt", i);
        printf("Reading %s\n", file_name);
        fp = fopen(file_name, "r");
        for (int l = 0; l < file_len; ++l)
        {
            fscanf(fp, "%d", &buffer[l] );
            printf("%d\n", buffer[l]);
        }
        fclose(fp);


        // does some calcs
        printf("Calculating");
        int out_values[file_len];

        //dim3 threadsPerBlock(file_len);
        basic_math<<<1, file_len>>>(buffer, out_values);


        // write to other files
        FILE* fo;
        char file_name_out[30];

        sprintf(file_name_out, "fake_fits_file_%d.txt", i);
        printf("Writing to %s\n", file_name_out);

        fo = fopen(file_name_out, "w");
        for (int l = 0; l < file_len; ++l)
        {
            fprintf(fo, "%d\n", out_values[l]);
        }

    }
}
