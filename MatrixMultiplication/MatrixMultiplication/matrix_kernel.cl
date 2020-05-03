
__kernel void multiplicationModule(__global const int *M1, __global const int *M2, __global int *M3)
{
   int iterator1 = get_global_id(0); 
   int iterator2 = get_global_id(1);
   int size_of_matrix = get_global_size(0);

   int iterator3, temp = 0;

   for (iterator3 = 0; iterator3 < size_of_matrix; iterator3++)   {

        temp += M1[iterator1 * size_of_matrix + iterator3] * M2[iterator3 * size_of_matrix + iterator2];
   
   }

   M3[iterator1 * size_of_matrix + iterator2] = temp;
}