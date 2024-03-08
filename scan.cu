//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!
// note in this code until m=21 reduce technich is faster than parallel scan technic but after that
// the parrallel scan is faster
#include "scan.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpuerrors.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include "helper.h"


#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z


#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

#define gdx gridDim.x
#define gdy gridDim.y
#define gdz gridDim.z

__device__ uint8_t multiply(uint8_t a, uint8_t b, uint8_t* alpha_to, uint8_t* index_of)
{
	if (a == 0 || b == 0){
		return 0;
	}
	else{
		return alpha_to[(uint32_t(index_of[a]) + uint32_t(index_of[b])) % 255];
	}
}
__device__ void matrixVectorMultiplyKernel(uint8_t* matrix, uint8_t* vector, uint8_t* res, int dim, uint8_t* alpha_to, uint8_t* index_of) {
    for (int i=0; i<dim; i++){
		res[i] = 0;
		for (int j=0; j<dim; j++){
            //printf("[i]=%d ,[j]= %d, %d\n", i , j,  multiply(matrix[i * dim + j], vector[j], alpha_to, index_of));
            res[i] ^= multiply(matrix[i * dim + j], vector[j], alpha_to, index_of);
        }
        //printf("vector[i]=%d , %d\n", i , vector[i]);
    }
}
__device__ void vectorAddKernel(uint8_t* a, uint8_t* b, int dim) {

    for (int j=0; j<dim; j++){
        a[j] ^= b[j];
    }
    
}
__device__ void matrix_to_matrix_multiply(uint8_t* a, uint8_t* b, uint8_t* c, int dim, uint8_t* alpha_to, uint8_t* index_of){
    
    for (int i=0; i<dim; i++){
        for (int j=0; j<dim; j++){
            c[i*dim+j] = 0;
            for (int k=0; k<dim; k++){
                c[i*dim+j] ^= multiply(a[i*dim+k], b[k*dim+j], alpha_to, index_of);
            }
        }
    }

}

dim3 getDimGrid(const int n){
    // We need enough blocks to cover all cells of input vector so if all of them are fit into one block, we need one and if not
    // we need n/512 due to the maximum possible number of threads in each block.
    if (n <= 512)
    {
        dim3 dimGrid(1, 1, 1);
        return dimGrid;
    }
    else
    {   
        if (n <= (1<<24)){
            dim3 dimGrid(n / 512, 1, 1);
            return dimGrid;
        }
        else{
            dim3 dimGrid(32768, n/(32768*512), 1);
            return dimGrid;
        }
    }
}
dim3 getDimBlock(const int n)
{
    // If all of them are fit into one block, we need one block with enough threads and if not 
    // 512 threads are specified due to the maximum possible number of threads in each block.
    if (n <= 512)
    {
        dim3 dimBlock(n, 1, 1);
        return dimBlock;
    }
    else
    {
        dim3 dimBlock(512, 1, 1);
        return dimBlock;
    }
}
__global__ void kernelFuncred(uint8_t* a, uint8_t* matrix, uint8_t* d_temp, uint8_t* e_d, const int m, const int n, const int tav, uint8_t* alpha_to, uint8_t* index_of)
{
    // This kernel implements scan algorithm in a way that in the end we have scan of each block (512 cells) 
    int n1;
    if(n>512)
        n1=512;
    else
        n1=n;
    __shared__ uint8_t Mds[2048];
    __shared__ uint8_t alpha_tos[256];
    __shared__ uint8_t index_ofs[256];
    __shared__ uint8_t matrixs[16];
    for(int i=0;i<16;i++){
        matrixs[i]=matrix[i];
        //printf("i =%d, mat= %d",i,matrixpow1[i] );
    }
    for(int i=0;i<256;i++){
        alpha_tos[i]=alpha_to[i];
        index_ofs[i]=index_of[i];
    }
    //printf("bx0=%d , tav= %d\n",bx, tav);

    int r = by * 32768 * 512 +bx  * 512 + tx;

    Mds[4 * tx] = a[4 * r]; // Copy to shared Memory
    Mds[4 * tx + 1] =  a[4 * r + 1];
    Mds[4 * tx + 2] =  a[4 * r + 2];
    Mds[4 * tx + 3] =  a[4 * r + 3];
    // ad[r] = tx + 1;  // For debug purposes.
    __syncthreads();
    uint8_t matrixpow1[16];
    uint8_t matrixpow2[16];
    for(int i=0;i<16;i++){
        matrixpow1[i]=matrixs[i];
        matrixpow2[i]=matrixs[i];
    }
    __syncthreads();
    int pow1;
    if(tav==0){
        pow1=1;
    }
    else if(tav==1){
        pow1=512;
    }
    else{
        pow1=512*512;
    }
    //cudaMalloc(&matrixpow, 16 * sizeof(uint8_t))
    // In this part the first stage of the tree in Blelloch algorithm is calculated.
    int j = 2;
    for (; j <= n1; j = j * 2)
    {   //printf("j=%d\n" ,j);
        if ((r + 1) % (j) == 0){
            for(int pow=1; pow<=(j/2*pow1); pow=pow*2){
                if(pow==1){
                    for(int i=0;i<16;i++){
                    matrixpow1[i]=matrix[i];
                    }
                }
                else if(pow==2){
                    matrix_to_matrix_multiply(matrixs, matrixs, matrixpow1, 4, alpha_tos, index_ofs);
                    for(int i=0;i<16;i++){
                        matrixpow2[i]=matrixpow1[i];
                    }
                }
                else{
                    matrix_to_matrix_multiply(matrixpow2, matrixpow2, matrixpow1, 4, alpha_tos, index_ofs);
                    for(int i=0;i<16;i++){
                        matrixpow2[i]=matrixpow1[i];
                    }
                }
                //if(pow==(j/2*pow1)){printf("pow =%d , j= %d \n", pow, j);}
            }
            //__syncthreads();
            //printf("mds tx =%d , tx-j/2=%d , %d \n" ,tx, tx-j/2, Mds[4*(tx-j/2)]);
            uint8_t d_temp1[4];
            d_temp1[0]=Mds[tx*4];
            d_temp1[1]=Mds[tx*4+1];
            d_temp1[2]=Mds[tx*4+2];
            d_temp1[3]=Mds[tx*4+3];
            matrixVectorMultiplyKernel(matrixpow1, &(Mds[4*(tx-j/2)]), &(Mds[4*tx]), 4, alpha_tos, index_ofs);
            //printf("r=%d , j=%d\n",r ,j);
            vectorAddKernel( &(Mds[tx*4]),  d_temp1, 4);
            //printf("HEllo2");

        }
        //printf("HEllo1");
        __syncthreads();
    }

    // Replace the last element in each block by 0 and also store it for later. This is done in the last thread of 
    // each block and is dependent on the number of threads.
    /*uint8_t e[4];
    //HANDLE_ERROR(cudaMalloc(&e, 4 * sizeof(uint8_t)));
    if (n >= 512 && tx == 1023)
    {
        e[0] = Mds[4*tx];
        e[1] = Mds[4*tx + 1];
        e[2] = Mds[4*tx + 2];
        e[3] = Mds[4*tx + 3];
        Mds[4*tx] = 0;
        Mds[4*tx + 1] = 0;
        Mds[4*tx + 2] = 0;
        Mds[4*tx + 3] = 0;
    }
    else if (n < 512 && tx == n - 1)
    {
        e[0] = Mds[4*tx];
        e[1] = Mds[4*tx + 1];
        e[2] = Mds[4*tx + 2];
        e[3] = Mds[4*tx + 3];
        Mds[4*tx] = 0;
        Mds[4*tx + 1] = 0;
        Mds[4*tx + 2] = 0;
        Mds[4*tx + 3] = 0;
    }
    __syncthreads();

    // In this part the second stage of the tree in Blelloch algorithm is calculated.
    j = n;
    for (; j >= 2; j = j / 2)
    {
        if ((r + 1) % j == 0)
        {
            uint8_t t1,t2,t3,t4;
            t1 = Mds[4*(tx)];
            t2 = Mds[4*(tx) + 1];
            t3 = Mds[4*(tx) + 2];
            t4 = Mds[4*(tx) + 3];
            for(int pow=1; pow<j/2; pow++){
                matrix_to_matrix_multiply(matrix, matrix, matrixpow, 4, alpha_to, index_of)
            }
            matrixVectorMultiplyKernel(matrixpow, Mds[4*(tx)], d_temp, 4, alpha_to, index_of)
            vectorAddKernel(Mds[(tx-j/2)*4] , d_temp, 4)
            Mds[4*tx] = Mds[4*(tx-j/2)];
            Mds[4*tx + 1] = Mds[4*(tx-j/2) + 1];
            Mds[4*tx + 2] = Mds[4*(tx-j/2) + 2];
            Mds[4*tx + 3] = Mds[4*(tx-j/2) + 3];
            Mds[4*(tx-j/2)] = t1;
            Mds[4*(tx-j/2) + 1] = t2;
            Mds[4*(tx-j/2) + 2] = t3;
            Mds[4*(tx-j/2) + 3] = t4;
        }
        __syncthreads();
    }*/
    
    // For converting exclusive scan to inclusive scan, we need to have a shift to left and then the last element
    // should be replaced by total sum.
    //11
    /*if (tx >= 1){
        a[4*(r - 1)] = Mds[4*tx];
        a[4*(r - 1) + 1] = Mds[4*tx + 1];
        a[4*(r - 1) + 2] = Mds[4*tx + 2];
        a[4*(r - 1) + 3] = Mds[4*tx + 3];
    } 
    if (n >= 512 && tx == 1023){
        a[4*r] = e[0];
        a[4*r + 1] = e[1];
        a[4*r + 2] = e[2];
        a[4*r + 3] = e[3];
    }
    else if (n < 512 && tx == n - 1){
        a[4*r] = e[0];
        a[4*r + 1] = e[1];
        a[4*r + 2] = e[2];
        a[4*r + 3] = e[3];
    }*/    

    a[4 * r] = Mds[4*tx];
    a[4 * r + 1] = Mds[4*tx + 1];
    a[4 * r + 2] = Mds[4*tx + 2];
    a[4 * r + 3] = Mds[4*tx + 3];
    __syncthreads();
    // The last element of each block shoud be stored as a output in order to add them to the next blocks. 
    // So that the scan algorithm is completed.
    if (tx == 512-1)
    {
        e_d[4 * (by * 32768 +bx)] = a[4 * r];
        e_d[4 * (by * 32768 +bx) + 1] = a[4 * r + 1];
        e_d[4 * (by * 32768 +bx) + 2] = a[4 * r + 2];
        e_d[4 * (by * 32768 +bx) + 3] = a[4 * r + 3];
    }

    //HANDLE_ERROR(cudaFree(matrixpow));
}
__global__ void kernelFunc1red(uint8_t* a, uint8_t* matrix, uint8_t* d_temp, uint8_t* e_d, const int m, const int n, const int tav, const int ifstat, uint8_t* alpha_to, uint8_t* index_of)
{
    // This kernel is used to add sum of all elements in previous blocks in order to complete scan algrorithm and
    // compensate the effect of having seperate blocks in kernelFunc
    //printf("HElloker1");
    int n1;
    if(n>512)
        n1=512;
    else
        n1=n;
    __shared__ uint8_t Mds[2048];
    __shared__ uint8_t alpha_tos[256];
    __shared__ uint8_t index_ofs[256];
    __shared__ uint8_t matrixs[16];
    for(int i=0;i<16;i++){
        matrixs[i]=matrix[i];
        //printf("i =%d, mat= %d",i,matrixpow1[i] );
    }
    for(int i=0;i<256;i++){
        alpha_tos[i]=alpha_to[i];
        index_ofs[i]=index_of[i];
    }
    //int tx = threadIdx.x;
    //int bx = blockIdx.x;
    //printf("bx=%d\n",bx);
    int r = by * 32768 * 512 + bx * 512 + tx;
    Mds[4 * tx] = a[4 * r]; // Copy to shared Memory
    Mds[4 * tx + 1] =  a[4 * r + 1];
    Mds[4 * tx + 2] =  a[4 * r + 2];
    Mds[4 * tx + 3] =  a[4 * r + 3];
    // ad[r] = tx + 1;  // For debug purposes.
    __syncthreads();
    //uint8_t e[4];
    //HANDLE_ERROR(cudaMalloc(&e, 4 * sizeof(uint8_t)));
    if(tav==ifstat){
        if (n >= 512 && tx == 512-1){
            d_temp[0] = Mds[4*tx];
            d_temp[1] = Mds[4*tx + 1];
            d_temp[2] = Mds[4*tx + 2];
            d_temp[3] = Mds[4*tx + 3];
            Mds[4*tx] = 0;
            Mds[4*tx + 1] = 0;
            Mds[4*tx + 2] = 0;
            Mds[4*tx + 3] = 0;
        }
        else if (n < 512 && tx == n - 1){
            d_temp[0] = Mds[4*tx];
            d_temp[1] = Mds[4*tx + 1];
            d_temp[2] = Mds[4*tx + 2];
            d_temp[3] = Mds[4*tx + 3];
            Mds[4*tx] = 0;
            Mds[4*tx + 1] = 0;
            Mds[4*tx + 2] = 0;
            Mds[4*tx + 3] = 0;
        }
    }
    __syncthreads();
    uint8_t matrixpow1[16];
    uint8_t matrixpow2[16];
    for(int i=0;i<16;i++){
        matrixpow1[i]=matrixs[i];
        matrixpow2[i]=matrixs[i];
    }
    
    __syncthreads();
     int pow1;
    if(tav==0){
        pow1=1;
    }
    else if(tav==1){
        pow1=512;
    }
    else{
        pow1=512*512;
    }
    // In this part the second stage of the tree in Blelloch algorithm is calculated.
    int j = n1;
    for (; j >= 2; j = j / 2){
        if ((r + 1) % j == 0){
            uint8_t t1,t2,t3,t4;
            t1 = Mds[4*(tx)];
            t2 = Mds[4*(tx) + 1];
            t3 = Mds[4*(tx) + 2];
            t4 = Mds[4*(tx) + 3];
            for(int pow=1; pow<=(j/2*pow1); pow=pow*2){
                if(pow==1){
                    for(int i=0;i<16;i++){
                    matrixpow1[i]=matrix[i];
                    }
                }
                else if(pow==2){
                    matrix_to_matrix_multiply(matrixs, matrixs, matrixpow1, 4, alpha_tos, index_ofs);
                    for(int i=0;i<16;i++){
                        matrixpow2[i]=matrixpow1[i];
                    }
                }
                else{
                    matrix_to_matrix_multiply(matrixpow2, matrixpow2, matrixpow1, 4, alpha_tos, index_ofs);
                    for(int i=0;i<16;i++){
                        matrixpow2[i]=matrixpow1[i];
                    }
                }
                //if(pow==(j/2*pow1)&& tav==2){printf("ker1  pow =%d , j= %d \n", pow, j);                }
            }
            uint8_t d_temp1[4];
            matrixVectorMultiplyKernel(matrixpow1, &(Mds[4*(tx)]), d_temp1, 4, alpha_tos, index_ofs);
            vectorAddKernel(&(Mds[(tx-j/2)*4]) , d_temp1, 4);
            Mds[4*tx] = Mds[4*(tx-j/2)];
            Mds[4*tx + 1] = Mds[4*(tx-j/2) + 1];
            Mds[4*tx + 2] = Mds[4*(tx-j/2) + 2];
            Mds[4*tx + 3] = Mds[4*(tx-j/2) + 3];
            Mds[4*(tx-j/2)] = t1;
            Mds[4*(tx-j/2) + 1] = t2;
            Mds[4*(tx-j/2) + 2] = t3;
            Mds[4*(tx-j/2) + 3] = t4;
        }
        __syncthreads();
    }
    /*if(tav==0){
        if (r >= 1){
            e_d[4*(r - 1)] = Mds[4*tx];
            e_d[4*(r - 1) + 1] = Mds[4*tx + 1];
            e_d[4*(r - 1) + 2] = Mds[4*tx + 2];
            e_d[4*(r - 1) + 3] = Mds[4*tx + 3];

        } 
        else if (r==n-1){
            e_d[4 * r] = d_temp[0];
            e_d[4 * r + 1] = d_temp[1];
            e_d[4 * r + 2] = d_temp[2];
            e_d[4 * r + 3] = d_temp[3];
        }
    }
    else{*/
        a[4 * r ] = Mds[4 * tx];
        a[4 * r + 1] = Mds[4 * tx + 1];
        a[4 * r + 2] = Mds[4 * tx + 2];
        a[4 * r + 3] = Mds[4 * tx + 3];
    
    __syncthreads();

    if(tav!=0){
        e_d[4 * (r*512+512-1)] = a[4 * r];
        e_d[4 * (r*512+512-1) + 1] = a[4 * r + 1];
        e_d[4 * (r*512+512-1) + 2] = a[4 * r + 2];
        e_d[4 * (r*512+512-1) + 3] = a[4 * r + 3];
    }
    __syncthreads();
}
__global__ void kernelFunc2red(uint8_t* a,  uint8_t* d_temp, uint8_t* c, const int n)
{
    int r =by * 32768 * 512+ bx * 512 + tx;
    if (r >= 1){
        c[4 * (r - 1)] = a[4 * r];
        c[4 * (r - 1) + 1] = a[4 * r + 1];
        c[4 * (r - 1) + 2] = a[4 * r + 2];
        c[4 * (r - 1) + 3] = a[4 * r + 3];

    } 
    if (r == n-1){
        c[4 * r] = d_temp[0];
        c[4 * r + 1] = d_temp[1];
        c[4 * r + 2] = d_temp[2];
        c[4 * r + 3] = d_temp[3];
    }
}
__global__ void kernelFunc( uint8_t* c , const uint8_t* const a , const uint8_t* const A , int i , const int n , uint8_t* alpha_to , uint8_t* index_of_g )
{

    int num = (by*gdx*bdx)+(bx*bdx)+(tx);
    __shared__ uint8_t alpha_tos[256];
    __shared__ uint8_t index_ofs[256];
    __shared__ uint8_t MAT[16];
    /*Mds[4 * tx] = a[4 * r]; // Copy to shared Memory
    Mds[4 * tx + 1] =  a[4 * r + 1];
    Mds[4 * tx + 2] =  a[4 * r + 2];
    Mds[4 * tx + 3] =  a[4 * r + 3];*/
    if( tx< 256)
    {
        alpha_tos[tx] = alpha_to[tx];
        index_ofs[tx] = index_of_g[tx];
        if((tx) <16){
            MAT[tx] = A[tx];
        }
    }
    __syncthreads();
    //printf("bx0=%d , tav= %d\n",bx, tav);


    if( num <  (n<<2) ) 
    {
        //condition shift wiht the num
        if( num >= (1<<(i+2)) )
        {
            int co = num % 4;
            
            int v = (num - (1<<(i+2))) - co;; 

            uint8_t temp2[4];
            //evry column
            for(int in = 0 ; in < 4 ; ++in)
            {
                temp2[in] = a[ ( v + in ) ];
            }
            uint8_t temp = 0;
            for(int j = 0 ; j < 4 ; ++j)
            { //again multiply
                if( MAT[co * 4 + j]== 0 || temp2[j]== 0){
                    temp ^= 0;
                    //printf("hello if")
                }
                else{
                    temp ^= alpha_tos[(uint32_t(index_ofs[temp2[j]]) + uint32_t(index_ofs[MAT[co * 4 + j]]))%255];
                    //printf("hello else j=%d ", j)
                }
            }
            c[num] ^= temp;
        }
    }
}
__global__ void kernelFunc2(  uint8_t* c , uint8_t* a, int i , int n)
{   
    int r1 = by * gdx * bdx + bx * bdx + tx;//the thread
    if(r1 < (n<<2) && r1 >= (1<<(i+2)))
    {
        a[r1] = c[r1];
    }
}
__global__ void kernelFunc3(  uint8_t* a , uint8_t* alpha_to , uint8_t* index_of )
{
 __shared__ uint8_t Mat[4][4];

 __shared__ uint8_t MAs[4][4];

  Mat[ty][tx] = a[ty * 4 + tx];

  __syncthreads();

  MAs[ty][tx] = 0 ;
  
  for(int i = 0 ; i < 4 ; ++i){
	//multiply function
	if (Mat [i][tx]==0 || Mat[ty][i] == 0){
		MAs [ty][tx] ^= 0 ;
	}
    else{
	    MAs[ty][tx] ^= alpha_to[(uint32_t(index_of[Mat[i][tx]]) + uint32_t(index_of[Mat[ty][i]]))%255];
	}
    //printf("hello i = %d\n" , i)
  }
	a[ty*4 + tx] = MAs[ty][tx];
}

void gpuKernel(  const uint8_t* const a, const uint8_t* const matrix, uint8_t* c, const int m, const int n, uint8_t* alpha_to, uint8_t* index_of)
{
     // Initialize the GPU and copy data to it.
    //int size = 4 * n * sizeof(uint8_t);
    uint8_t* d_a;
 	uint8_t* d_c;
	uint8_t* d_matrix;
	uint8_t*  Alphato;
	uint8_t* INdexof;
    //uint8_t* e_d;
    //uint8_t* e_d2;
    HANDLE_ERROR(cudaMalloc((void**)&d_a , 4 * n * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_c , 4 * n * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_matrix , 4 * 4 * sizeof(uint8_t)));
	HANDLE_ERROR(cudaMalloc((void**)&Alphato , 256 * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&INdexof , 256 * sizeof(uint8_t)));
	HANDLE_ERROR(cudaMemcpy(d_a , a , 4 * n * (sizeof(uint8_t)) , cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_c , a , 4 * n * (sizeof(uint8_t)) , cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy( Alphato , alpha_to , 256 * (sizeof(uint8_t)) , cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(INdexof , index_of , 256 * (sizeof(uint8_t)) , cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_matrix  , matrix , 4 * 4 * (sizeof(uint8_t)) , cudaMemcpyHostToDevice));
	/*if(n>=512){
    HANDLE_ERROR(cudaMalloc(&e_d, 4 * (n / 512) * sizeof(uint8_t)));}
    if(n>=512*512){
    HANDLE_ERROR(cudaMalloc(&e_d2, 4 * (n / (512*512)) * sizeof(uint8_t)));}*/
    dim3 dimBlock(1024);
    dim3 dimGrid(512 , 512);
	dim3 block ( 4 , 4  ); 
	float fn = n ;
	float step = log2(fn);
    // Call the kernel function.

     /*if (n < 512)
            kernelFunc<<<dimGrid, dimBlock>>>(d_a, d_matrix, d_temp, e_d, m, n,0,alpha_to, index_of);
        else
            kernelFunc<<<dimGrid, dimBlock>>>(d_a, d_matrix, d_temp, e_d,m,512,0,alpha_to , index_of);
    */
	for(int i = 0 ; i < step ; ++i)
	{
        //printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
		kernelFunc<<< dimGrid , dimBlock >>>(d_c , d_a , d_matrix , i , n , Alphato , INdexof ) ;
		//cudaDeviceSynchronize();
        kernelFunc2<<< dimGrid , dimBlock >>>(d_c , d_a , i , n );
		//cudaDeviceSynchronize();
        kernelFunc3<<< 1 , block >>>( d_matrix , Alphato , INdexof );
        //cudaDeviceSynchronize();
    }
    /*if (n <= 512){
        dimGrid = getDimGrid(n);
        dimBlock = getDimBlock(n);
        //printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
        kernelFuncred<<<dimGrid, dimBlock>>>(d_a, d_matrix, d_temp, e_d, m, n, 0, Alphato, INdexof);
        //cudaDeviceSynchronize();
        kernelFunc1red<<<dimGrid, dimBlock>>>(d_a, d_matrix, d_temp, d_c, m, n, 0, 0,Alphato, INdexof);
        //cudaDeviceSynchronize();
        kernelFunc2red<<<dimGrid, dimBlock>>>(d_a, d_temp, d_c, n);
        //cudaDeviceSynchronize();
    }
    else{
        if (n<=512 *512){
            dimGrid = getDimGrid(n);
            dimBlock = getDimBlock(n);
            //printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
            kernelFuncred <<< dimGrid , dimBlock >>> (d_a, d_matrix, d_temp, e_d, m, n, 0, Alphato, INdexof);
            //cudaDeviceSynchronize();
            dimGrid = getDimGrid(n/512);
            dimBlock = getDimBlock(n/512);
            //printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
            kernelFuncred<<<dimGrid, dimBlock>>>(e_d, d_matrix, d_temp, e_d2, m, n/512, 1, Alphato, INdexof);
            //cudaDeviceSynchronize();
            kernelFunc1red<<<dimGrid, dimBlock>>>(e_d, d_matrix, d_temp, d_a, m, n/512, 1, 1, Alphato, INdexof);
            //cudaDeviceSynchronize();
            dimGrid = getDimGrid(n);
            dimBlock = getDimBlock(n);
            kernelFunc1red<<<dimGrid, dimBlock>>>(d_a,  d_matrix, d_temp, d_c, m, n, 0, 1, Alphato, INdexof);
            //printf("n=%d\t", n);
            //cudaDeviceSynchronize();
            kernelFunc2red<<<dimGrid, dimBlock>>>(d_a, d_temp, d_c, n);
            //cudaDeviceSynchronize();
        }
        else{
            dimGrid = getDimGrid(n);
            dimBlock = getDimBlock(n);
            //printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
            kernelFuncred<<<dimGrid, dimBlock>>>(d_a, d_matrix, d_temp, e_d, m, n, 0, Alphato, INdexof);
            //cudaDeviceSynchronize();
            dimGrid = getDimGrid(n/512);
            dimBlock = getDimBlock(n/512);
            kernelFuncred<<<dimGrid, dimBlock>>>(e_d, d_matrix, d_temp, e_d2, m, n/512, 1, Alphato, INdexof);
            //cudaDeviceSynchronize();
            dimGrid = getDimGrid(n/(512*512));
            dimBlock = getDimBlock(n/(512*512));
            kernelFuncred<<<dimGrid, dimBlock>>>(e_d2, d_matrix, d_temp, e_d2, m, n/(512*512), 2, Alphato, INdexof);
            //cudaDeviceSynchronize();
            kernelFunc1red<<<dimGrid, dimBlock>>>(e_d2, d_matrix, d_temp, e_d, m, n/(512*512), 2, 2, Alphato, INdexof);
            //cudaDeviceSynchronize();
            dimGrid = getDimGrid(n/512);
            dimBlock = getDimBlock(n/512);
            kernelFunc1red<<<dimGrid, dimBlock>>>(e_d, d_matrix, d_temp, d_a, m, n/512, 1, 2, Alphato, INdexof);
            //cudaDeviceSynchronize();
            dimGrid = getDimGrid(n);
            dimBlock = getDimBlock(n);
            kernelFunc1red<<<dimGrid, dimBlock>>>(d_a, d_matrix, d_temp, e_d, m, n, 0, 2, Alphato, INdexof);
            //cudaDeviceSynchronize();
            kernelFunc2red<<<dimGrid, dimBlock>>>(d_a, d_temp, d_c, n);
            //cudaDeviceSynchronize();
        }
    }*/
    // Copy the result back to the host.
    //printf("n=%d",(int)(sizeof(c)));

	HANDLE_ERROR(cudaMemcpy(c , d_c , 4 * n * sizeof(uint8_t) , cudaMemcpyDeviceToHost));

    // Free the memory on the GPU.

	HANDLE_ERROR(cudaFree(d_a));
	HANDLE_ERROR(cudaFree(d_c));
	HANDLE_ERROR(cudaFree(d_matrix));
	HANDLE_ERROR(cudaFree(INdexof));
	HANDLE_ERROR(cudaFree(Alphato));
    /*if(n>512){
    HANDLE_ERROR(cudaFree(e_d));}
    if(n>512*512){
    HANDLE_ERROR(cudaFree(e_d2));}*/
 }






