
// CUDA runtime 
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>

// CUDA random number generator
#include <curand.h>
#include <curand_kernel.h>

// standard include
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <time.h>

// define parameters for host random number generation
#define RANKSZMAX 2147483648.0		// 2^31 constant for random number generation (long; 4 bytes +/-)
#define SEED  37418					// seed for host random number generation, also used for cuRAND (use different for different realizations)

// define the lattice size
#define DIM_SIZE 512	// linear size of lattice, should be divisible by 2xDIM_QUADRANT
#define DIM_QUADRANT 4  // linear size of lattice done by one thread

//define CUDA thread block dimensions (each block has 32x2 threads; optimized because a CUDA warp (threads that go parallel) contains 32 threads)
#define THREAD_BLOCK_X 32
#define THREAD_BLOCK_Y 2

// host random number generation
long ksz_num[256];
int ksz_i;
void seed(long seed) // seeds RNG
{
	ksz_num[0] = (long)fmod(16807.0*(double)seed,2147483647.0);
	for(ksz_i = 1; ksz_i<256; ksz_i++)
	{
		ksz_num[ksz_i] = (long)fmod(16807.0 * (double)ksz_num[ksz_i - 1],2147483647.0);
	}
}

long randl(long num) // number between 0 and num-1 (of type long)
{
	ksz_i = ++ksz_i & 255;
	ksz_num[ksz_i] = ksz_num[(ksz_i - 103) & 255] ^ ksz_num[(ksz_i - 250) & 255];
	ksz_i = ++ksz_i & 255;
	ksz_num[ksz_i] = ksz_num[(ksz_i - 30) & 255] ^ ksz_num[(ksz_i - 127) & 255];
	return(ksz_num[ksz_i] % num);
}

// creates the square lattice on host, stores in player_n
long L,SSIZE,i,j,iu,ju,player1,player2;
long player_n[DIM_SIZE*DIM_SIZE][4]; 
void squarelattice(void)
{
	for(i = 0; i<L; i++)
	{
		for(j = 0; j<L; j++)
		{
			player1 = L*j + i;

			iu = i + 1;
			ju = j;
			if(iu>(L - 1)) iu = iu - L;
			player2 = L*ju + iu;
			player_n[player1][0] = player2;

			iu = i;
			ju = j + 1;
			if(ju>(L - 1)) ju = ju - L;
			player2 = L*ju + iu;
			player_n[player1][1] = player2;

			iu = i - 1;
			ju = j;
			if(iu<0) iu = iu + L;
			player2 = L*ju + iu;
			player_n[player1][2] = player2;

			iu = i;
			ju = j - 1;
			if(ju<0) ju = ju + L;
			player2 = L*ju + iu;
			player_n[player1][3] = player2;
		}
	}
}

// creates coalesced lattice
long player_n_index_for_normal_in_coalesced[DIM_SIZE*DIM_SIZE];
long player_n_index_for_coalesced_in_normal[DIM_SIZE*DIM_SIZE];
long player_n_coalesced[DIM_SIZE*DIM_SIZE][4]; 
unsigned char player_s[DIM_SIZE*DIM_SIZE];
unsigned char player_s_coalesced[DIM_SIZE*DIM_SIZE];
void coalescelattice(void)
{
	// first create indexes
	long x = 0;
	for(long ii = 0; ii < DIM_QUADRANT; ii++)
	{
		for(i = 0; i < L; i += DIM_QUADRANT)
		{
			for(j = 0; j < L; j++)
			{
				player_n_index_for_normal_in_coalesced[x] = (i + ii) * L + j;
				player_n_index_for_coalesced_in_normal[(i + ii) * L + j] = x;
				x++;
			}
		}
	}

	// make the lattice
	for(i = 0; i < L*L; i++)
	{
		for(j = 0; j < 4; j++)
		{
			player_n_coalesced[i][j] = player_n_index_for_coalesced_in_normal[player_n[player_n_index_for_normal_in_coalesced[i]][j]];
		}
	}

	// copy strategies from the "original" lattice accordingly
	for(i = 0; i < L*L; i++)
	{
		player_s_coalesced[i] = player_s[player_n_index_for_normal_in_coalesced[i]];
	}
}

// decomposes the lattice into L / DIM_QUADRANT squares, to each be processed by a thread
long quadrant_n[(DIM_SIZE * DIM_SIZE) / (DIM_QUADRANT * DIM_QUADRANT)][9];
void decomposelattice(void)
{
	long qi = 0;
	long QL = L / DIM_QUADRANT;

	for(i = 0; i<QL; i++)
	{
		for(j = 0; j<QL; j++)
		{
			qi = QL*j + i;

			iu = i - 1;
			ju = j - 1;
			if(iu<0) iu = iu + QL;
			if(ju<0) ju = ju + QL;
			quadrant_n[qi][0] = QL*ju + iu;

			iu = i;
			ju = j - 1;
			if(ju<0) ju = ju + QL;
			quadrant_n[qi][1] = QL*ju + iu;

			iu = i + 1;
			ju = j - 1;
			if(iu>(QL - 1)) iu = iu - QL;
			if(ju<0) ju = ju + QL;
			quadrant_n[qi][2] = QL*ju + iu;

			iu = i - 1;
			ju = j;
			if(iu<0) iu = iu + QL;
			quadrant_n[qi][3] = QL*ju + iu;

			quadrant_n[qi][4] = QL*j + i;

			iu = i + 1;
			ju = j;
			if(iu>(QL - 1)) iu = iu - QL;
			quadrant_n[qi][5] = QL*ju + iu;

			iu = i - 1;
			ju = j + 1;
			if(iu<0) iu = iu + QL;
			if(ju>(QL - 1)) ju = ju - QL;
			quadrant_n[qi][6] = QL*ju + iu;

			iu = i;
			ju = j + 1;
			if(ju>(QL - 1)) ju = ju - QL;
			quadrant_n[qi][7] = QL*ju + iu;

			iu = i + 1;
			ju = j + 1;
			if(iu>(QL - 1)) iu = iu - QL;
			if(ju>(QL - 1)) ju = ju - QL;
			quadrant_n[qi][8] = QL*ju + iu;
		}
	}
}

// CUDA device constant memory variables
__device__ __constant__ long d_L,d_QPerLine,d_size;		// linear size of lattice,  L/DIM_QUADRANT, SSIZE/(DIM_QUADRANT * DIM_QUADRANT * 4);
__device__ __constant__ float d_R;						// multiplication factor
__device__ __constant__ float d_K;						// Fermi inverse temperature
__device__ __constant__ char Xoffset[4]{1,0,-1,0};		// offset for 1,2,3,4 updating of decomposed lattice
__device__ __constant__ char Yoffset[4]{0,1,0,-1};		// offset for 1,2,3,4 updating of decomposed lattice

// device random number generation
__global__ void seedcurand(const unsigned long seed,curandState *const state) // seeds RNG (each thread gets its own, but equal seed); run as kernel, hence __global__
{
	long ti = (blockDim.y * blockDim.x * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;

	if (ti<d_size) // only if L divisible by 64 is the number of threads exactly equal to the number of 4x4 blocks that make up 1/4 of the whole LxL lattice, otherwise we have a bit more threads, so need this if{} because ti is used for the indexing of the 4x4 blocks 
	{
		curand_init(seed,ti,0,&state[ti]);
	}
}

__device__ long d_randl(curandState *const state,long scope) // number between 0 and scope-1 (of type long); called only from kernel (the pgg main kernel), hence __device__
{
	return (curand(state) % scope);
}

__device__ float d_randd(curandState *const state) // double random number in  [0, 1); called only from kernel (the pgg main kernel), hence __device__
{
	return ((float)curand_uniform(state));
}

// device kernel for cunting strategies
__global__ void countstrats(long size, unsigned char *const player_s,long *const Spop)
{
	long ti = (blockDim.y * blockDim.x * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;

	if (ti<size)
	{
		atomicAdd((unsigned int*)&Spop[player_s[ti]],1);
	}
}

// CUDA main PGG kernel
__global__ void pggkernel(const char quadrantIdx,curandState *const state,long *const quadrant_n,long *const player_n,unsigned char *const player_s)
{
	// shared memory for lattice data with faster access
	__device__ __shared__ unsigned char player_s_shared[THREAD_BLOCK_X*THREAD_BLOCK_Y][DIM_QUADRANT + 6][DIM_QUADRANT + 6];

	// counters
	long ti,threadX,qi,qy,base_quadrant,p_offset,p1i,p2i,qix,qiy;
	unsigned char y,sy,nbh,n_nbh,nbsx,nbsy;
	unsigned char p1,p1x,p1y,p2,p2x,p2y;

	// payoffs of source and target players and the adaptation rate derived therefrom (via the Fermi function)	
	float P_source,P_target,adaptation_rate;

	// strategy holders and corresponding contributions (of neighbors and central players)
	unsigned char strat1,strat2;
	unsigned char contrib0,contrib;

	//thread index overall & thread index within a block
	ti = (blockDim.y * blockDim.x * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;
	threadX = blockDim.x * threadIdx.y + threadIdx.x;
	if (ti < d_size) // only if L divisible by 64 is the number of threads exactly equal to the number of 4x4 blocks that make up 1/4 of the whole LxL lattice, otherwise we have a bit more threads, so need this if{} because ti is used for the indexing of the 4x4 blocks 
	{
		//index and coordinates of the zero quadrant for picking the source player
		qix = (ti * 2 + (quadrantIdx % 2)) % d_QPerLine;
		qiy = 2 * ((ti * 2) / d_QPerLine) + (quadrantIdx / 2);
		qi = qiy * d_QPerLine + qix;

		// copy player_s data to shared memory
		sy = 0; y = 0;
		//if in corner quadrant, do copy strategies in each quadrant seperately (the else{} below could be omitted, see pgg4.cu, only slightly faster this way)
		if((qix == 0 || qix == (d_QPerLine - 1)) || (qiy == 0 || qiy == (d_QPerLine - 1)))
		{

			for (qy = 0; qy < 3; qy++)
			{
				(qy == 1) ? sy += DIM_QUADRANT : sy += 3;
				for(; y < sy; y++)
				{
					p_offset = ((y + (DIM_QUADRANT - 3)) % DIM_QUADRANT)  * d_L * d_QPerLine;
					base_quadrant = 9 * qi + qy * 3;
					memcpy(&player_s_shared[threadX][y][0],&player_s[DIM_QUADRANT * quadrant_n[base_quadrant] + p_offset + (DIM_QUADRANT - 3)],3 * sizeof(unsigned char));
					memcpy(&player_s_shared[threadX][y][3],&player_s[DIM_QUADRANT * quadrant_n[base_quadrant + 1] + p_offset],DIM_QUADRANT * sizeof(unsigned char));
					memcpy(&player_s_shared[threadX][y][3 + DIM_QUADRANT],&player_s[DIM_QUADRANT * quadrant_n[base_quadrant + 2] + p_offset],3 * sizeof(unsigned char));
				}
			}
		}
		else
		{
			for (qy = 0; qy < 3; qy++)
			{
				(qy == 1) ? sy += DIM_QUADRANT : sy += 3;
				for(; y < sy; y++)
				{
					p_offset = ((y + (DIM_QUADRANT - 3)) % DIM_QUADRANT) * d_L * d_QPerLine + (qiy - 1) * d_L + qix * DIM_QUADRANT;
					memcpy(&player_s_shared[threadX][y][0],&player_s[qy * d_L + p_offset - 3],(DIM_QUADRANT + 6) * sizeof(unsigned char));
				}
			}
		}

		for (int i = 0; i < DIM_QUADRANT*DIM_QUADRANT; i++) // elementary MC steps within a quadrant
		{
			p1 = d_randl(&state[ti],DIM_QUADRANT * DIM_QUADRANT);				// choose a source index within quadrant
			p1x = p1 % DIM_QUADRANT + 3;
			p1y = p1 / DIM_QUADRANT + 3;

			p2 = d_randl(&state[ti],4);
			p2x = p1x + Xoffset[p2];
			p2y = p1y + Yoffset[p2];;

			strat1 = player_s_shared[threadX][p1y][p1x];					// strategy of source 
			strat2 = player_s_shared[threadX][p2y][p2x];					// strategy of target

			if(strat1 != strat2)							// if different strategies, otherwise no need to do anything, search for another pair
			{
				// set to zero the payoff of source and target
				P_source = 0.0;
				P_target = 0.0;

				// calculate the payoff of source
				contrib0 = strat1;						// source contribution to the 0th game (group in which player1 is central)
				for(nbh = 0; nbh < 4; nbh++)
				{
					nbsx = p1x + Xoffset[nbh];
					nbsy = p1y + Yoffset[nbh];
					contrib = player_s_shared[threadX][nbsy][nbsx];		// neighbor's contribution to his/her own game, which is evaluated in what follows 					
					contrib0 += contrib;								// neighbor's contribution to 0th game

					for(n_nbh = 0; n_nbh < 4; n_nbh++)
					{
						contrib += player_s_shared[threadX][nbsy + Yoffset[n_nbh]][nbsx + Xoffset[n_nbh]];
					}
					P_source = P_source + ((d_R*contrib) / 5.0f) - strat1;
				}
				P_source = P_source + ((d_R*contrib0) / 5.0f) - strat1;

				// calculate the payoff of target
				contrib0 = strat2;						// target contribution to the 0th game (group in which player1 is central)
				for(nbh = 0; nbh < 4; nbh++)
				{
					nbsx = p2x + Xoffset[nbh];
					nbsy = p2y + Yoffset[nbh];
					contrib = player_s_shared[threadX][nbsy][nbsx];		// neighbor's contribution to his/her own game, which is evaluated in what follows 					
					contrib0 += contrib;								// neighbor's contribution to 0th game

					for(n_nbh = 0; n_nbh < 4; n_nbh++)
					{
						contrib += player_s_shared[threadX][nbsy + Yoffset[n_nbh]][nbsx + Xoffset[n_nbh]];
					}
					P_target = P_target + ((d_R*contrib) / 5.0f) - strat2;
				}
				P_target = P_target + ((d_R*contrib0) / 5.0f) - strat2;

				// finally, copy or not the startegy
				adaptation_rate = 1 / (1 + exp((P_target - P_source) / d_K));

				if(d_randd(&state[ti]) < adaptation_rate)
				{
					//update the target in the shared mem
					player_s_shared[threadX][p2y][p2x] = strat1;
					p1i = qi * DIM_QUADRANT + (p1y - 3) * d_L * d_QPerLine + (p1x - 3);	//source index within lattice
					p2i = player_n[p1i * 4 + p2];										//target index within lattice
					player_s[p2i] = strat1;												//update the target in the lattice					
				}
			} // if strategies different
		} // ends one full MCS per quadrant	
	}
}

//CUDA device memory pointers used in the main program
long *d_player_n;			// the square lattice, coalesced on device
long *d_quadrant_n;			// part of the square lattice (SSIZE / (DIM_QUADRANT*DIM_QUADRANT))*9; needed for threads to get strategies within shared memory player_s_shared (see pggkernel)
unsigned char *d_player_s;	// strategies on device
long* d_Spop;				// number of strategies on the device
curandState *d_state;		// RNG on device

// main program
int main(void)
{
	// system size and coalesced/decomposed lattice parameters
	L=DIM_SIZE;
	SSIZE=L*L;
	long QPerLine=L/DIM_QUADRANT;
	long size = SSIZE / (DIM_QUADRANT * DIM_QUADRANT * 4); // this is the actual number of all threads needed

	// game parameters, Fermi K and multiplication factor R
	float K=0.5;
	float R=3.98; 

	long stratCount[2];
	float FC,FD;

	// copy all above parameters to device memory; all declared __device__ __constant__ hence stored in constant memory space on the device for the entire simulation
	// these will be accessible to all the threads (regardless of block and grid) and to the host 
	cudaError_t err = cudaSuccess; 	// first initialize variable err to catch errors if any
	err = cudaMemcpyToSymbol(*(&d_L),&L,sizeof(long));
	err = cudaMemcpyToSymbol(*(&d_QPerLine),&QPerLine,sizeof(long));
	err = cudaMemcpyToSymbol(*(&d_K),&K,sizeof(float));
	err = cudaMemcpyToSymbol(*(&d_R),&R,sizeof(float));
	err = cudaMemcpyToSymbol(*(&d_size),&size,sizeof(long));

	// seed RNG for host
	seed(SEED);

	//set grid and block dimensions for our lattice (we use 32x2 2D blocks of threads, and 1D grids of blocks with so many elements as needed to accomodate 4*DIM_QUADRANT^2 threads (1,2,3,4 decomposition; see also Xoffset, Yoffset))
	dim3 dimBlock(THREAD_BLOCK_X, THREAD_BLOCK_Y, 1);
	dim3 dimGrid((int)ceil((SSIZE/(float)(DIM_QUADRANT*DIM_QUADRANT*4.0))/(float)(dimBlock.x*dimBlock.y)), 1, 1); // this could also be int since the other two dimensions are not needed
	dim3 dimGridcs((int)ceil(SSIZE/(float)(dimBlock.x*dimBlock.y)),1,1); // this is the dimGrid used just for counting strategies; basically every players gets its own thread

	//seed RNG for device; cuRAND	
	err = cudaMalloc(&d_state, size*sizeof(curandState));  // allocate memory for each player
	seedcurand << <dimGrid, dimBlock >> >(SEED, d_state); // run the kernel for seeding cuRAND
	
	// use host RNG for random initial state
	FILE* out=fopen("gridbegin.dat","w");
	for (i = 0; i<SSIZE; i++)
	{
		player_s[i]=(char)randl(2);	// 0: defector, 1: cooperator

		if((i % L)==0 && i>0)
		{
			fprintf(out,"\n");
		}
		fprintf(out,"%d ",(int)player_s[i]);
	}
	fclose(out);

	// create the square lattice on host, and the coalesced lattice, and decomposes the lattice, 
	squarelattice();
	coalescelattice();
	decomposelattice();

	// allocate memory for the lattice on device, also for strategy count (d_Spop)
	err = cudaMalloc(&d_player_n, sizeof(long) * 4 * SSIZE);
	err = cudaMalloc(&d_quadrant_n, sizeof(long) * (SSIZE / (DIM_QUADRANT*DIM_QUADRANT)) * 9);
	err = cudaMalloc(&d_player_s, sizeof(unsigned char) * SSIZE);
	err = cudaMalloc(&d_Spop,sizeof(long) * 2);

	// copy lattice data to device
	err = cudaMemcpy(d_player_n, player_n_coalesced, sizeof(long) * 4 * SSIZE, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_quadrant_n, quadrant_n, sizeof(long) * (SSIZE / (DIM_QUADRANT*DIM_QUADRANT)) * 9, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_player_s, player_s_coalesced, sizeof(unsigned char) * SSIZE, cudaMemcpyHostToDevice);

	// stopwatch
	time_t starttime, stoptime;
	time(&starttime);

	out=fopen("time.dat","w"); // open the file for the time series
	long mctot=250000; // number of full MCS
	for (long steps=0; steps<mctot; steps++) // full MCS main loop
	{
		for (char quadrantIdx = 0; quadrantIdx < 4; quadrantIdx++) // this goes over 1,2,3,4 to avoid conflict (could be done randomly)
		{
			pggkernel << <dimGrid, dimBlock >> >(quadrantIdx, d_state, d_quadrant_n, d_player_n, d_player_s); // run the main PGG kernel
		}

		if ((steps % 100) == 0) // every so many MCS do stats
		{
			// count player strategies on device and copy to host
			err = cudaMemset(d_Spop,0,sizeof(long)*2);
			countstrats << <dimGridcs,dimBlock >> >(SSIZE,d_player_s,d_Spop);
			err = cudaMemcpy(stratCount,d_Spop,sizeof(long)*2,cudaMemcpyDeviceToHost);

			// determine fractions and print to screen and file
			FD=stratCount[0]/(float)SSIZE;
			FC=stratCount[1]/(float)SSIZE;
			fprintf(out,"%ld %.8f %.8f %.8f %d %d\n",steps,FC,FD,FC + FD,stratCount[1],stratCount[0]);
			printf("%ld %.4f %.4f %.4f\t -> %d:%d\n",steps,FC,FD,FC + FD,stratCount[1],stratCount[0]);

			if (stratCount[1] == 0 || stratCount[0] == 0) break; // exit MCS main loop if either strategy dies out
		}
	}

	// stopwatch
	time(&stoptime);
	printf("\nDone in %.0f seconds\n", difftime(stoptime,starttime));

	fclose(out); // close file

	// free device memory
	cudaFree(d_player_n);
	cudaFree(d_player_s);
	cudaFree(d_quadrant_n);
	cudaFree(d_state);
	cudaFree(d_Spop);

	// make a snapshot of the lattice, first move from coalesced to "usual" lattice and then print out the matrix 
	for(i=0;i<SSIZE;i++)
	{
		player_s[i]=player_s_coalesced[player_n_index_for_coalesced_in_normal[i]];
	}
	out=fopen("gridend.dat","w");
	for(i=0; i<SSIZE; i++) // random initial distribution of strategies
	{
		if ((i % L)==0 && i>0)
		{
			fprintf(out,"\n");
		}
		fprintf(out,"%d ",(int)player_s[i]);
	}
	fclose(out);

	// print errors if any
	if (err!=cudaSuccess)
	{
		printf("\nFailed to execute properly (error: %s)\n",cudaGetErrorString(err));
		getchar();exit(EXIT_FAILURE);
	}

	getchar();return(0);
}