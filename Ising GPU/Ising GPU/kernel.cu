#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuda.h>
#include "book.h"
#include <stdlib.h>
#include <curand.h>
#include <time.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

# define CURAND_CALL(x) do { if ((x)!=CURAND_STATUS_SUCCESS ) {\
printf (" Error at % s :% d\ n" , __FILE__ , __LINE__ ) ; \
return EXIT_FAILURE ;}} while (0)

void wyswietl(int *sp, int const l)
{
	for (int i1 = 0; i1 < l; i1++)
	{
		for (int i2 = 0; i2 < l; i2++)
		{
			printf("%+d\t", sp[i2 + i1 * l]);
		}
		printf("\n");
	}
}

__global__ void setup_kernel(curandState *state) {
	int id = threadIdx.x + blockIdx.x * 64;

	/* Each thread gets same seed, a different sequence number , no offset */
	curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, int *result) {
	int id = threadIdx.x + blockIdx.x * 64;
	unsigned int x;
	/* Copy state to local memory for efficiency */
	curandState localState = state[id];

	/* Generate pseudo -random unsigned ints */
	for (int n = 0; n < 100000; n++) {
		x = curand(&localState);
	}

	/* Copy state back to global memory */
	state[id] = localState;

}



__global__ void krok(curandState *state, int *sp, int *par, int *magi, float *avmag, float *sus, float *hca, float *aven, float *worki) //krok próby, sprawdzenie przekazywania tablicy orgina³u
{

	int l = par[0];
	int n = par[1];
	int tmin = par[2];
	int tmax = par[3];
	int dmagn = par[4];

	int mag[100000];
	float work[8] = {0,0,0,0,0,0,0,0};

	int t0 = tmin + blockIdx.x;
	curandState localState = state[t0];
	
	if(t0<=(tmax))
	{	
	int w[5];
		for (int i = 0; i < 5; i++)
		{
			int s = i * 4 - 8;
			w[i] = (int)(exp(-(float)s / ((float)t0 / 100)) * 1000);
		}

		for (int i = 0; i < n; i++)
		{
			mag[i] = 0;
			for (int i0 = 0; i0 < l; i0++)
			{
				for (int i1 = 0; i1 < l; i1++)
				{
					int dE = 0;
					dE = 2 * sp[i0*l + i1] * (sp[((i0 - 1 + l) % l)*l + i1] + sp[((i0 + 1) % l)*l + i1] + sp[i0*l + (i1 - 1 + l) % l] + sp[i0*l + (i1 + 1) % l]);
					int x = curand(&localState)% 1000;
					if (w[(dE + 8) / 4] > x)
					{
						sp[i0*l + i1] = sp[i0*l + i1] * (-1);
						dE = -dE;
					}
					mag[i] += sp[i0*l + i1];
					work[1] += (float)(-dE/2);
					work[3] += (float)(-dE / 2);
				}
			}

	

			avmag[t0 - tmin] += abs(mag[i]);
			work[6] += (mag[i] * mag[i]);;//
			work[7] += abs(mag[i]);//
			work[2] += work[1] * work[1];//
			work[1] = 0;//
			work[4] += work[3];
			work[3] = 0;

		}
		avmag[t0 - tmin] = avmag[t0 - tmin] / (float)(n*l*l);

		work[6] = work[6]/n;
		work[7] = (work[7] / n)*(work[7] / n);
		sus[t0-tmin] = l * l*(work[6] - work[7]) * 100 / t0;
		work[6] = 0;
		work[7] = 0;

		work[2] = work[2] / n;
		aven[t0-tmin] = work[4] / n;
		work[4] = (work[4] / n)*(work[4] / n);
		hca[t0-tmin] = (work[2] - work[4]) * 100 * 100 / (t0 * t0 * l * l);//jeszcze zredukowane wartoœci do sprawdzenia czy nie brakuje podzielenia przez kb
		work[2] = 0;
		work[4] = 0;
	}
		/* Copy state back to global memory */
		state[t0] = localState;
}




int main(int argc, char *argv[])
{
	//dane podstawowe
	int l = 10;
	int const n = 50000;

	int tmin = 200;
	int tmax = 250;
	
	//dane pomocnicze
	int dmagn = 200;

	//tablice na obliczenia CPU
	int par[10] = {l, n, tmin, tmax, dmagn, 0, 0, 0, 0, 0};//tablica parametrów
	int *magn = (int*)malloc((n) * sizeof(int));//magnetyzacja dla ka¿dego kruku w 1 temp
	float *aven = (float*)malloc((tmax - tmin + 1) * sizeof(float));//œrednia energia od temp
	float *avemagn = (float*)malloc((tmax - tmin + 1) * sizeof(float));//œrednia magnetyzacja od temp
	float *susc = (float*)malloc((tmax - tmin + 1) * sizeof(float));//susceptibility (podatnoœæ magnetyczna)
	float *hcap = (float*)malloc((tmax - tmin + 1) * sizeof(float));//pojemnoœc cieplna (heat capacity)
	float *work = (float*)malloc((20) * sizeof(float));//mozna zrobic lokalnego worksheeta w funkcji 

	//zerowanie tablic CPU
	for (int i = 0; i < n; i++)
	{
		magn[i] = 0;
	}
	for (int i = 0; i < 20; i++)
	{
		work[i] = 0;
	}
	for (int i = 0; i < tmax - tmin + 1; i++)
	{
		avemagn[i] = 0;
		susc[i] = 0;
		hcap[i] = 0;
		aven[i] = 0;
	}

	//przestrzen CPU
	int *space = (int*)malloc((l * l) * sizeof(int));

	//generowanie konfiguracji poczatkowej
	srand(time(NULL));
	for (int i = 0; i < l*l; i++)
	{
		space[i] = (rand() % 2) * 2 - 1;
	}

	//tablice na GPU pod obliczenia
	int *dev_space, *dev_magn, *dev_par;
	float *dev_aven, *dev_avemagn, *dev_susc, *dev_hcap, *dev_work;
	
	cudaMalloc((void**)&dev_space, l * l * sizeof(int*));
	cudaMalloc((void**)&dev_par, 10 * sizeof(int*));
	cudaMalloc((void**)&dev_magn, n * sizeof(int*));

	cudaMalloc((void**)&dev_aven, (tmax - tmin + 1) * sizeof(float*));
	cudaMalloc((void**)&dev_avemagn, (tmax - tmin + 1) * sizeof(float*));
	cudaMalloc((void**)&dev_susc, (tmax - tmin + 1) * sizeof(float*));
	cudaMalloc((void**)&dev_hcap, (tmax - tmin + 1) * sizeof(float*));
	cudaMalloc((void**)&dev_work, 20 * sizeof(float*));


	cudaMemcpy(dev_space, space, l * l * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_par, par, 10 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_magn, magn, n * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_aven, aven, (tmax - tmin + 1) * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_avemagn, avemagn, (tmax - tmin + 1) * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_susc, susc, (tmax - tmin + 1) * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_hcap, hcap, (tmax - tmin + 1) * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_work, work, 20 * sizeof(float*), cudaMemcpyHostToDevice);




	//przygotowanie generatora curand

	long long total;
	curandState *devStates;
	int sampleCount = 0;//argument, który potem zrównoleglimy

	/* Allocate space for prng states on device */
	cudaMalloc((void **)&devStates, 64 * 64 * sizeof(curandState));

	/* Setup prng states */
	setup_kernel << <64, 64 >> > (devStates);

	//W³aœciwa funkcja j¹dra

	wyswietl(space, l);

	krok<<<64, 1 >>>(devStates, dev_space, dev_par, dev_magn, dev_avemagn, dev_susc, dev_hcap, dev_aven, dev_work);

	cudaMemcpy(space, dev_space, l * l * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(par, dev_par, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(magn, dev_magn, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(aven, dev_aven, (tmax - tmin + 1) * sizeof(float*), cudaMemcpyDeviceToHost);
	cudaMemcpy(avemagn, dev_avemagn, (tmax - tmin + 1) * sizeof(float*), cudaMemcpyDeviceToHost);
	cudaMemcpy(susc, dev_susc, (tmax - tmin + 1) * sizeof(float*), cudaMemcpyDeviceToHost);
	cudaMemcpy(hcap, dev_hcap, (tmax - tmin + 1) * sizeof(float*), cudaMemcpyDeviceToHost);
	cudaMemcpy(work, dev_work, 20 * sizeof(float*), cudaMemcpyDeviceToHost);



	
	for (int i = 0; i <= tmax - tmin; i++)
	{
		printf("%d\t%f\t%f\t%f\t%f\t%f\n", tmin, aven[i], susc[i], hcap[i], bind[i], avemagn[i]);
	}

	printf("\navmagn:\n%f", avemagn[0]);
	printf("\n\nsusc:\n%f", susc[0]);
	printf("\n%f", susc[1]);
	printf("\n%f", susc[2]);
	printf("\n\nhcap:\n%f", hcap[0]);
	printf("\n%f", hcap[2]);
	printf("\n%f", hcap[4]);
	printf("\n\navenergy:\n%f", hcap[5]);

	printf("\n");
	
	wyswietl(space, l);


	free(space);
	free(magn);
	free(avemagn);
	free(susc);
	free(hcap);
	free(work);
	free(aven);
	free(bind);

	CUDA_CALL(cudaFree(dev_space));
	CUDA_CALL(cudaFree(dev_par));
	CUDA_CALL(cudaFree(dev_magn));
	CUDA_CALL(cudaFree(dev_avemagn));
	CUDA_CALL(cudaFree(dev_aven));
	CUDA_CALL(cudaFree(dev_hcap));
	CUDA_CALL(cudaFree(dev_susc));
	CUDA_CALL(cudaFree(dev_work));
	CUDA_CALL(cudaFree(devStates));
	CUDA_CALL(cudaFree(dev_bind));

	return 0;
}