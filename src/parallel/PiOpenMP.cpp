// Standard Library 
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// For OpenMP
#include <omp.h>

using namespace std;

int main(int argc, char**argv) {
    if(argc>1)
        omp_set_num_threads(atoi(argv[1]));

    int i = 0;     // Loop counter
	int count = 0; // Number of successful trials
    int niter;     // Total trials
    unsigned seed; // To store the seed state of each random number

    niter = 1000000000;

    double x; // X coordinate
    double y; // Y coordinate

    double start_time;
    double end_time;

    start_time = omp_get_wtime();
    #pragma omp parallel private(i,x,y,seed)
    {
        seed = 25234 + 17*omp_get_thread_num();
        #pragma omp parallel for reduction(+:count) schedule(static)
        for(i=0; i < niter; i++) {
            x = (double)rand_r(&seed)/RAND_MAX;
            y = (double)rand_r(&seed)/RAND_MAX;
            if(x*x + y*y <= 1.0)
                count++;
            //printf("Thread %d, Iter: %d\n",omp_get_thread_num(),i);
        }
    }
    end_time = omp_get_wtime();
    
    // Compute Pi
    double pi = 4.0*(double)count/(double)niter;
    printf("# of trials= %d , estimate of pi is %g, time= %f \n",niter,pi, end_time-start_time);
    return 0;
}

