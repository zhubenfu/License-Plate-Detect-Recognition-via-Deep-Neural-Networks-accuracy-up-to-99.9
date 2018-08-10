#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <windows.h>
LARGE_INTEGER m_liPerfFreq;
LARGE_INTEGER m_liPerfStart;
LARGE_INTEGER liPerfNow;
double dfTim;
void getStartTime()
{
	QueryPerformanceFrequency(&m_liPerfFreq);
	QueryPerformanceCounter(&m_liPerfStart);
}

void getEndTime()
{
	QueryPerformanceCounter(&liPerfNow);
	dfTim=( ((liPerfNow.QuadPart - m_liPerfStart.QuadPart) * 1000.0f)/m_liPerfFreq.QuadPart);
}

void test()
{
	static long num_steps = 1000000;
	double step, pi;
	double x, sum=0.0;
	int i=0;
	step = 1.0/(double)num_steps;
	getStartTime();

//#pragma omp parallel for private(x) reduction(+:sum)
	for (i=0; i<num_steps; i++)
	{
		x = (i+0.5)*step;
		sum += 4.0/(1.0+x*x);
		//printf("%f,%d\n",omp_get_thread_num());
	}
	getEndTime();
	printf("%f\n",dfTim);
	pi = step * sum;
	printf ("pi=%f\n",pi);

}