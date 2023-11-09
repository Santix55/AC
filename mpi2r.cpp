/**
 * Implementación de la simulación en MPI basada en la paralelización de OpenMP
 * @author Santiago Millán Giner
 * @file secuencial.cpp
 */
#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <time.h>
#include <cstdlib>
using namespace std;

// VARABLES QUE DETERMINAN LA TALLA DEL PROBLEMA //
unsigned N;     // ancho/alto de la superficie cuadrada
unsigned P;     // número de particulas de muestra

// ARRAYS DINÁMICOS DE ENTRADA DE DATOS DE PARTICULAS //
unsigned* x;    // posiciones de las particulas en eje X
unsigned* y;    // posiciones de las particulas en eje Y
float* e;       // energía de las particulas (siempre positivo)

// ARRAYS DINÁMICOS DE RESULTADOS //
float*  T;      // soluciones de T para todas las particulas
float** E;      // soluciones de E para todas las posiciones x∈[0..N[ y∈[0..N[

// VARIABLES AUXILIARES //
unsigned N2;    // variable auxiliar que guarda N² para evitar calcularla varias veces

/**
 * Calcula y devuelve A(p,x,y)
 * @param p Índice de la particula
 * @param X Posición de el punto en el eje X sobre el cual se quiere calcular la distancia
 * @param Y Posición de el punto en el eje Y sobre el cual se quiere calcular la distancia
 * @return Factor de atenuación
 */
float A(const unsigned p, const unsigned X, const unsigned Y) {
    // El cast a entero de unsigned evita el underflow
    float dx = ((int) X) - ((int)x[p]);    dx *= dx;  // dx = (x-xp)²
    float dy = ((int) Y) - ((int)y[p]);    dy *= dy;  // dy = (y-yp)²
    return exp(sqrt(dx+dy));
}

/**
 * Calcula y devuelve EA(p,x,y)
 * @param p Índice de la particula
 * @param X Posición de X del puto sobre que se quiere calcular la distancia
 * @param Y Posición de Y del puto sobre que se quiere calcular la distancia
 * @return Energía acumlada de la particula p respecto al punto (x,y)
 */
float EA(const unsigned p, const unsigned X, const unsigned Y) {
    return e[p] / ( A(p,X,Y)*N2 );
}

/**
 * Inicializa todos los arrays dinámicos globales que existen en el programa
 */
void init_arrays() {
    x = new unsigned [P];
    y = new unsigned [P];
    e = new float [P];

    T = new float [P];

    // los resultados de E se guardan en una matriz NxN de posiciones de memoria consecutivas
    float* E_mem = (float *) malloc (sizeof(float) * N * N); // Reserva N² posiciones de floats
    E = (float **) malloc (sizeof(float*) * N);              // Reserva N arrarys de float
    for (int i=0; i<N; i++)
        E[i] = &(E_mem[i*N]); // Cada posición de E[i] apunta a la primera posición de cada fila de N elementos
	
	// Instaciar todos los sumatorios a 0
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
			E[i][j] = 0;
	
	for(int p=0; p<P; p++)
		T[p] = 0;
}

/**
 * (Código del enunciado)
 * Obtiene el instante de tiempo actual en segundos
 * @return Instnte de tiempo actual en segundos
 */
double get_time() {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (double) t.tv_sec + ((double) t.tv_nsec)/1e9;
}


/**
 * Si se le pasan 2 el primero instancia N y el segundo P
 * en caso contrario, ejecuta los datos de ejemplo
 * @return devuelve 0 si el programa finaliza con éxito
 */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
	
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	double t0, t1;
	
	// CARGAR DATOS DE ENTRADA
	if(rank==0) {
		if (argc>=3) {
			N = atoi(argv[1]);
			P = atoi(argv[2]);
			
			init_arrays();
			
			for (unsigned p=0; p<P; p++) {
				x[p] = (unsigned) ((N - 1) * (rand() / (float) RAND_MAX) + 0.5);
				y[p] = (unsigned) ((N - 1) * (rand() / (float) RAND_MAX) + 0.5);
				e[p] = 100 + 10000 * (rand() / (float) RAND_MAX);
			}
		}
		else {
			cout << "MODO DEPURACION" << endl;

			N = 4;
			P = 3;

			init_arrays();

			x[0]=0;     y[0]=0;     e[0]=100.0;
			x[1]=2;     y[1]=0;     e[1]=200.0;
			x[2]=3;     y[2]=2;     e[2]= 50.0;
		}
		
		t0 = get_time();
	}
	
	// CÁLCULO // Estrategia: Dividir T y reducir 
	
	// Pasar datos de entrada
	MPI_Bcast (&N, 1, MPI_INT,
	           0, MPI_COMM_WORLD);

	MPI_Bcast (&P, 1, MPI_INT,
	           0, MPI_COMM_WORLD);

	N2 = N*N;

	if(rank!=0) {
		x = new unsigned[P];
		y = new unsigned[P];
		e = new float[P];
	}
	
	MPI_Bcast (x, P, MPI_INT,
	           0, MPI_COMM_WORLD);
	
	
	MPI_Bcast (y, P, MPI_INT,
	           0, MPI_COMM_WORLD);
	  
	MPI_Bcast (e, P, MPI_FLOAT,
	           0, MPI_COMM_WORLD);
	
	const int lN = N/size;	//< números de valores de X locales
	const int lN2 = lN*N;

	// Reservar memoria para los procesos locales
	float* lT = new float[P];
	
	float* lE_mem = new float[lN2];
	float** lE = new float* [lN];
	for(int i=0; i<lN; i++)
		lE[i] = &lE_mem[i*N];
	
	for(int p=0; p<P; p++)
		lT[p] = 0;
	
	for(int X=0; X<lN; X++)
		for(int Y=0; Y<N; Y++)
			lE[X][Y] = 0;

	// Cálculo paralelo
    for (unsigned X=0; X<lN; X++)
        for (unsigned Y=0; Y<N; Y++)
			for(unsigned p=0; p<P; p++){

				float dx = ((int) X+rank*lN) - ((int)x[p]);    dx *= dx;  // dx = (x-xp)²
				float dy = ((int) Y) - ((int)y[p]);            dy *= dy;  // dy = (y-yp)²
				float A_pxy_ = exp(sqrt(dx+dy));
				
				float EA_pxy_ = e[p] / ( A_pxy_*N2 );
				
                lT[p] += EA_pxy_;
                lE[X][Y] += EA_pxy_;
			}

	

	MPI_Gather (lE_mem, lN2, MPI_FLOAT,
			    rank==0? E[0]:NULL, lN2, MPI_FLOAT,
				0, MPI_COMM_WORLD);
	
	MPI_Reduce(lT, T , P, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	
	
	// MOSTRAR RESULTADOS
	if(rank==0) {
		double t1 = get_time();
		cout << "Tiempo de ejecución = " << t1 - t0 << "s" << endl;
		
		// mostrar valores finales solo en modo prueba
		if(argc < 3) {
			cout << "T(p)" << endl;
			for (unsigned p=0; p<P; p++)
				cout << "T(" << p << ") = " << T[p] << endl;

			cout<< endl;

			cout << "E(x,y)" << endl;
			for (unsigned i=0; i<N; i++) {
				for (unsigned j=0; j<N; j++)
					cout << fixed << setprecision(2) <<E[j][i] << '\t';
				cout << endl;
			}
		}
	}
	
	MPI_Finalize();
	
    return 0;
}
