/**
 * @author Santiago Millán Giner
 * @file secuencial.cpp
 */

#include <vector>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <time.h>
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
 * Calcula T(p) para todas las particulas p
 * Calcula E(x,y) para todas la superficie NxN
 */
void calc() {
    N2 = N*N;

    float EA_pxy_; //< guarda el resultado de EA de forma temporal para un valor de p,x,y determinado

    for (unsigned X=0; X<N; X++)
        for (unsigned Y=0; Y<N; Y++)
            E[X][Y] = 0;

   /* for (unsigned p=0; p<P; p++) {
        float Tp = 0.0;
    }*/

    // T Auxiliar es una matiz de floats de nthreads x P
    // en ella se almacenan los resultados parciales de T[p] con cada hilo, para sumarse y obtener el final
    float** T_aux;

    #pragma omp parallel
    {
        unsigned nthreads = omp_get_num_threads(); // Número de hilos
        int tid = omp_get_thread_num();            // Identidicador del hilo

        #pragma omp single
        {
            //cout << "nthreads = " << nthreads << endl;

            T_aux = new float* [nthreads];
            /*
            for(int i=0; i<nthreads; i++) {
                T_aux[i] = new float[P];
                for (unsigned p=0; p<P; p++)
                    T_aux[i][p] = 0;
            }
            */
        }

        T_aux[tid] = new float[P];
        for (unsigned p=0; p<P; p++)
            T_aux[tid][p] = 0;

        /*
        // Mostrar hilos después de la inicialización
        #pragma omp critical
        {
            cout << "Hola soy el hilo " << tid << endl;
            for (unsigned p=0; p<P; p++)
                cout << T_aux[tid][p] << endl;
            cout << endl;
        };
        */

        #pragma omp for schedule(static) collapse(2) private(EA_pxy_)
        for (unsigned X=0; X<N; X++) {  // paralelo
            for (unsigned Y=0; Y<N; Y++) {  // paralelo
                for (unsigned p=0; p<P; p++) {
                    EA_pxy_ = EA(p,X,Y);
                    T_aux[tid][p] += EA_pxy_;
                    E[X][Y] += EA_pxy_;
                }
            }
        }

        /*
        // Mostrar resultados parciles
        #pragma omp critical
        {
            cout << "Adios soy el hilo " << tid << endl;
            for(int p=0; p<P; p++)
                cout <<T_aux[tid][p] <<endl;
            cout<<endl;
        };
         */

        #pragma omp for schedule(static)
        for (unsigned p=0; p<P; p++){
            for (unsigned th=0; th<nthreads; th++) {
                T[p] += T_aux[th][p];
            }
        }

        /* Borrar T_aux
        delete[] T_aux[tid];
        #pragma omp single
        {
            delete [] T_aux;
        }*/
    }
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
    for (unsigned i=0; i<N; i++)
        E[i] = &(E_mem[i*N]); // Cada posición de E[i] apunta a la primera posición de cada fila de N elementos
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
 * Realiza una simulación con valores de energía de 100 a 101000
 * Y posiciones dentro de la superficie [0..N-1] x [0..N-1]
 * ¡ Recuerda que N y P deben de estar inicializadas !
 */
void experimentacion() {

    init_arrays();

    for (unsigned p=0; p<P; p++) {
        x[p] = (unsigned) ((N - 1) * (rand() / (float) RAND_MAX) + 0.5);
        y[p] = (unsigned) ((N - 1) * (rand() / (float) RAND_MAX) + 0.5);
        e[p] = 100 + 10000 * (rand() / (float) RAND_MAX);
    }
	omp_set_num_threads(8);
    double t0 = get_time();
    calc();
    double t1 = get_time();

    cout << "Tiempo de ejecución = " << t1 - t0 << "s" << endl;

    // Borra los arrays dinámicos, para que no influyan en otros experimentos
    delete[] x;
    delete[] y;
    delete[] e;

    delete[] T;
    delete[] E;
}

/**
 * Función con la finalidad para comprobar que el procedimiento de caluclo es correcto.
 *
 * Calcula T(p) para todas las particulas p
 * y E(x,y) para toda la superficie NxN
 * para la siguente muestra de datos:
 *
 * (x1 , y1 , e1 ) = (0, 0, 100)
 * (x2 , y2 , e2 ) = (2, 0, 200)
 * (x3 , y3 , e3 ) = (3, 2, 50)
 *
 * Posteiromente se muestra por pantalla.
 */
void depuracion() {
    N = 4;
    P = 3;

    init_arrays();

    x[0]=0;     y[0]=0;     e[0]=100.0;
    x[1]=2;     y[1]=0;     e[1]=200.0;
    x[2]=3;     y[2]=2;     e[2]= 50.0;

    calc();

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

/**
 * Realiza mulitples experimentos, con las tallas de las tablas
 */
void multiple() {
    cout << "MODO MULTIPLE" << endl;

    vector<unsigned> tallasP ({25,50,100});
    vector<unsigned> tallasN ({100,1000, 2500, 5000, 7500, 10000});

    for(auto tallaP :tallasP) {
        for(auto tallaN: tallasN) {
            P = tallaP;
            N = tallaN;
            cout <<"P = "<< P;
            cout <<"N = "<< N;
            experimentacion();
        }
    }
}

/**
 * Función principal.
 * Introduce un argumento al comando para ejecutarlo en modo multiple.
 * Si no se pone dicha opción te deja elegir entre el modo depuraacion o experimentación.
 * NOTA:
 *   El modo multiple esta pensado para ser ejecutado con un script, mientras que el normal esta pensado para ser
 *   ejecutado manulmente.
 * @return devuelve 0 si el programa finaliza con éxito
 */
int main(int argc, char* argv[]) {
    if (argc > 1) {
        multiple();
    } else {
        cout << "Introudce 0 para el MODO DEPURACION" << endl;
        cout << "Intorduce 1 para el MODO EXPERIMENTACION" << endl;
        cout << "> ";

        // modo_exp indica si esta en modo experimentación
        bool modo_exp;
        cin >> modo_exp;

        if (modo_exp) {
            cout << "MODO EXPERIMENTACION" << endl;
            cout << "Introduce los siguientes valores" << endl;
            cout << "P = "; cin >> P;
            cout << "N = "; cin >> N;
            experimentacion();
        }
        else {
            cout << "MODO DEPURACION" << endl;
            depuracion();
        }
    }

    return 0;
}
