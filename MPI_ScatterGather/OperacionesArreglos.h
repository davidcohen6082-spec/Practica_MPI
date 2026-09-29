//   COHEN NAPOLES DAVID
//   PARDO CORREA GONZALO
//   TORRES RIVERA BRAYAN OSWALDO

#pragma once

#include <cstdio>
#include <string>
#include <omp.h>

using namespace std;

enum TipoLlenado { ASCENDENTE, DESCENDENTE, ALEATORIO };

// Todas las funciones trabajan UNICAMENTE con la seccion local del proceso MPI que las ejecuta.
//   nLocal       : elementos de la seccion local
//   offsetGlobal : posicion global del primer elemento de la seccion (rank * nLocal)
//   nTotal       : tamano real del arreglo global (para no imprimir el relleno)
//   equipo, rank : identificacion del equipo y del proceso MPI (para los mensajes)
//   detallado    : true = imprime equipo + proceso MPI + hilo OpenMP + posicion + operacion
class OperacionesArreglos {
public:
    static const int MIN_ALEATORIO = 1;
    static const int MAX_ALEATORIO = 1000000;

    // Genera los valores de la seccion local de un arreglo.
    // ASCENDENTE/DESCENDENTE: valores consecutivos (primera ejecucion).
    // ALEATORIO: valores en [1, 1,000,000] (segunda ejecucion), seguro entre hilos.
    void crearArregloMPI(int* arr, int nLocal, int offsetGlobal, int nTotal,
                         TipoLlenado tipo, unsigned long long semilla,
                         const char* nombreArreglo, const string& equipo,
                         int rank, bool detallado) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            int pos = offsetGlobal + i;
            int valor = 0;
            if (pos < nTotal) {
                if (tipo == ASCENDENTE)
                    valor = pos + 1;
                else if (tipo == DESCENDENTE)
                    valor = nTotal - pos;
                else
                    valor = MIN_ALEATORIO + (int)(splitmix64(semilla + (unsigned long long)pos) %
                            (unsigned long long)(MAX_ALEATORIO - MIN_ALEATORIO + 1));
            }
            arr[i] = valor;

            if (detallado && pos < nTotal) {
                int hilo = omp_get_thread_num();
#pragma omp critical(salida_detallada)
                {
                    printf("[Equipo: %s] [Proceso MPI: %d] [Hilo OpenMP: %d] [Posicion: %d] [Operacion: Crear %s] Valor generado: %d\n",
                        equipo.c_str(), rank, hilo, pos, nombreArreglo, valor);
                    fflush(stdout);
                }
            }
        }
    }

    void sumar(int* A, int* B, long long* R, int nLocal, int offsetGlobal, int nTotal,
               const string& equipo, int rank, bool detallado) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            R[i] = (long long)A[i] + B[i];
            int pos = offsetGlobal + i;
            if (detallado && pos < nTotal)
                detalleBinario(equipo, rank, pos, "Suma", A[i], B[i], '+', R[i]);
        }
    }

    void restar(int* A, int* B, long long* R, int nLocal, int offsetGlobal, int nTotal,
                const string& equipo, int rank, bool detallado) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            R[i] = (long long)A[i] - B[i];
            int pos = offsetGlobal + i;
            if (detallado && pos < nTotal)
                detalleBinario(equipo, rank, pos, "Resta", A[i], B[i], '-', R[i]);
        }
    }

    // El resultado es long long porque 1,000,000 * 1,000,000 no cabe en un int.
    void multiplicar(int* A, int* B, long long* R, int nLocal, int offsetGlobal, int nTotal,
                     const string& equipo, int rank, bool detallado) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            R[i] = (long long)A[i] * B[i];
            int pos = offsetGlobal + i;
            if (detallado && pos < nTotal)
                detalleBinario(equipo, rank, pos, "Multiplicacion", A[i], B[i], '*', R[i]);
        }
    }

    void cuadrado(int* A, long long* R, int nLocal, int offsetGlobal, int nTotal,
                  const string& equipo, int rank, bool detallado) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            R[i] = (long long)A[i] * A[i];
            int pos = offsetGlobal + i;
            if (detallado && pos < nTotal) {
                int hilo = omp_get_thread_num();
#pragma omp critical(salida_detallada)
                {
                    printf("[Equipo: %s] [Proceso MPI: %d] [Hilo OpenMP: %d] [Posicion: %d] [Operacion: Cuadrado] %d ^ 2 = %lld\n",
                        equipo.c_str(), rank, hilo, pos, A[i], R[i]);
                    fflush(stdout);
                }
            }
        }
    }

private:
    // Generador pseudoaleatorio sin estado compartido: cada posicion global produce
    // siempre el mismo valor para una semilla, sin importar que hilo o proceso lo calcule.
    static unsigned long long splitmix64(unsigned long long x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    static void detalleBinario(const string& equipo, int rank, int pos, const char* operacion,
                               int a, int b, char simbolo, long long r) {
        int hilo = omp_get_thread_num();
#pragma omp critical(salida_detallada)
        {
            printf("[Equipo: %s] [Proceso MPI: %d] [Hilo OpenMP: %d] [Posicion: %d] [Operacion: %s] %d %c %d = %lld\n",
                equipo.c_str(), rank, hilo, pos, operacion, a, simbolo, b, r);
            fflush(stdout);
        }
    }
};
