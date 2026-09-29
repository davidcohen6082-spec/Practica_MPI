//   COHEN NAPOLES DAVID
//   PARDO CORREA GONZALO
//   TORRES RIVERA BRAYAN OSWALDO

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include "OperacionesArreglos.h"

using namespace std;

static const char* INTEGRANTES = "COHEN NAPOLES DAVID, PARDO CORREA GONZALO y TORRES RIVERA BRAYAN OSWALDO";

struct Contexto {
    int rank;
    int size;
    int nLocal;       
    int nTotal;        
    string equipo;    
    char* nombres;    
    bool detallado;   

    const char* nombreDe(int r) const { return nombres + r * MPI_MAX_PROCESSOR_NAME; }
};

template <typename T>
void distribuirSG(const Contexto& c, T* global, T* local, MPI_Datatype tipo, const char* etiqueta) {
    if (c.detallado && c.rank == 0) {
        for (int dest = 0; dest < c.size; dest++) {
            int ini = dest * c.nLocal;
            printf("[Equipo: %s] [Proceso MPI: 0] [Scatter %s] Seccion [%d - %d] -> Proceso MPI %d (Equipo: %s)\n",
                c.equipo.c_str(), etiqueta, ini, ini + c.nLocal - 1, dest, c.nombreDe(dest));
        }
        fflush(stdout);
    }

    MPI_Scatter(global, c.nLocal, tipo, local, c.nLocal, tipo, 0, MPI_COMM_WORLD);

    if (c.detallado) {
        printf("[Equipo: %s] [Proceso MPI: %d] [Scatter %s] Seccion [%d - %d] recibida\n",
            c.equipo.c_str(), c.rank, etiqueta, c.rank * c.nLocal, c.rank * c.nLocal + c.nLocal - 1);
        fflush(stdout);
    }
}

template <typename T>
void recolectarSG(const Contexto& c, T* global, T* local, MPI_Datatype tipo, const char* etiqueta) {
    if (c.detallado) {
        printf("[Equipo: %s] [Proceso MPI: %d] [Gather %s] Seccion [%d - %d] -> Proceso MPI 0\n",
            c.equipo.c_str(), c.rank, etiqueta, c.rank * c.nLocal, c.rank * c.nLocal + c.nLocal - 1);
        fflush(stdout);
    }

    MPI_Gather(local, c.nLocal, tipo, global, c.nLocal, tipo, 0, MPI_COMM_WORLD);

    if (c.detallado && c.rank == 0) {
        for (int origen = 0; origen < c.size; origen++) {
            int ini = origen * c.nLocal;
            printf("[Equipo: %s] [Proceso MPI: 0] [Gather %s] Seccion [%d - %d] recibida del proceso MPI %d (Equipo: %s)\n",
                c.equipo.c_str(), etiqueta, ini, ini + c.nLocal - 1, origen, c.nombreDe(origen));
        }
        fflush(stdout);
    }
}

template <typename T>
void imprimirArreglo(const string& nombre, T* arr, int n) {
    cout << "Arreglo " << nombre << ": [ ";
    for (int i = 0; i < n; i++) {
        cout << arr[i];
        if (i < n - 1) cout << ", ";
    }
    cout << " ]" << endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) cout << INTEGRANTES << endl;

    char nombreEquipo[MPI_MAX_PROCESSOR_NAME] = { 0 };
    int longNombre = 0;
    MPI_Get_processor_name(nombreEquipo, &longNombre);

    char* nombres = nullptr;
    if (rank == 0) nombres = new char[size * MPI_MAX_PROCESSOR_NAME];
    MPI_Gather(nombreEquipo, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
        nombres, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    printf("[Equipo: %s] [Proceso MPI: %d de %d] [%s] Hilos OpenMP disponibles: %d\n",
        nombreEquipo, rank, size, rank == 0 ? "MAESTRO" : "TRABAJADOR", omp_get_max_threads());
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    int n = 0;
    if (rank == 0) {
        while (n <= 0) {
            cout << "Ingrese el tamano de los arreglos: " << flush;
            if (!(cin >> n) || n <= 0) {
                n = 0;
                cin.clear();
                cin.ignore(10000, '\n');
                cout << "Valor invalido." << endl;
            }
        }
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Contexto c;
    c.rank = rank;
    c.size = size;
    c.nTotal = n;
    c.nLocal = (n + size - 1) / size;   
    c.equipo = nombreEquipo;
    c.nombres = nombres;
    c.detallado = (n <= 100);           

    int nPad = c.nLocal * size;
    int offset = rank * c.nLocal;

    OperacionesArreglos ops;

    int* A = nullptr;
    int* B = nullptr;
    long long* R = nullptr;
    if (rank == 0) {
        A = new int[nPad];
        B = new int[nPad];
        R = new long long[nPad];
    }
    int* localA = new int[c.nLocal];
    int* localB = new int[c.nLocal];
    long long* localR = new long long[c.nLocal];

    if (rank == 0) {
        cout << "\nTamano: " << n << " | Procesos: " << size << " | Elementos por proceso: " << c.nLocal;
        if (nPad != n) cout << " (relleno de " << (nPad - n) << " elementos)";
        cout << "\nEquipos participantes:" << endl;
        for (int i = 0; i < size; i++)
            cout << "  Proceso MPI " << i << " -> " << c.nombreDe(i) << endl;
    }

    bool creados = false;
    int opcion = 0;
    cout << fixed << setprecision(6);

    do {
        if (rank == 0) {
            cout << "\nMENU:" << endl;
            cout << "1. Crear arreglos" << endl;
            cout << "2. Sumar arreglos" << endl;
            cout << "3. Restar arreglos" << endl;
            cout << "4. Multiplicar arreglos" << endl;
            cout << "5. Cuadrado de un arreglo (A)" << endl;
            cout << "6. Salir" << endl;
            cout << "Seleccione una opcion: " << flush;
            if (!(cin >> opcion)) {
                cin.clear();
                cin.ignore(10000, '\n');
                opcion = 0;
            }
        }
        MPI_Bcast(&opcion, 1, MPI_INT, 0, MPI_COMM_WORLD);

        switch (opcion) {
        case 1: {
            unsigned long long semilla = (rank == 0) ? (unsigned long long)time(NULL) : 0ULL;
            MPI_Bcast(&semilla, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();

            ops.crearArregloMPI(localA, c.nLocal, offset, n, c.detallado ? ASCENDENTE : ALEATORIO,
                semilla, "A", c.equipo, rank, c.detallado);
            ops.crearArregloMPI(localB, c.nLocal, offset, n, c.detallado ? DESCENDENTE : ALEATORIO,
                semilla + 1, "B", c.equipo, rank, c.detallado);

            recolectarSG(c, A, localA, MPI_INT, "A");
            recolectarSG(c, B, localB, MPI_INT, "B");

            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            creados = true;

            if (rank == 0) {
                cout << "Arreglos creados." << endl;
                if (c.detallado) {
                    imprimirArreglo("A", A, n);
                    imprimirArreglo("B", B, n);
                }
                cout << "Tiempo de ejecucion (Crear arreglos): " << (t1 - t0) << " segundos" << endl;
            }
            break;
        }

        case 2:
        case 3:
        case 4:
        case 5: {
            if (!creados) {
                if (rank == 0) cout << "Primero debe crear los arreglos (opcion 1)." << endl;
                break;
            }
            const char* nombreOp = (opcion == 2) ? "Suma" : (opcion == 3) ? "Resta" :
                (opcion == 4) ? "Multiplicacion" : "Cuadrado";

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();

            distribuirSG(c, A, localA, MPI_INT, "A");
            if (opcion != 5) distribuirSG(c, B, localB, MPI_INT, "B");

            if (opcion == 2)      ops.sumar(localA, localB, localR, c.nLocal, offset, n, c.equipo, rank, c.detallado);
            else if (opcion == 3) ops.restar(localA, localB, localR, c.nLocal, offset, n, c.equipo, rank, c.detallado);
            else if (opcion == 4) ops.multiplicar(localA, localB, localR, c.nLocal, offset, n, c.equipo, rank, c.detallado);
            else                  ops.cuadrado(localA, localR, c.nLocal, offset, n, c.equipo, rank, c.detallado);

            recolectarSG(c, R, localR, MPI_LONG_LONG, "R");

            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();

            if (rank == 0) {
                if (c.detallado) imprimirArreglo(string("R (") + nombreOp + ")", R, n);
                cout << "Tiempo de ejecucion (" << nombreOp << "): " << (t1 - t0) << " segundos" << endl;
            }
            break;
        }

        case 6:
            if (rank == 0) cout << "Saliendo..." << endl;
            break;

        default:
            if (rank == 0) cout << "Opcion invalida, intente de nuevo." << endl;
            break;
        }

        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

    } while (opcion != 6);

    delete[] localA;
    delete[] localB;
    delete[] localR;
    if (rank == 0) {
        delete[] A;
        delete[] B;
        delete[] R;
        delete[] nombres;
    }

    MPI_Finalize();

    if (rank == 0) cout << INTEGRANTES << endl;
    return 0;
}