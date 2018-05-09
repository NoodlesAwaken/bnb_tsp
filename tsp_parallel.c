//
//  tsp_parallel.c
//  PDSassign3
//
//  Created by Gregory Min on 3/5/18.
//  Copyright © 2018 Gregory Min. All rights reserved.
//

#include "tsp_parallel.h"
#include <stdio.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

static int distance[CITY * CITY];       // distance matrix represented as array
static int size;                        // number of cities
static int current_bound = INT_MAX;     // upper bound initialized int max
static int route[CITY];                 // stores the current visiting sequence
static int final[CITY];                 // stores best solution find so far
static int cost = INT_MAX;              // current cost of best solution
static int workers = 1;                 // counter of worker processes
static int log = 0;                     // argument log flag

int main(int argc, const char * argv[]) {
    load_matrix(argv[1]);
    if (argc > 2)                       // if log is presented then produce log
        if (!strcmp(argv[2], "log")) log = 1;

    MPI_Init(NULL, NULL);
    
    time_t start, end;
    // start timer
    start = clock();
    
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    // calculating how deep to go in the search tree
    int jobs = size - 1;
    int depth = 1;
    int nodes[size];                    // number of nodes in different level
    nodes[0] = 1;                       // root has only one node
    while (jobs < world_size - 1) {     // while subsets of jobs less than the number of processes
        nodes[depth] = nodes[depth - 1] * (size - depth);
        depth++;                        // which level we need to explore in search tree
        jobs *= (size - depth);         // number of subsets
    }
    
    int expend = 0;                     // number of nodes we need to explore
    for (int i = 0; i < depth; i++) expend += nodes[i];
    
    // build a stack to store the decomposed jobs
    struct Node * job_queue = malloc(sizeof(struct Node));
    job_queue->next = NULL;
    job_queue->end = job_queue;
    int visit[size];                        // visited city marked as 1, non-visited 0
    memset(visit, 0, size * sizeof(int));
    memset(route, 0, size * sizeof(int));
    visit[0] = 1;                           // starting from city 1, mark it as visited
    memcpy(job_queue->data, visit, size * sizeof(int));
    memcpy(job_queue->data + size, route, size * sizeof(int));
    // this is the first node in the serach tree
    job_queue->data[size * 2] = 0;          // city number
    job_queue->data[size * 2 + 1] = 0;      // tree level
    job_queue->data[size * 2 + 2] = 0;      // route cost
    
    // breadth-first-search algorithm to explore first 'expend' nodes in search tree
    for (int i = 0; i < expend; i++) {
        for (int j = 0; j < size; j++) {    // for each non-visited city
            if (job_queue->data[j] == 0) {  // create a new node in the job queue
                struct Node * new_node = malloc(sizeof(struct Node));
                new_node->next = NULL;
                new_node->end = new_node;
                memcpy(new_node->data, job_queue->data, size * 2 * sizeof(int));
                new_node->data[j] = 1;      // mark this city as visited
                // mark this city on the route
                new_node->data[size + new_node->data[size * 2 + 1] + 1] = j;
                new_node->data[size * 2] = j;
                new_node->data[size * 2 + 1]++;
                new_node->data[size * 2 + 2] += get_dis(j, job_queue->data[size * 2]);
                job_queue->end->next = new_node;
                job_queue->end = new_node;
            }
        }
        // explored one node then pop it out
        job_queue->next->end = job_queue->end;
        job_queue = job_queue->next;
    }
    
    /* Distribute jobs between each worker process. The index array marks
     the starting and ending index of each worker process in the job queue.
     e.g. 15 cities, 3 worker processes, there will be 14 nodes to be explored.
     The index = {0, 5, 10, 14}; worker 1 will take 0~4, worker 2 will take 5~9
     and worker 3 take 10~13 from the job queue */
    int index[world_size];
    index[0] = 0;
    int rmd = jobs%(world_size - 1);
    for (int i = 1; i < world_size; i++) {
        index[i] = index[i - 1] + jobs/(world_size - 1) + ((rmd > 0)? 1 : 0);
        rmd--;
    }

    /* master process works like a server socket, listening to any incoming
     requests. If it's an update request then it will update and send to other
     worker processes. If it's a solution then check if it is better than current
     solution. If it's a finish signal means that worker process has finished
     its job, worker counter increases one. Once all the worker processes have
     finished then master process can exit */
    if (world_rank == 0) {
        while (1) {
            master_update(world_size);

            if (workers == world_size)
                break;
        }
//        show_result();
        
        // stop timer
        end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("%.5f\n", time_spent);  // show execution time /*execution time: */
    } else {
        // pop job queue untill this worker process's starting index
        for (int i = 0; i < index[world_rank - 1]; i++)
            job_queue = job_queue->next;
        
        // loop through starting index to finishing index explore each node
        for (int i = index[world_rank - 1]; i < index[world_rank]; i++) {
            int new_visit[size];
            memcpy(route, job_queue->data + size, size * sizeof(int));
            memcpy(new_visit, job_queue->data, size * sizeof(int));
            tsp(job_queue->data[size * 2 + 2], job_queue->data[size * 2 + 1], new_visit);
            job_queue = job_queue->next;
        }

        int finish = 1;
        // send finish signal of this worker process
        MPI_Send(&finish, 1, MPI_INT, 0, update_tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

// master update current bound
void master_update(int worlds) {
    MPI_Status status;
    int update_flag = 0;
    MPI_Iprobe(MPI_ANY_SOURCE, update_tag, MPI_COMM_WORLD, &update_flag, &status);
    
    if (update_flag) {
        int rcv_bound[size + 2];
        MPI_Recv(&rcv_bound, size + 2, MPI_INT, MPI_ANY_SOURCE, update_tag, MPI_COMM_WORLD, &status);
        if (rcv_bound[0] == 1) {
            workers++;
            return;
        }
        if (rcv_bound[1] < current_bound) {
            current_bound = rcv_bound[1];
            cost = rcv_bound[1];
            memcpy(final, rcv_bound + 2, size * sizeof(int));
            for (int i = 1; i < worlds; i++)
                MPI_Send(&current_bound, 1, MPI_INT, i, update_tag, MPI_COMM_WORLD);
        }
    }
}

// get the distance between city a and b
int get_dis(int city_a, int city_b) {
    return (city_a > city_b) ? distance[city_a * size + city_b] : distance[city_b * size + city_a];
}

// load the distance matrix from txt file
void load_matrix(const char * name) {
    FILE *file;
    char *line = NULL;
    char *token = NULL;
    size_t len = 0;
    file = fopen(name, "r");
    getline(&line, &len, file);                 // get the first line: number of cities
    size = atoi(line);
    memset(distance, 0, size * size * sizeof(int));
    for (int i = 1; i < size; i++) {
        getline(&line, &len, file);
        token = strtok(line, " ");
        for (int j = 0; j < i; j++) {
            distance[i * size + j] = atoi(token);
            token = strtok(NULL, " ");
        }
    }
}

// branch and bound algorithm recursive
void tsp(int current_weight, int level, int v[size]) {
    if (level == size - 1) {
        if (current_weight < current_bound) {
            cost = current_weight;
            current_bound = current_weight;
            memcpy(final, route, size * sizeof(int));
            int result[size + 2];
            result[0] = 0;
            result[1] = cost;
            memcpy(result + 2, route, size * sizeof(int));
            // send the best solution from this worker process to master
            MPI_Send(&result, size + 2, MPI_INT, 0, update_tag, MPI_COMM_WORLD);
        }
        return;
    }
    int t[size];
    
    worker_update();
    for (int i = 0; i < size; i++) {
        if (v[i] == 0) {
            int tmp = current_weight + get_dis(route[level], i);
            
            if (tmp < current_bound) {
                memcpy(t, v, size * sizeof(int));
                route[level + 1] = i;
                t[i] = 1;
                tsp(tmp, level + 1, t);
            } else {
                if (log) {
                    printf("prune branch: ");
                    for (int i = 0; i < level + 1; i++)
                        printf("%d ", route[i] + 1);
                    printf("%d\n", i + 1);
                }
            }
        }
    }
}

// worker update current bound
void worker_update() {
    int bound_flag = 0;
    MPI_Status status;
    MPI_Iprobe(0, update_tag, MPI_COMM_WORLD, &bound_flag, &status);
    if (bound_flag) {
        int new_bound;
        MPI_Recv(&new_bound, 1, MPI_INT, 0, update_tag, MPI_COMM_WORLD, &status);
        if (new_bound < current_bound) current_bound = new_bound;
    }
}

// display result: best route
void show_result() {
    printf("best route: %d", final[0] + 1);
    for (int i = 1; i < size; i++)
        printf("->%d", final[i] + 1);
    printf("\nDistance: %d\n", cost);
}
